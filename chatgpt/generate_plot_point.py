import openai
import re
import os
import json
import pandas as pd
import logging
from datetime import datetime
import argparse
import tiktoken
import nltk
import random
from nltk.tokenize import sent_tokenize

# generate plot point of story through openai api

# load config file
with open('config.json', 'r', encoding='utf8') as f:
    config = json.load(f)

# openai api config
openai.organization = config['openai']['organization']
openai.api_key = config['openai']['api_key']
openai_gpt_model = config['openai']['gpt_model']

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--num", type=int, help="number of plot points", default=10)
parser.add_argument("-m", "--max", type=int, help="max score of plot points", default=10)
parser.add_argument("-s", "--story_folder", type=str, help="story file folder", default="../data/FairytaleQA_Dataset/FairytaleQA_Dataset/split_for_training/train")

# load nlp encoding
nltk.download('punkt')
gpt_encoding = tiktoken.encoding_for_model(openai_gpt_model)

# logging config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# generate plot point prompt for openai api
def generate_prompt_plot_point(story_text, num):
    prompt = f'''Please help this story by organizing the {num} key points of the plot into columns (example: [{{"id": 1, "plot_point": "first plot"}}, {{"id": 2, "plot_point": "second plot"}}]) in JSON format in order, don't provide additional information or comment.
--
{story_text}
'''
    return prompt


# save message history to json file
def save_message_history(chat_messages, folder_name):
    # save message history to json file
    dt_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    file_name = os.path.join(folder_name, f'history_{dt_string}.json')
    with open(file_name, 'w', encoding='utf8') as f:
        json.dump(chat_messages, f, ensure_ascii=False, indent=4)


# read story text from file
def read_story_text(story_name, max_token_size=-1):
    df = pd.read_csv(os.path.join(args.story_folder, f'{story_name}-story.csv'))
    df = df.sort_values(by=['section'])
    paragraph_list = [line.replace('\n', ' ') for line in df['text'].tolist()]
    # remove empty paragraph
    paragraph_list = [para for para in paragraph_list if para != '']
    story_text = '\n'.join(paragraph_list)

    # if the story is too long, delete the random sentence, until the story is short enough
    while len(gpt_encoding.encode(story_text)) > max_token_size > 0:
        # randomly choose a paragraph
        para_idx = random.randint(0, len(paragraph_list) - 1)
        # randomly choose a sentence
        para_sent_tokenize = sent_tokenize(paragraph_list[para_idx])
        sent_idx = random.randint(0, len(para_sent_tokenize) - 1)
        # delete the sentence
        paragraph_list[para_idx] = ' '.join(
            sent_tokenize(paragraph_list[para_idx])[:sent_idx] + sent_tokenize(paragraph_list[para_idx])[
                                                                 sent_idx + 1:])
        if paragraph_list[para_idx] == '':
            paragraph_list.pop(para_idx)
        story_text = '\n'.join(paragraph_list)
    return story_text


def call_chatgpt(messages):
    response = openai.ChatCompletion.create(
        model=openai_gpt_model,
        temperature=0.5,
        messages=messages
    )
    response_message = response.choices[0].message.content
    return response_message


# call chatgpt with try, only valid when response is json format.
def call_chatgpt_try_repeat(messages, maximum_try=0):
    success_flag = False
    error_count = 0
    _response_message = None
    _json_response = None
    while not success_flag:
        try:
            _response_message = call_chatgpt(messages)
            _json_response = json.loads(_response_message)
            success_flag = True
        except json.decoder.JSONDecodeError:
            logging.info(f'Invalid JSON format, try again. (error count: {error_count})')
            error_count += 1

        if error_count > maximum_try:
            logging.warning('Too many errors, exit.')
            break

    return _response_message, _json_response


def generate_plot_point(num_of_plot_point, max_score, story_name):
    story_text = read_story_text(story_name, max_token_size=-1)
    plot_point_data = []

    # STEP 1: generate plot point
    # generate plot point prompt and get response
    logging.info(f'Generate plot point for {story_name}')
    # calculate the number of tokens in the prompt (by openai)
    # if the prompt is too long, calculate the gap between the story and the max_token_size,
    # then delete the random sentence, until the prompt is short enough
    max_token_size = 4096 - (num_of_plot_point * 60)
    len_story_text = len(gpt_encoding.encode(story_text))
    messages_max_token_gap = len_story_text
    while True:
        story_text = read_story_text(story_name, max_token_size=len_story_text-messages_max_token_gap)
        prompt_plot_point = generate_prompt_plot_point(story_text, num=num_of_plot_point)
        chat_messages = [{"role": "user", "content": prompt_plot_point}]
        messages_max_token_gap = num_tokens_from_messages(chat_messages, openai_gpt_model) - max_token_size
        if messages_max_token_gap <= 0:
            break
        else:
            logging.info(f'Reduce the length of the messages (gap={messages_max_token_gap})')

    response_message, json_response = call_chatgpt_try_repeat(chat_messages, maximum_try=5)
    chat_messages.append({"role": "assistant", "content": response_message})

    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    plot_point_data.append({
        'story_name': story_name,
        'timestamp': dt_string,
        'generate_func': config['openai']['gpt_model'],
        'plot_point_nums': num_of_plot_point,
        'plot_points': json_response,
        'score': max_score,
    })

    file_name = os.path.join('plot_point', f'{story_name}.json')
    if os.path.exists(file_name):
        with open(file_name, 'r', encoding='utf8') as f:
            original_json_data = json.load(f)
    else:
        original_json_data = []
    original_json_data.append(plot_point_data[0])
    with open(file_name, 'w', encoding='utf8') as f:
        json.dump(original_json_data, f, ensure_ascii=False, indent=4)

    save_message_history(chat_messages, 'history')
    return plot_point_data


# Returns the number of tokens used by a list of messages.
def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        # gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        # Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"num_tokens_from_messages() is not implemented for model {model}.")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


if __name__ == '__main__':
    args = parser.parse_args()
    logging.info(f'config: {args}')

    # list all story files in a directory
    story_file_list = []
    for entry in os.listdir(args.story_folder):
        if os.path.isfile(os.path.join(args.story_folder, entry)) and entry.endswith('-story.csv'):
            story_file_list.append(entry.replace('-story.csv', ''))

    # check folder exist
    if not os.path.exists('plot_point'):
        os.mkdir('plot_point')
    if not os.path.exists('history'):
        os.mkdir('history')

    fail_story_list = []
    for story in story_file_list:
        try:
            plot_point_data = generate_plot_point(args.num, args.max, story)
        except Exception as e:
            logging.error(e)
            fail_story_list.append(story)

    logging.info(f'Success Generate plot point for {len(story_file_list) - len(fail_story_list)} stories.')
    if len(fail_story_list) > 0:
        logging.warning(f'Fail Generate plot point for {len(fail_story_list)} stories.')
        logging.warning(f'Fail story list: {[story for story in fail_story_list]}')
