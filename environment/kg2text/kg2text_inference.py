import os
import torch
import json
from transformers import T5Tokenizer
from kg2text_model import Kg2TextModel, Kg2TextDataset, Kg2TextTrainer
import random
import numpy as np
import argparse
import logging
import warnings
from tqdm import tqdm

# set logging format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# set the hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="model checkpoint", type=str, default='model/kg2text_model.pt')
parser.add_argument("-t", "--tokenizer", help="tokenizer checkpoint", type=str, default='t5-base')
parser.add_argument("-s", "--seed", help="random seed", type=int, default=42)
parser.add_argument("-c", "--cuda", help="cuda device", type=int, default=0, choices=[0, 1, 2])
parser.add_argument("-i", "--input", help="input file", type=str, default='../../data/kg2text/kg2text_test.json')
parser.add_argument("-st", "--summary", help="story summary file", type=str, default='../../data/summary/summary_test.json')
parser.add_argument("-o", "--output", help="output file", type=str, default='output/kg2text_test_result.json')
parser.add_argument("-b", "--batch_size", help="batch size", type=int, default=16)
args = parser.parse_args()

MODEL = args.model
TOKENIZER = args.tokenizer
SEED = int(args.seed)
CUDA = args.cuda
BATCH_SIZE = args.batch_size
INPUT_FILE = args.input
SUMMARY_FILE = args.summary
OUTPUT_FILE = args.output

logging.info(f'model ckpt: {MODEL}')
logging.info(f'lm model: {TOKENIZER}')
logging.info(f'seed: {SEED}')
logging.info(f'cuda: {CUDA}')
logging.info(f'batch size: {BATCH_SIZE}')
logging.info(f'input file: {INPUT_FILE}')
logging.info(f'summary file: {SUMMARY_FILE}')
logging.info(f'output file: {OUTPUT_FILE}')

# setting the seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# set the cuda device
os.environ['CUDA_VISIBLE_DEVICES'] = str(CUDA)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ignore the warning
warnings.filterwarnings('ignore')


# start
model_name = TOKENIZER
model_checkpoint = MODEL
model = Kg2TextModel(model_name).to(device)
model.load_state_dict(torch.load(model_checkpoint))
tokenizer = T5Tokenizer.from_pretrained(model_name)
model.eval()

def replace_after_word(text, word, replacement):
    index = text.find(word)
    if index != -1:
        substring = text[index:]
        replaced_text = text.replace(substring, replacement, 1)
        return replaced_text
    else:
        return text

dataset = Kg2TextDataset(data_path=INPUT_FILE, summary_path=SUMMARY_FILE, tokenizer=tokenizer)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

pred_data = []
dataset_idx = 0
for batch in tqdm(data_loader):
    input_ids, attention_mask, labels = batch
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels[labels == -100] = tokenizer.pad_token_id
    labels = labels.to(device)
    output_ids = model.generate(input_ids=input_ids,
                                attention_mask=attention_mask,
                                max_length=128,
                                num_beams=2)
    output_str_list = []
    for output_id in output_ids:
        output_str = tokenizer.decode(output_id, skip_special_tokens=True)
        output_str_list.append(output_str)
    # print the input and output
    for i in range(len(input_ids)):
        story_name = dataset.data[dataset_idx]['story_name']
        input_str = tokenizer.decode(input_ids[i], skip_special_tokens=True)
        output_str = output_str_list[i]
        label_str = tokenizer.decode(labels[i], skip_special_tokens=True)
        pred_data.append({
            'story_name': story_name,
            'input': replace_after_word(input_str, 'content:', '').replace('graph to text:', '').strip(),
            'output': output_str,
            'label': label_str
        })
        # print('-----------------')
        dataset_idx + 1

with open(OUTPUT_FILE, 'w') as f:
    json.dump(pred_data, f, indent=4, ensure_ascii=False)