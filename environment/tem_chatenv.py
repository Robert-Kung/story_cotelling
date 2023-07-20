import os
import json
import torch
import random
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from .tem_graph import KnowledgeGraph
from .kg2text.kg2text_model import Kg2TextModel
from .reward.ranking_model_compare_story import RankingModel
from stanza.server import CoreNLPClient
import logging
import warnings


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class StoryBotRetellEnv:
    def __init__(self, story_summary_dataset: dict, 
                 reward_model_ckpt: str, 
                 kg2text_model_ckpt: str, 
                 embedding_model_name: str, 
                 device: str,
                 corenlp_client: CoreNLPClient):
        # config
        self.device = device
        self.end_dialogue_keyword = 'The end'
        self.bot_name = 'agent1'
        self.user_name = 'agent2'
        self.penalty = 0.1
        self.dialogue_history_limit = 20
        
        # current story info
        self.current_story_name = None
        self.current_story_summary = None
        self.current_story_kg = None
        self.current_score = 0

        self.dialogue_log_list = []
        self.current_dialogue_log = None
        self.current_candidate_response_list = []
        self.current_candidate_response_kg_list = []
        
        # story summary dataset
        self.story_summary_dataset = story_summary_dataset

        # reward model
        self.reward_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        self.reward_model_device = self.device
        self.reward_model = RankingModel(self.reward_model_device)
        self.reward_model.load_state_dict(torch.load(reward_model_ckpt,
                                                     map_location=self.reward_model_device))
        self.reward_model.to(self.reward_model_device)
        self.reward_model.eval()

        # kg2text model
        self.kg2text_tokenizer = AutoTokenizer.from_pretrained('t5-base')
        self.kg2text_model_device = self.device
        self.kg2text_model = Kg2TextModel('t5-base')
        self.kg2text_model.load_state_dict(torch.load(kg2text_model_ckpt,
                                                      map_location=self.reward_model_device))
        self.kg2text_model.to(self.kg2text_model_device)
        self.kg2text_model.eval()

        # embedding model
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        self.embedding_model_device = self.device
        self.embedding_model = AutoModel.from_pretrained(embedding_model_name)
        self.embedding_model.to(self.embedding_model_device)
        self.embedding_model.eval()

        # action space
        self.action_space = ['action0', 'action1', 'action2', 'action3', 'action4']

        # observation space
        self.observation_space = self.embedding_model.config.hidden_size * (1 + len(self.action_space))

        # openie client
        self.corenlp_client = corenlp_client


    def reset(self, story_name: str, story_kg=None, story_kg_file=None):
        # 初始化(創建)一個環境
        self.current_story_name = story_name
        self.current_story_summary = self.story_summary_dataset[self.current_story_name]

        if story_kg is not None:
            self.current_story_kg = story_kg
        else:
            self.story_kg_file = story_kg_file
            self.current_story_kg = KnowledgeGraph(device=self.embedding_model_device,
                                                   model=self.embedding_model,
                                                   tokenizer=self.embedding_tokenizer,
                                                   story_kg_file=self.story_kg_file)
        self.current_story_kg.set_kg_status(None, None, None, -1)
        self.clear_dialogue_info()

    
    def step(self, action, filtered_kg=None):
        penalty = 0
        response = self.current_candidate_response_list[action]
        filtered_kg = self.current_candidate_response_kg_list[action]

        if response == '' or filtered_kg == []:
            filtered_kg, _ = self.action5()
            penalty = self.penalty
            if filtered_kg is None:
                response = ''
            else:
                response = self.kg2text_plus([filtered_kg])[0]
        
        # response_template_list = [
        #     '<|response|>'
        # ]

        # if action==0:
        #     pass
        # elif action==1:  # subject related
        #     response_template_list.append('Just like the <|subject|>, <|response|>')
        #     response_template_list.append('As you know what<|subject|> do, <|response|>')
        #     response_template_list.append("Let's talk about <|subject|>, <|response|>")
        #     response_template_list.append("Similar to <|subject|>, <|response|>")
        #     response_template_list.append("As if <|subject|>, <|response|>")
        #     response_template_list.append("As you might expect <|subject|>, <|response|>")
        #     response_template_list.append("Similar to the experience of the <|subject|>, <|response|>")
        #     response_template_list.append("The <|subject|> is similar that <|response|>")
        # elif action==2:  # relation related
        #     response_template_list.append('What could <|subject|> <|relation|>, <|response|>')
        #     response_template_list.append('Do you know <|relation|> <|object|>, <|response|>')
        #     response_template_list.append('You say <|subject|> <|relation|> <|object|>. Maybe <|response|>')
        # elif action==3:  # object related
        #     response_template_list.append('Just like the <|object|>, <|response|>')
        #     response_template_list.append('As you know what<|object|> do, <|response|>')
        #     response_template_list.append("Let's talk about <|object|>, <|response|>")
        #     response_template_list.append("Similar to <|object|>, <|response|>")
        #     response_template_list.append("As if <|object|>, <|response|>")
        #     response_template_list.append("As you might expect <|object|>, <|response|>")
        #     response_template_list.append("Similar to the experience of the <|object|>, <|response|>")
        #     response_template_list.append("The <|object|> is similar that <|response|>")
        
        # response_template = random.choice(response_template_list)
        # # TODO: adjust the fact index
        # response_template = response_template.replace('<|subject|>', self.current_discussion_fact_list[0].subject)
        # response_template = response_template.replace('<|relation|>', self.current_discussion_fact_list[0].relation)
        # response_template = response_template.replace('<|object|>', self.current_discussion_fact_list[0].object)
        # response_template = response_template.replace('<|response|>', response)
        # response = response_template

        if filtered_kg is not None:
            self.current_story_kg.set_kg_status_by_facts(filtered_kg, len(self.dialogue_log_list) + 1)

        self.current_dialogue_log = {
            'story_name': self.current_story_name,
            'idx': len(self.dialogue_log_list),
            'role': self.bot_name,
            'action': action,
            'reward': None,
            'score': None,
            'done': None,
            'dialogue': response,
            'dialogue_kg': filtered_kg,
            # 'dialogue_kg_sen_idx': None,
            'dialogue_match_kg': None,
            # 'dialogue_match_kg_sen_idx': None,
        }

        self.dialogue_log_list.append(self.current_dialogue_log)

        reward, score = self.reward(penalty=penalty)
        done, done_msg = self.done()
        self.dialogue_log_list[-1]['reward'] = reward
        self.dialogue_log_list[-1]['score'] = score
        self.dialogue_log_list[-1]['done'] = done_msg

        # return self.observation(), self.reward(penalty), self.done()
        return response, filtered_kg

    def action0(self):
        current_discussion_idx = min(self.current_discussion_idx_list)
        # 選擇後續的劇情
        filtered_kg = self.current_story_kg.filter_kg_by_sentence_idx(None, None, None, -1, current_discussion_idx)
        if filtered_kg != []:
            remain_history_length = self.dialogue_history_limit - len(self.dialogue_log_list) + 1
            remain_kg_length = len(filtered_kg)
            step = int(remain_kg_length / remain_history_length)
            filtered_kg = filtered_kg[step:step+3]
        return filtered_kg, self.current_story_kg.kg[current_discussion_idx]
    
    def action1(self):
        filtered_kg = []
        for select_fact in self.current_discussion_fact_list:
            filtered_kg = self.current_story_kg.filter_kg(select_fact.subject, None, None, -1, fuzzy=True)[:3]
            if filtered_kg != []:
                return filtered_kg, select_fact
        return filtered_kg, None
    
    def action2(self):
        filtered_kg = []
        for select_fact in self.current_discussion_fact_list:
            filtered_kg = self.current_story_kg.filter_kg(None, select_fact.relation, None, -1, fuzzy=True)[:3]
            if filtered_kg != []:
                return filtered_kg, select_fact
        return filtered_kg, None

    def action3(self):
        filtered_kg = []
        for select_fact in self.current_discussion_fact_list:
            filtered_kg = self.current_story_kg.filter_kg(None, None, select_fact.object, -1, fuzzy=True)[:3]
            if filtered_kg != []:
                return filtered_kg, select_fact
        return filtered_kg, None


    def action4(self):
        # 選擇結束
        return None, None
    

    def action5(self):
        # 隨機選擇欲回覆的fact
        remain_kg = self.current_story_kg.filter_kg(None, None, None, -1)
        if remain_kg == []:
            return remain_kg, None

        random_kg_idx = random.choice(remain_kg).sentence_idx
        filtered_kg = []
        for kg in remain_kg:
            if kg.sentence_idx == random_kg_idx:
                filtered_kg.append(kg)
        return filtered_kg, None


    def render(self, input_sentence, input_kg=None):
        topk_idx_list, topk_facts = self.current_story_kg.sentence_match_fact(input_sentence, 3)
        self.current_discussion_idx_list = topk_idx_list
        self.current_discussion_fact_list = topk_facts
        self.current_story_kg.set_kg_status_by_facts(topk_facts, len(self.dialogue_log_list))

        self.current_dialogue_log = {
            'story_name': self.current_story_name,
            'idx': len(self.dialogue_log_list),
            'role': self.user_name,
            'action': None,
            'reward': None,
            'score': None,
            'done': None,
            'dialogue': input_sentence,
            'dialogue_kg': input_kg,
            # 'dialogue_kg_sen_idx': None,
            'dialogue_match_kg': topk_facts,
            # 'dialogue_match_kg_sen_idx': None,
        }
        self.dialogue_log_list.append(self.current_dialogue_log)

        reward, score = self.reward()
        done, done_msg = self.done()
        self.dialogue_log_list[-1]['reward'] = reward
        self.dialogue_log_list[-1]['score'] = score
        self.dialogue_log_list[-1]['done'] = done_msg


    def reward(self, penalty=0, current_sentence=None):
        dialogue_history = [log['dialogue'] for log in self.dialogue_log_list]
        if current_sentence is not None:
            dialogue_history.append(current_sentence)
        
        input_ids = self.reward_tokenizer.encode('<s>' + '</s>'.join(dialogue_history),
                                                 add_special_tokens=False)
        attention_mask = [1] * len(input_ids)
        input_ids_summary = self.reward_tokenizer.encode('<s>' + self.current_story_summary + '</s>',
                                                         add_special_tokens=False)
        attention_mask_summary = [1] * len(input_ids_summary)

        # check if the sequence length exceeds the maximum length
        if len(input_ids) > self.reward_tokenizer.model_max_length:
            # if the length is exceeded, truncate the sequence
            input_ids = input_ids[:self.reward_tokenizer.model_max_length]
            attention_mask = attention_mask[:self.reward_tokenizer.model_max_length]
        else:
            # if not exceeded, fill the sequence
            padding_length = self.reward_tokenizer.model_max_length - len(input_ids)
            input_ids = input_ids + [self.reward_tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [self.reward_tokenizer.pad_token_id] * padding_length

        if len(input_ids_summary) > self.reward_tokenizer.model_max_length:
            input_ids_summary = input_ids_summary[:self.reward_tokenizer.model_max_length]
            attention_mask_summary = attention_mask_summary[:self.reward_tokenizer.model_max_length]
        else:
            padding_length = self.reward_tokenizer.model_max_length - len(input_ids_summary)
            input_ids_summary = input_ids_summary + [self.reward_tokenizer.pad_token_id] * padding_length
            attention_mask_summary = attention_mask_summary + [self.reward_tokenizer.pad_token_id] * padding_length
        
        last_state_score = self.current_score
        self.current_score = self.reward_model(torch.tensor(input_ids).unsqueeze(0).to(self.reward_model_device),
                                               torch.tensor(attention_mask).unsqueeze(0).to(self.reward_model_device),
                                               torch.tensor(input_ids_summary).unsqueeze(0).to(self.reward_model_device),
                                               torch.tensor(attention_mask_summary).unsqueeze(0).to(self.reward_model_device))[0, 0].item()
        reward = self.current_score - last_state_score - penalty
        score = self.current_score

        entity_score = 0
        if len(self.dialogue_log_list) > 1:
            entity_score = self.current_story_kg.find_two_sentence_distance(corenlp_client=self.corenlp_client,
                                                                            sentence1=self.dialogue_log_list[-1]['dialogue'],
                                                                            sentence2=self.dialogue_log_list[-2]['dialogue'])
            if entity_score > 0:
                entity_score = 0.01
        reward += entity_score

        return reward, score


    def done(self):
        done = False
        done_msg = None

        if len(self.dialogue_log_list) >= self.dialogue_history_limit:
            done = True
            done_msg = 'DHL'
        elif self.dialogue_log_list[-1]['dialogue'] == self.end_dialogue_keyword:
            done = True
            done_msg = 'EDK'
        elif self.current_story_kg.filter_kg(None, None, None, -1) == []:
            done = True
            done_msg = 'NRF'

        return done, done_msg


    def observation(self):
        # Get observation (story_kg + dialogue_history)
        history_str = '[CLS] ' + ' [SEP]'.join([log['dialogue'] for log in self.dialogue_log_list])
        history_embedding = self.get_embedding(history_str, add_special_tokens=False)

        candidate_response_list = []
        action0_kg, _ = self.action0()
        action1_kg, _ = self.action1()
        action2_kg, _ = self.action2()
        action3_kg, _ = self.action3()
        action4_kg, _ = self.action4()
        self.current_candidate_response_kg_list = [
                action0_kg, 
                action1_kg, 
                action2_kg, 
                action3_kg, 
                action4_kg
            ]
        candidate_response_list = self.kg2text_plus(self.current_candidate_response_kg_list)           
        candidate_response_list[-1] = self.end_dialogue_keyword

        candidate_response_embedding = []
        for candidate_response in candidate_response_list:
            candidate_response_str = f'[CLS] {candidate_response} [SEP]'
            str_embedding = self.get_embedding(candidate_response_str, add_special_tokens=False)
            candidate_response_embedding.append(str_embedding)
        candidate_response_embedding = torch.stack(candidate_response_embedding, dim=0)
        candidate_response_embedding = candidate_response_embedding.view(-1)
            
        # candidate_response_str = '[CLS] ' + ' [SEP]'.join(candidate_response_list)
        # candidate_response_embedding = self.get_embedding(candidate_response_str, add_special_tokens=False)

        self.current_candidate_response_list = candidate_response_list
        observation = torch.cat((history_embedding, candidate_response_embedding), dim=0)
        return observation


    def count_actions(self):
        # count each action be used
        action_count = {}
        for i in range(len(self.action_space)):
            action_count[i] = 0
        for log in self.dialogue_log_list:
            if log['action'] is not None:
                action_count[log['action']] += 1
        return action_count


    def get_embedding(self, sentence, add_special_tokens=True):
        # Get sentence embedding
        encoded_input = self.embedding_tokenizer.encode_plus(
            sentence,
            add_special_tokens=add_special_tokens,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.embedding_model_device)

        with torch.no_grad():
            outputs = self.embedding_model(**encoded_input)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
        return embedding


    def kg2text_plus(self, kg_list):
        # Convert kg to text
        prompt_list = []
        none_list = []
        for id, kg in enumerate(kg_list):
            if kg is None:
                none_list.append(id)
                t_str = '[ ]'
            else:
                t_str = ''.join([str(fact) for fact in kg])
            prompt = f'graph to text: {t_str} content: {self.current_story_summary}'
            prompt_list.append(prompt)
        
        encoding = self.kg2text_tokenizer(prompt_list,
                                          max_length=1024,
                                          pad_to_max_length=True,
                                          truncation=True,
                                          return_tensors='pt')
        input_ids = encoding.input_ids
        attention_mask = encoding.attention_mask
        outputs = self.kg2text_model.generate(input_ids=input_ids.to(self.kg2text_model_device),
                                                     attention_mask=attention_mask.to(self.kg2text_model_device),
                                                     max_length=64,
                                                     num_beams=2)
        decoded_outputs = [self.kg2text_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        for id in none_list:
            decoded_outputs[id] = ''
        return decoded_outputs


    def clear_dialogue_info(self):
        self.dialogue_log_list = []
        self.current_dialogue_log = {
            'story_name': None,
            'idx': None,
            'role': None,
            'action': None,
            'reward': None,
            'score': None,
            'done': None,
            'dialogue': None,
            'dialogue_kg': None,
            # 'dialogue_kg_sen_idx': None,
            'dialogue_match_kg': None,
            # 'dialogue_match_kg_sen_idx': None,
        }


    def export_dialogue_history(self, filename='dialogue_history.csv'):
        # export dialogue history
        df = pd.DataFrame(self.dialogue_log_list)
        # df.to_csv(filename, index=False)
        return df


    def get_sentence_embeddings(self, sentences):
        # Tokenize sentences
        encoded_input = self.embedding_tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(self.embedding_model_device)

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.embedding_model(**encoded_input)

        # Perform pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings


    def two_paragraph_similarity(self, paragraph1, paragraph2):
        paragraph1_embeddings = self.get_sentence_embeddings(paragraph1)
        paragraph2_embeddings = self.get_sentence_embeddings(paragraph2)
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        return cos(paragraph1_embeddings[0].unsqueeze(0), paragraph2_embeddings[0].unsqueeze(0))


    def dialogue_summary_similarity(self):
        # Get the similarity between the first paragraph and the second paragraph
        paragraph1 = self.current_story_summary
        paragraph2 = ' '.join([log['dialogue'] for log in self.dialogue_log_list])
        similarity = self.two_paragraph_similarity(paragraph1, paragraph2)
        return similarity.item()
