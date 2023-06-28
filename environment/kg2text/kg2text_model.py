from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import DataLoader, Dataset
import torch
from torch import nn
import logging
import json
import os


class Kg2TextModel(nn.Module):
    def __init__(self, model_name_or_path):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs
    
    def generate(self, input_ids, attention_mask, num_beams=2, max_length=128, skip_special_tokens=True):
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=num_beams,
            max_length=max_length
        )
        return outputs

    def save(self, path):
        torch.save(self.state_dict(), path)


class Kg2TextDataset(Dataset):
    def __init__(self, data_path, summary_path, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.data = []
        self.summary_data = {}
        self.load_summary_data(summary_path)
        self.load_data(data_path)
        self.max_source_length = 1024
        self.max_target_length = 64
    
    def load_summary_data(self, summary_path):
        with open(summary_path, 'r', encoding='utf8') as f:
            self.summary_data = json.load(f)

    def load_data(self, data_path):
        with open(data_path, 'r', encoding='utf8') as f:
            json_data = json.load(f)
            json_data = json_data['data']
        for item in json_data:
            t_str = ''
            for t in item['triple']:
                t_str += f'[{t["subject"]}, {t["relation"]}, {t["object"]}]'
            
            prompt = f'graph to text: {t_str} content: {self.summary_data[item["story_name"]]}'
            # prompt = f'graph to text: {t_str}'
            self.data.append(
                {
                    'story_name': item['story_name'],
                    'input': prompt,
                    'output': item['raw_text']
                }
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(item['input'],
                                   max_length=self.max_source_length,
                                   pad_to_max_length=True,
                                   truncation=True,
                                   return_tensors='pt')
        input_ids = encoding.input_ids[0]
        attention_mask = encoding.attention_mask[0]
        target_encoding = self.tokenizer(item['output'],
                                           max_length=self.max_target_length, 
                                           pad_to_max_length=True,
                                           truncation=True, 
                                           return_tensors='pt')
        labels = target_encoding.input_ids[0]

        # replace padding token id's of the labels by -100 so it's ignored by the loss
        labels[labels == self.tokenizer.pad_token_id] = -100

        return input_ids, attention_mask, labels


class Kg2TextTrainer:
    def __init__(self, model, train_dataloader, val_dataloader, test_dataloader, device, lr=1e-4):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.best_val_loss = 1e10
        self.best_val_epoch = 0
        self.model_name = 'kg2text_model.pt'

    def train(self):
        self.model.train()
        train_loss = 0
        for batch in self.train_dataloader:
            input_ids = batch[0].to(self.device)
            attention_mask = batch[1].to(self.device)
            labels = batch[2].to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask, labels)
            # loss = self.criterion(logits, labels)
            loss = logits.loss
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
        return train_loss / len(self.train_dataloader)

    def evaluate(self, dataloader):
        self.model.eval()
        loss = 0
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device)
                logits = self.model(input_ids, attention_mask, labels)
                # loss += self.criterion(logits, labels).item()
                loss += logits.loss.item()
        return loss / len(dataloader)

    def run(self, epoch):
        train_loss = self.train()
        val_loss = self.evaluate(self.val_dataloader)
        test_loss = self.evaluate(self.test_dataloader)
        logging.info('Epoch: {}, Train Loss: {:.4f}, Val Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch, train_loss, val_loss, test_loss))
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_val_epoch = epoch
            self.model.save(self.model_name)
        return val_loss, test_loss

    def get_best_model(self):
        self.model.load(self.model_name)
        return self.model
