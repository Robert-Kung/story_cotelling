from transformers import RobertaModel
import torch
from torch import nn
import logging

# class of the ranking model
# input sequence: [CLS] plot_point [SEP] plot_point [SEP]...[SEP] plot_point [SEP]
# output: ranking score (0 to 10)
class RankingModel(nn.Module):
    def __init__(self, device):
        super(RankingModel, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base').to(device)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(self.roberta.config.hidden_size * 2, self.roberta.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.roberta.config.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, story_input_ids, story_attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        outputs_story = self.roberta(input_ids=story_input_ids, attention_mask=story_attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        pooled_output_story = outputs_story[1]
        pooled_output_story = self.dropout(pooled_output_story)
        pooled_output_cat = torch.cat((pooled_output, pooled_output_story), dim=1)
        logits = self.classifier(pooled_output_cat)
        return self.sigmoid(logits)
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


# class of the dataset and dataloader
# input sequence: [CLS] plot_point [SEP] plot_point [SEP]...[SEP] plot_point [SEP]
# output: ranking score (0 to 10)
class RankingDataset(torch.utils.data.Dataset):
    def __init__(self, data, summary, tokenizer):
        self.data = data
        self.summary = summary
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        item = self.data[index]

        input_str_list = []
        input_str_list.append('<s>' + '</s>'.join([it for it in item['dialogue_history']]))
        input_str_list.append('<s>' + self.summary[item['target_story']] + '</s>')
        input_encoding = self.tokenizer(input_str_list,
                                        add_special_tokens=False,
                                        padding='max_length',
                                        truncation=True,
                                        return_tensors='pt')
        input_ids = input_encoding['input_ids'][0]
        attention_mask = input_encoding['attention_mask'][0]
        input_ids_summary = input_encoding['input_ids'][1]
        attention_mask_summary = input_encoding['attention_mask'][1]

        # # encode the dialogue history, each dialogue is separated by [SEP]
        # input_ids = self.tokenizer.encode('<s>' + '</s>'.join([it for it in item['dialogue_history']]), add_special_tokens=False)
        # attention_mask = [1] * len(input_ids)
        # input_ids_summary = self.tokenizer.encode('<s>' + self.summary[item['target_story']] + '</s>', add_special_tokens=False)
        # attention_mask_summary = [1] * len(input_ids_summary)

        # # check if the sequence length exceeds the maximum length
        # if len(input_ids) > self.tokenizer.model_max_length:
        #     # if the length is exceeded, truncate the sequence
        #     input_ids = input_ids[:self.tokenizer.model_max_length]
        #     attention_mask = attention_mask[:self.tokenizer.model_max_length]
        # else:
        #     # if not exceeded, fill the sequence
        #     padding_length = self.tokenizer.model_max_length - len(input_ids)
        #     input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
        #     attention_mask = attention_mask + [self.tokenizer.pad_token_id] * padding_length

        # if len(input_ids_summary) > self.tokenizer.model_max_length:
        #     input_ids_summary = input_ids_summary[:self.tokenizer.model_max_length]
        #     attention_mask_summary = attention_mask_summary[:self.tokenizer.model_max_length]
        # else:
        #     padding_length = self.tokenizer.model_max_length - len(input_ids_summary)
        #     input_ids_summary = input_ids_summary + [self.tokenizer.pad_token_id] * padding_length
        #     attention_mask_summary = attention_mask_summary + [self.tokenizer.pad_token_id] * padding_length

        label = float(item['score']) / 10
        # return input_ids, attention_mask, input_ids_summary, attention_mask_summary, label
        return torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(input_ids_summary), torch.tensor(attention_mask_summary), torch.tensor(label)
    
    def __len__(self):
        return len(self.data)
    

# class of the trainer
class RankingTrainer:
    def __init__(self, model, train_dataloader, val_dataloader, test_dataloader, device, lr=1e-5):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.model.to(self.device)
        self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.best_val_loss = float('inf')
        self.best_val_epoch = 0

    def train(self):
        self.model.train()
        train_loss = 0
        for batch in self.train_dataloader:
            input_ids = batch[0].to(self.device)
            attention_mask = batch[1].to(self.device)
            story_input_ids = batch[2].to(self.device)
            story_attention_mask = batch[3].to(self.device)
            label = batch[4].to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask, story_input_ids, story_attention_mask)
            loss = self.criterion(logits, label)
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
                story_input_ids = batch[2].to(self.device)
                story_attention_mask = batch[3].to(self.device)
                label = batch[4].to(self.device)
                logits = self.model(input_ids, attention_mask, story_input_ids, story_attention_mask)
                loss += self.criterion(logits, label).item()
        return loss / len(dataloader)

    def run(self, epoch, model_name):
        for e in range(epoch):
            train_loss = self.train()
            val_loss = self.evaluate(self.val_dataloader)
            test_loss = self.evaluate(self.test_dataloader)
            logging.info('Epoch: {}, Train Loss: {:.4f}, Val Loss: {:.4f}, Test Loss: {:.4f}'.format(e, train_loss, val_loss, test_loss))
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_epoch = e
                torch.save(self.model.state_dict(), model_name)
        logging.info('Best epoch: {}, Best Val Loss: {:.4f}'.format(self.best_val_epoch, self.best_val_loss))
