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
            nn.Linear(self.roberta.config.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return self.sigmoid(logits)
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


# class of the dataset and dataloader
# input sequence: [CLS] plot_point [SEP] plot_point [SEP]...[SEP] plot_point [SEP]
# output: ranking score (0 to 10)
class RankingDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        item = self.data[index]
        # encode the dialogue history, each dialogue is separated by <s> and </s>
        input_encoding = self.tokenizer('<s>' + '</s>'.join([it for it in item['dialogue_history']]),
                                        add_special_tokens=False,
                                        padding='max_length',
                                        return_tensors='pt')
        # encode the dialogue history, each dialogue is separated by [SEP]
        input_ids = self.tokenizer.encode('<s>' + '</s>'.join([it for it in item['dialogue_history']]), add_special_tokens=False)
        attention_mask = [1] * len(input_ids)

        # check if the sequence length exceeds the maximum length
        if len(input_ids) > self.tokenizer.model_max_length:
            # if the length is exceeded, truncate the sequence
            input_ids = input_ids[:self.tokenizer.model_max_length]
            attention_mask = attention_mask[:self.tokenizer.model_max_length]
        else:
            # if not exceeded, fill the sequence
            padding_length = self.tokenizer.model_max_length - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [self.tokenizer.pad_token_id] * padding_length
        # padding_length = 512 - len(input_ids)
        # input_ids = input_ids + ([0] * padding_length)
        # attention_mask = attention_mask + ([0] * padding_length)
        label = float(item['score']) / 10
        return torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(label)
    
    def __len__(self):
        return len(self.data)
    

# class of the trainer
class RankingTrainer:
    def __init__(self, model, train_dataloader, val_dataloader, test_dataloader, device, lr=1e-4):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.model.to(self.device)
        self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.best_val_loss = 1e10
        self.best_val_epoch = 0

    def train(self, epoch):
        self.model.train()
        train_loss = 0
        for batch in self.train_dataloader:
            input_ids = batch[0].to(self.device)
            attention_mask = batch[1].to(self.device)
            label = batch[2].to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask)
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
                label = batch[2].to(self.device)
                logits = self.model(input_ids, attention_mask)
                loss += self.criterion(logits, label).item()
        return loss / len(dataloader)

    def run(self, epoch):
        train_loss = self.train(epoch)
        val_loss = self.evaluate(self.val_dataloader)
        test_loss = self.evaluate(self.test_dataloader)
        logging.info('Epoch: {}, Train Loss: {:.4f}, Val Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch, train_loss, val_loss, test_loss))
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_val_epoch = epoch
            self.model.save('ranking_model.pt')
        return val_loss, test_loss

    def get_best_model(self):
        self.model.load('ranking_model.pt')
        return self.model
