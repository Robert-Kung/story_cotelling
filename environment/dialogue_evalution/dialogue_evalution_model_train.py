from transformers import RobertaTokenizer
from dialogue_evalution import DialogueEvalutionModel, DialogueEvalutionDataset, DialogueEvalutionTrainer
from datetime import datetime
import torch
import warnings
import logging
import os
import random
import json
import numpy as np
import argparse
import logging

# get the current time string
NOW_STR = datetime.now().strftime('%Y%m%d_%H%M%S')

# set logging format
logging.basicConfig(filename=f'../../log/dialogue_evalution_{NOW_STR}.log',
                    level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# set the hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", help="number of epochs", type=int, default=10)
parser.add_argument("-b", "--batch_size", help="batch size", type=int, default=2)
parser.add_argument("-l", "--learning_rate", help="learning rate", type=float, default=1e-5)
parser.add_argument("-d", "--dropout", help="dropout rate", type=float, default=0.1)
parser.add_argument("-s", "--seed", help="random seed", type=int, default=42)
parser.add_argument("-c", "--cuda", help="cuda device", type=int, default=0, choices=[0, 1, 2])
parser.add_argument("-m", "--model", help="model path", type=str, default=f"model/dialogue_evalution_{NOW_STR}.pt")
args = parser.parse_args()

# set the hyperparameters
EPOCHS = int(args.epochs)
BATCH_SIZE = int(args.batch_size)
LEARNING_RATE = float(args.learning_rate)
DROPOUT = float(args.dropout)
SEED = int(args.seed)
CUDA = args.cuda
MODEL_NAME = args.model

# show the hyperparameters
logging.info(f'epochs: {EPOCHS}')
logging.info(f'batch_size: {BATCH_SIZE}')
logging.info(f'learning_rate: {LEARNING_RATE}')
logging.info(f'dropout: {DROPOUT}')
logging.info(f'seed: {SEED}')
logging.info(f'cuda: {CUDA}')
logging.info(f'save_model: {MODEL_NAME}')

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

# load the summary data
with open('../../data/summary/summary_train.json', 'r', encoding='utf8') as f:
    train_story_summary = json.load(f)
with open('../../data/summary/summary_val.json', 'r', encoding='utf8') as f:
    val_story_summary = json.load(f)
with open('../../data/summary/summary_test.json', 'r', encoding='utf8') as f:
    test_story_summary = json.load(f)

# load the plot_point data
with open('../../data/plot_point/plot_point_train.json', 'r', encoding='utf8') as f:
    train_data = json.load(f)
with open('../../data/plot_point/plot_point_val.json', 'r', encoding='utf8') as f:
    val_data = json.load(f)
with open('../../data/plot_point/plot_point_test.json', 'r', encoding='utf8') as f:
    test_data = json.load(f)


# load the tokenizer and dataset
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
train_dataset = DialogueEvalutionDataset(train_data, train_story_summary, tokenizer)
val_dataset = DialogueEvalutionDataset(val_data, val_story_summary, tokenizer)
test_dataset = DialogueEvalutionDataset(test_data, test_story_summary, tokenizer)

# load the dataloader
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# load the model and trainer
model = DialogueEvalutionModel(device)
trainer = DialogueEvalutionTrainer(model, train_dataloader, val_dataloader, test_dataloader, device)

logging.info('Start training')

trainer.run(EPOCHS, MODEL_NAME)
        
logging.info('Finish training')
