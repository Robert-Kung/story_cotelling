from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import DataLoader, Dataset
import logging
import os
import torch
import argparse
import random
import numpy as np
import warnings
from datetime import datetime
from kg2text_model import Kg2TextModel, Kg2TextDataset, Kg2TextTrainer

# get the current time string
NOW_STR = datetime.now().strftime('%Y%m%d_%H%M%S')

# set logging format
logging.basicConfig(filename=f'../../log/kg2text_{NOW_STR}.log',
                    level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# set the hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", help="number of epochs", type=int, default=10)
parser.add_argument("-b", "--batch_size", help="batch size", type=int, default=1)
parser.add_argument("-l", "--learning_rate", help="learning rate", type=float, default=1e-4)
parser.add_argument("-s", "--seed", help="random seed", type=int, default=42)
parser.add_argument("-c", "--cuda", help="cuda device", type=int, default=0, choices=[0, 1, 2])
parser.add_argument("-n", "--name", help="save model name", type=str, default=f'model/kg2text_model_{NOW_STR}.pt')
args = parser.parse_args()

# set the hyperparameters
EPOCHS = int(args.epochs)
BATCH_SIZE = int(args.batch_size)
LEARNING_RATE = float(args.learning_rate)
SEED = int(args.seed)
CUDA = args.cuda
SAVE_MODEL_NAME = args.name

# show the hyperparameters
logging.info(f'epochs: {EPOCHS}')
logging.info(f'batch_size: {BATCH_SIZE}')
logging.info(f'learning_rate: {LEARNING_RATE}')
logging.info(f'seed: {SEED}')
logging.info(f'cuda: {CUDA}')
logging.info(f'save_model name: {SAVE_MODEL_NAME}')

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


model_name = 't5-base'
model = Kg2TextModel(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

train_dataset = Kg2TextDataset('../../data/kg2text/kg2text_train.json', '../../data/summary/summary_train.json', tokenizer)
val_dataset = Kg2TextDataset('../../data/kg2text/kg2text_val.json', '../../data/summary/summary_val.json', tokenizer)
test_dataset = Kg2TextDataset('../../data/kg2text/kg2text_test.json', '../../data/summary/summary_test.json', tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


trainer = Kg2TextTrainer(train_dataloader=train_dataloader,
                         val_dataloader=val_dataloader,
                         test_dataloader=test_dataloader,
                         model=Kg2TextModel(model_name),
                         device=device,
                         lr=LEARNING_RATE)
trainer.model_name = SAVE_MODEL_NAME

logging.info('Start training')

best_val_loss = float('inf')
for epoch in range(EPOCHS):
    val_loss, test_loss = trainer.run(epoch)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        
logging.info(f'Best epoch: {best_epoch}, Best validation loss: {best_val_loss}')