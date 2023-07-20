import os
import json
import random
import warnings
import logging
import argparse
import torch
import numpy as np
from datetime import datetime
from single_dqn import DQN, DQNAgent, DQNTrainer
from environment.graph import KnowledgeGraph
from environment.chatenv_copy import StoryBotRetellEnv

# get the current time string
NOW_STR = datetime.now().strftime('%Y%m%d_%H%M%S')

# set logging format
logging.basicConfig(filename=f'log/dqn_single_{NOW_STR}.log',
                    level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# set the hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", help="number of epochs", type=int, default=10)
parser.add_argument("-b", "--batch_size", help="batch size", type=int, default=1)
parser.add_argument("-l", "--learning_rate", help="learning rate", type=float, default=0.001)
parser.add_argument("-s", "--seed", help="random seed", type=int, default=42)
parser.add_argument("-c", "--cuda", help="cuda device", type=int, default=0, choices=[0, 1, 2])
parser.add_argument("-m", "--name1", help="save_model_1 name", type=str, default=f'model/dqn1_{NOW_STR}.pth')
parser.add_argument("-n", "--name2", help="save_model_2 name", type=str, default=f'model/dqn2_{NOW_STR}.pth')
args = parser.parse_args()

# set the hyperparameters
EPOCHS = int(args.epochs)
BATCH_SIZE = int(args.batch_size)
LEARNING_RATE = float(args.learning_rate)
SEED = int(args.seed)
CUDA = args.cuda
SAVE_MODEL_NAME_1 = args.name1
SAVE_MODEL_NAME_2 = args.name2

# show the hyperparameters
logging.info(f'epochs: {EPOCHS}')
logging.info(f'batch_size: {BATCH_SIZE}')
logging.info(f'learning_rate: {LEARNING_RATE}')
logging.info(f'seed: {SEED}')
logging.info(f'cuda: {CUDA}')
logging.info(f'save model_1 name: {SAVE_MODEL_NAME_1}')
logging.info(f'save model_2 name: {SAVE_MODEL_NAME_2}')

# setting the seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# set the cuda device
device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

# ignore the warning
warnings.filterwarnings('ignore')


story_summary_dataset = {}
    
with open('data/summary/summary_train.json', 'r', encoding='utf8') as f:
    story_summary_dataset = {**story_summary_dataset, **json.load(f)}

storybot_env_1 = StoryBotRetellEnv(story_summary_dataset,
                         reward_model_ckpt='environment/reward/model/ranking_model_best_c.pt', 
                         kg2text_model_ckpt='environment/kg2text/model/kg2text_model.pt', 
                         embedding_model_name='sentence-transformers/all-MiniLM-L6-v2', 
                         device=device)
storybot_env_2 = StoryBotRetellEnv(story_summary_dataset,
                         reward_model_ckpt='environment/reward/model/ranking_model_best_c.pt', 
                         kg2text_model_ckpt='environment/kg2text/model/kg2text_model.pt', 
                         embedding_model_name='sentence-transformers/all-MiniLM-L6-v2', 
                         device=device)

# set the agent
agent_1 = DQNAgent(env=storybot_env_1, lr=LEARNING_RATE)
agent_2 = DQNAgent(env=storybot_env_2, lr=LEARNING_RATE)

# set the trainer
trainer = DQNTrainer(env1=storybot_env_1,
                     env2=storybot_env_2,
                     agent1=agent_1,
                     agent2=agent_2,
                     story_summary_path='data/summary/summary_train.json',
                     story_kg_path='data/kg/new_train',
                     epoch=EPOCHS,
                     batch_size=BATCH_SIZE)

trainer.agent1_model_name = SAVE_MODEL_NAME_1
trainer.agent2_model_name = SAVE_MODEL_NAME_2

logging.info('Start training')
trainer.train()
logging.info('Finish training')