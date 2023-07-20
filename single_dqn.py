import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from environment.graph import KnowledgeGraph
from environment.chatenv_copy import StoryBotRetellEnv
import json
import logging
from datetime import datetime


# class of DQN
class DQN(nn.Module):
    def __init__(self, env, gamma=0.99, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.01, lr=0.001) -> None:
        super(DQN, self).__init__()
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.lr = lr
        self.memory = []
        self.model = nn.Sequential(
            torch.nn.Linear(self.env.observation_space, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, len(self.env.action_space))
        )
        self.model.to(self.env.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def forward(self, x):
        return self.model(x)
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randint(0, len(self.env.action_space)-1)
        else:
            return torch.argmax(self.forward(state)).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.forward(next_state))
            target_f = self.forward(state)
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = F.mse_loss(self.forward(state), target_f)
            loss.backward()
            self.optimizer.step()
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay
            
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path):
        torch.save(self.model.state_dict(), path)


# class of DQN agent
class DQNAgent:
    def __init__(self, env, gamma=0.99, epsilon=1.0, epsilon_decay=0.7, epsilon_min=0.01, lr=0.001) -> None:
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.lr = lr
        self.memory = []
        self.model = DQN(self.env, self.gamma, self.epsilon, self.epsilon_decay, self.epsilon_min, self.lr)

    def act(self, state):
        return self.model.act(state)

    def remember(self, state, action, reward, next_state, done):
        self.model.remember(state, action, reward, next_state, done)

    def replay(self, batch_size=32):
        self.model.replay(batch_size)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model.model.load_state_dict(torch.load(path))
        self.model.model.eval()


# trainer of DQN agent
class DQNTrainer:
    def __init__(self, env1: StoryBotRetellEnv, env2: StoryBotRetellEnv, agent1: DQNAgent, agent2: DQNAgent, \
                 story_summary_path, story_kg_path, epoch=100, episodes=100, batch_size=8) -> None:
        # environment
        self.env1 = env1
        # self.env2 = env2

        # agent
        self.agent1 = agent1
        self.agent2 = agent2
        self.agent1_model_name = 'model/dqn1.pth'
        self.agent2_model_name = 'model/dqn2.pth'

        # training parameters
        self.episodes = episodes
        self.batch_size = batch_size
        self.epoch = epoch

        # all story names and knowledge graphs
        self.story_summary_path = story_summary_path
        self.story_kg_path = story_kg_path
        with open(self.story_summary_path, 'r') as f:
            self.story_summary = json.load(f)
        self.story_name_list = list(self.story_summary.keys())

        self.kg_dict = {}
        for story_name in self.story_name_list:
            kg = KnowledgeGraph(device=self.env1.embedding_model_device,
                                model=self.env1.embedding_model,
                                tokenizer=self.env1.embedding_tokenizer,
                                story_kg_file=os.path.join(self.story_kg_path, story_name+'.json'))
            self.kg_dict[story_name] = kg

        # current story name and knowledge graph
        self.current_story_name = None
        self.current_kg = None

    def set_current_story_kg(self, story_name):
        self.current_story_name = story_name
        self.current_kg = self.kg_dict[story_name]
    
    def merge_dicts(self, dict1, dict2):
        merged_dict = dict1.copy()
        for key, value in dict2.items():
            if key in merged_dict:
                merged_dict[key] += value
            else:
                merged_dict[key] = value
        return merged_dict

    def train(self):
        dt_start_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        df = pd.DataFrame(columns=['epoch', 'episode', 'story_name', 'score1', 'score2', 'epsilon1', 'epsilon2'])

        best_score_1 = -1
        # best_score_2 = -1

        dialogue_history_df1 = pd.DataFrame()

        for e in range(self.epoch):
            score1_list = []
            score2_list = []
            similarity1_list = []
            similarity2_list = []
            action_counter = {}
            # for episode in range(self.episodes):
            for episode, story in enumerate(self.story_name_list):
                self.set_current_story_kg(story)
                self.env1.reset(story_name=self.current_story_name, story_kg=self.current_kg)
                # self.env2.reset(story_name=self.current_story_name, story_kg=self.current_kg)
                
                output_dialogue2 = ''
                output_kg2 = None
                self.env1.render(input_sentence=output_dialogue2, input_kg=output_kg2)
                while True:
                    # two agent talk with each other
                    # self.env1.render(input_sentence=output_dialogue2, input_kg=output_kg2)
                    done1, done1_msg = self.env1.done()
                    if not done1:
                        state1 = self.env1.observation()
                        state1 = torch.tensor(state1, dtype=torch.float32).unsqueeze(0)
                        action1 = self.agent1.act(state1)
                        output_dialogue1, output_kg1 = self.env1.step(action1)
                        next_state1 = self.env1.observation()
                        reward1, score1 = self.env1.reward()
                        done1, done1_msg = self.env1.done()
                        self.agent1.remember(state1, action1, reward1, next_state1, done1)

                    if done1:
                        break
                    
                    # self.env1.render(input_sentence=output_dialogue1, input_kg=output_kg1)
                    done2, done2_msg = self.env1.done()
                    if not done2:
                        state2 = self.env1.observation()
                        state2 = torch.tensor(state2, dtype=torch.float32).unsqueeze(0)
                        action2 = self.agent2.act(state2)
                        output_dialogue2, output_kg2 = self.env1.step(action2)
                        next_state2 = self.env1.observation()
                        reward2, score2 = self.env1.reward()
                        done2, done2_msg = self.env1.done()
                        self.agent2.remember(state2, action2, reward2, next_state2, done2)

                    if done2:
                        break

                self.agent1.replay(self.batch_size)
                self.agent2.replay(self.batch_size)

                final_score1 = self.env1.current_score
                # final_score2 = self.env2.current_score

                similarity1 = self.env1.dialogue_summary_similarity()
                # similarity2 = self.env2.dialogue_summary_similarity()

                action_counter = self.merge_dicts(action_counter, self.env1.count_actions())
                # action_counter = self.merge_dicts(action_counter, self.env2.count_actions())


                # append the result to df
                df = df.append({'epoch': e, 'episode': episode, 'story_name': self.current_story_name, \
                                'score1': final_score1, \
                                'epsilon1': self.agent1.model.epsilon, 'epsilon2': self.agent2.model.epsilon, \
                                'similarity1': similarity1}, ignore_index=True)
                
                dialogue_history_df1 = dialogue_history_df1.append(self.env1.dialogue_log_list, ignore_index=True)
                dialogue_history_df1.to_csv(f'output/dialogue_history1_{dt_start_str}.csv', index=False)
                
                score1_list.append(final_score1)
                # score2_list.append(final_score2)
                similarity1_list.append(similarity1)
                # similarity2_list.append(similarity2)
            
            self.agent1.model.update_epsilon()
            self.agent2.model.update_epsilon()

            score1_mean = np.mean(score1_list)
            # score2_mean = np.mean(score2_list)
            similarity1_mean = np.mean(similarity1_list)
            # similarity2_mean = np.mean(similarity2_list)
            logging.info('epoch: {:2}, score: {:.4f}, similarity: {:.4f}, action:{}'.format(e, score1_mean, similarity1_mean, action_counter))

            if score1_mean > best_score_1:
                best_score_1 = score1_mean
                self.agent1.save(self.agent1_model_name)
            # if score1_mean > best_score_1:
            #     best_score_2 = score2_mean
            #     self.agent2.save(self.agent2_model_name)
            
            # self.agent1.save(self.agent1_model_name)
            # self.agent2.save(self.agent2_model_name)
            df.to_csv(f'result_{dt_start_str}.csv', index=False)
