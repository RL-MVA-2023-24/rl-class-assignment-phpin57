from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from interface import Agent
import torch
import torch.nn as nn
import numpy as np
import random
import os

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
        self.index = int(self.index)
    def sample(self, batch_size):
        return random.sample(self.data, batch_size)
    def __len__(self):
        return len(self.data)

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)
    
class ProjectAgent(Agent):
    def __init__(self):
        self.model = QNetwork(6, 4)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.02
        self.action_dim = 4
        self.replay_buffer_capacity = 1e6
        self.batch_size = 32
        self.discount_factor = 0.99

        self.replay_buffer = ReplayBuffer(capacity=self.replay_buffer_capacity)

    def act(self, state, use_random=False):
        if use_random and np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_dim)
        q_values = self.model(torch.FloatTensor(state))
        return torch.argmax(q_values).item()

    def train_step(self, state, action, next_state, reward, done):
        self.replay_buffer.append(state, action, reward, next_state, done)
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

        
        q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.model(next_states).max(1)[0].detach()

        targets = rewards + self.discount_factor * next_q_values

        loss = self.loss_fn(q_values, targets.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self):
        self.model.load_state_dict(torch.load(os.getcwd() +'/src/model.pth'))
        
def train_agent(agent, env, nb_episodes=1500, initial_steps=5000):
    for _ in range(initial_steps):
        state = env.reset()[0]
        action = agent.act(state, use_random=True)
        next_state, reward, _, _ ,_= env.step(action)
        done = False
        agent.replay_buffer.append(state, action, reward, next_state, done)

    for episode in range(nb_episodes):
        done = False
        state = env.reset()[0]
        num_steps = 0
        cum_reward=0
        while num_steps<200:
            num_steps += 1
            action = agent.act(state, use_random=True)
            next_state, reward, _, _, _ = env.step(action)
            cum_reward+=reward
            agent.train_step(state, action, next_state, reward, done)
            state = next_state
        print(f'Episode {episode} finished with {reward}, cumulated reward of {cum_reward}')
        agent.decay_epsilon()
        if episode % 100 == 0:
            agent.save('model.pth')

replay_buffer = ReplayBuffer(capacity=1e6)  # Create an instance of the replay buffer

"""
# Inside the training loop
replay_buffer.append(state, action, next_state, reward, done)  # Store the experience in the replay buffer

# Sample a batch of experiences from the replay buffer
batch_size = 32
experiences = replay_buffer.sample(batch_size)

# Iterate over the sampled experiences and perform training steps
for experience in experiences:
    state, action, next_state, reward, done = experience
    agent.train_step(state, action, next_state, reward, done)

"""

agent = ProjectAgent()
if __name__ == "__main__":
    train_agent(agent, env)
