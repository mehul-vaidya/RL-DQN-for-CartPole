import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

#Create the Neural Network (Q-Network)
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.net(x)

#Setup Environment & Hyperparameters
env = gym.make("CartPole-v1")

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#two brains are created.
#policy brain - learning brain
#target brain - teaching brain
policy_net = DQN(state_size, action_size).to(device) #create brain with random knowledge
target_net = DQN(state_size, action_size).to(device) #create brain with random knowledge
target_net.load_state_dict(policy_net.state_dict())  #copies knowledge
target_net.eval() #Put the teacher brain in evaluation mode

optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)

memory = deque(maxlen=10000)

gamma = 0.99
epsilon = 1.0 #1 means explore . 0.1 means use what is learned
epsilon_decay = 0.995
min_epsilon = 0.01
batch_size = 64
episodes = 500
target_update = 10

#----------------------------------------Helper Function---------------------------------------

#“Should the dog try something random, or use what it has learned?”
def select_action(state):
    if random.random() < epsilon:
        return env.action_space.sample() #“Pick a completely random action.”
    state = torch.FloatTensor(state).unsqueeze(0).to(device) #If NOT exploring → use brain
    return torch.argmax(policy_net(state)).item() #“According to my brain, this move gives the most treats.”

#In Deep RL, the agent does not learn immediately from one step.
#It first stores experiences, then learns from them later.
def store_experience(s, a, r, s_next, done):
    memory.append((s, a, r, s_next, done))

def train_step():
    #“If I don’t have enough past experiences, don’t learn yet.”
    if len(memory) < batch_size:
        return

    #“Let me review random memories from my diary.”
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    #Convert memories into tensors
    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).unsqueeze(1).to(device)
    rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

    #Brain predicts Q-values for all actions
    #Select only the actions actually taken
    current_q = policy_net(states).gather(1, actions)

    #Teacher brain estimates best future reward
    #Picks maximum action value
    next_q = target_net(next_states).max(1)[0].unsqueeze(1)

    #“Actual outcome = treat now + possible treats later.”
    target_q = rewards + gamma * next_q * (1 - dones)

    #Measure mistake (loss)
    loss = nn.MSELoss()(current_q, target_q)

    #“Change my brain slightly so I’m less wrong next time.”
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#----------------------------------------Training Loop--------------------------------------
#training loop
for episode in range(episodes): #Loop over episodes (training days)

    #Reset the environment
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done: #Run until episode ends
        action = select_action(state) #Choose an action (explore or exploit)
        next_state, reward, done, truncated, _ = env.step(action)  #Perform the action

        store_experience(state, action, reward, next_state, done) #Store the experience
        train_step() #learn from experience

        #Move to the next state
        state = next_state
        total_reward += reward

    #Reduce exploration slowly
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    #Update target network occasionally
    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode {episode}, Reward: {total_reward}")

    MODEL_PATH = "dqn_cartpole.pth"
    torch.save(policy_net.state_dict(), MODEL_PATH)

    print("✅ Model saved to dqn_cartpole.pth")

#----------------------------------------Test---------------------------------------
# #Test the Trained Agent
# env = gym.make("CartPole-v1", render_mode="human")
# state, _ = env.reset()
# done = False
#
# while not done:
#     state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
#     action = torch.argmax(policy_net(state_tensor)).item()
#     state, reward, done, truncated, _ = env.step(action)
#
# env.close()