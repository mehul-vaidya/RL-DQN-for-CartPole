import gymnasium as gym
import torch
import torch.nn as nn

# Same DQN architecture (MUST MATCH training)
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

# Setup environment
env = gym.make("CartPole-v1", render_mode="human")

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create model
policy_net = DQN(state_size, action_size).to(device)

# Load trained weights
policy_net.load_state_dict(torch.load("dqn_cartpole.pth", map_location=device))
policy_net.eval()  # IMPORTANT

print("✅ Model loaded successfully")

# Run one episode
state, _ = env.reset()
done = False

while not done:
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    action = torch.argmax(policy_net(state_tensor)).item()
    state, reward, done, truncated, _ = env.step(action)

env.close()
