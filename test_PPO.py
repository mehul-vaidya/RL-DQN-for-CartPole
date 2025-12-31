import gymnasium as gym
import torch
import torch.nn as nn

# ===============================
# Actor-Critic Network
# (Must match training exactly)
# ===============================
class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU()
        )

        self.actor = nn.Sequential(
            nn.Linear(128, action_size),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.actor(x), self.critic(x)


# ===============================
# Setup Environment
# ===============================
env = gym.make("CartPole-v1", render_mode="human")

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# Load Trained Model
# ===============================
model = ActorCritic(state_size, action_size).to(device)
model.load_state_dict(torch.load("ppo_cartpole.pth", map_location=device))
model.eval()  # IMPORTANT: disable training behavior

print("✅ PPO model loaded successfully")

# ===============================
# Run One Episode
# ===============================
state, _ = env.reset()
done = False
total_reward = 0

while not done:
    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)

    with torch.no_grad():  # no learning during testing
        probs, _ = model(state_t)
        action = torch.argmax(probs).item()  # best action only

    state, reward, done, truncated, _ = env.step(action)
    total_reward += reward

print("🎯 Total Reward:", total_reward)
env.close()
