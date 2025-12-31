import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ===============================
# Actor-Critic Network
# ===============================
#actor is decision make and Critic is evaluator
class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU()
        )

        # Actor: decides action probabilities
        self.actor = nn.Sequential(
            nn.Linear(128, action_size),
            nn.Softmax(dim=-1)
        )

        # Critic: estimates how good the state is
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.actor(x), self.critic(x)


# ===============================
# Setup
# ===============================
env = gym.make("CartPole-v1")

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ActorCritic(state_size, action_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=3e-4)

gamma = 0.99
clip_eps = 0.2
ppo_epochs = 10
episodes = 500


# ===============================
# Memory (On-policy)
# ===============================
states = []
actions = []
log_probs = []
rewards = []
values = []
dones = []


# ===============================
# Action Selection
# ===============================
def select_action(state):
    state = torch.FloatTensor(state).unsqueeze(0).to(device)

    probs, value = model(state)
    dist = torch.distributions.Categorical(probs)

    action = dist.sample()
    return action.item(), dist.log_prob(action), value


# ===============================
# Compute Returns & Advantages
# ===============================
#“How many treats did I actually get from each step… and was it more or less than I expected?”
def compute_returns_advantages():
    #Start With Empty Memory
    returns = []
    G = 0

    #Walk Backwards Through the Episode
    for r, d in zip(reversed(rewards), reversed(dones)):
        #Reset When Episode Ends
        if d:
            G = 0
        #Calculate Discounted Reward
        G = r + gamma * G
        #Store Return in Correct Order
        returns.insert(0, G)

    #Convert to Tensors
    returns = torch.tensor(returns).to(device)
    values_t = torch.tensor(values).squeeze().to(device)

    #Calculate Advantage (MOST IMPORTANT)
    advantages = returns - values_t
    #Return Both
    return returns, advantages


# ===============================
# PPO Update
# ===============================
def ppo_update():
    #Convert Stored Memories to Tensors
    states_t = torch.FloatTensor(states).to(device)
    actions_t = torch.LongTensor(actions).to(device)
    old_log_probs_t = torch.stack(log_probs).detach().to(device)

    returns, advantages = compute_returns_advantages()

    #Repeat PPO Update a Few Times
    for _ in range(ppo_epochs):
        #Current Brain Makes New Predictions
        probs, values_new = model(states_t)
        dist = torch.distributions.Categorical(probs)

        #Compare New Policy vs Old Policy
        new_log_probs = dist.log_prob(actions_t)
        ratio = torch.exp(new_log_probs - old_log_probs_t)

        #PPO Safety Clamp (MOST IMPORTANT PART)- improve but slowly
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages

        #Actor Loss (Policy Learning) /Critic Loss (Value Learning)
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = nn.MSELoss()(values_new.squeeze(), returns)

        #Total Loss
        loss = actor_loss + 0.5 * critic_loss

        #Update the Brain
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# ===============================
# Training Loop
# ===============================
for episode in range(episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0

    states.clear()
    actions.clear()
    log_probs.clear()
    rewards.clear()
    values.clear()
    dones.clear()

    while not done:
        action, log_prob, value = select_action(state)
        next_state, reward, done, truncated, _ = env.step(action)

        states.append(state)
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        values.append(value.item())
        dones.append(done)

        state = next_state
        total_reward += reward

    ppo_update()
    print(f"Episode {episode} | Reward: {total_reward}")

# ===============================
# Save Model
# ===============================
torch.save(model.state_dict(), "ppo_cartpole.pth")
print("✅ PPO model saved")


# ===============================
# Test
# ===============================
env = gym.make("CartPole-v1", render_mode="human")
state, _ = env.reset()
done = False

while not done:
    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
    probs, _ = model(state_t)
    action = torch.argmax(probs).item()

    state, reward, done, truncated, _ = env.step(action)

env.close()
