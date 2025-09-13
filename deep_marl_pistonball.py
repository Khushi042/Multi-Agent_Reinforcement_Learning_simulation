import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import imageio
import pygame
from pettingzoo.butterfly import pistonball_v6

# -----------------------------
# Neural Network for DQN
# -----------------------------
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# -----------------------------
# Hyperparameters
# -----------------------------
gamma = 0.99
lr = 0.001
epsilon = 0.2
episodes = 20
max_steps = 100

# -----------------------------
# Initialize Environment
# -----------------------------
# Use render_mode="human" to ensure Pygame window opens
env = pistonball_v6.parallel_env(n_pistons=5, continuous=False, render_mode="human")
obs_raw, _ = env.reset()
agents = list(obs_raw.keys())
first_agent = agents[0]
obs_dim = obs_raw[first_agent].size
act_dim = env.action_space(first_agent).n

# Create DQN and optimizer for each agent
networks = {agent: DQN(obs_dim, act_dim) for agent in agents}
optimizers = {agent: optim.Adam(networks[agent].parameters(), lr=lr) for agent in agents}

# Store rewards and frames for GIF
episode_rewards_list = []
frames = []

# -----------------------------
# Training Loop with GIF capture
# -----------------------------
try:
    for ep in range(episodes):
        obs_raw, _ = env.reset()
        total_reward = {agent: 0 for agent in agents}

        for step in range(max_steps):
            # Render environment (Pygame window active)
            env.render()

            # Capture frame immediately after render
            surface = pygame.display.get_surface()
            if surface is not None:
                frame = pygame.surfarray.array3d(surface)
                frame = np.transpose(frame, (1, 0, 2))  # H, W, C
                frames.append(frame)

            # Select actions
            actions = {}
            for agent in agents:
                state = torch.FloatTensor(obs_raw[agent].flatten())
                if np.random.rand() < epsilon:
                    actions[agent] = env.action_space(agent).sample()
                else:
                    with torch.no_grad():
                        q_values = networks[agent](state)
                        actions[agent] = torch.argmax(q_values).item()

            # Step environment
            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            next_obs_raw = next_obs
            done_flags = {agent: terminations[agent] or truncations[agent] for agent in agents}

            # Update DQNs
            for agent in agents:
                state = torch.FloatTensor(obs_raw[agent].flatten())
                next_state = torch.FloatTensor(next_obs_raw[agent].flatten())
                reward = rewards[agent]

                target = reward + gamma * torch.max(networks[agent](next_state)).item()
                output = networks[agent](state)[actions[agent]]

                loss = nn.MSELoss()(output, torch.tensor(target))
                optimizers[agent].zero_grad()
                loss.backward()
                optimizers[agent].step()

                total_reward[agent] += reward

            obs_raw = next_obs_raw
            if all(done_flags.values()):
                break

        print(f"Episode {ep+1} Total Rewards: {total_reward}")
        episode_rewards_list.append(np.mean(list(total_reward.values())))

except KeyboardInterrupt:
    print("Training stopped manually.")

finally:
    # -----------------------------
    # Close environment and Pygame
    # -----------------------------
    env.close()
    pygame.quit()
    print("Environment and Pygame closed.")

# -----------------------------
# Save GIF safely
# -----------------------------
gif_filename = "pistonball_demo.gif"
if len(frames) > 0:
    imageio.mimsave(gif_filename, frames, fps=10)
    print(f"GIF saved as {gif_filename}")
else:
    print("No frames captured. GIF not saved.")

# -----------------------------
# Plot Rewards
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(range(1, len(episode_rewards_list)+1), episode_rewards_list, marker='o')
plt.title("Average Episode Rewards Over Time")
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.grid(True)
plt.show()
