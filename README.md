# Multi-Agent_Reinforcement_Learning_simulation
Multi-Agent Reinforcement Learning (MARL) simulation using Pistonball environment from PettingZoo. Multiple agents learn to coordinate actions via DQNs, with rewards tracked per episode. Training results are visualized and saved as a GIF animation.
# Multi-Agent Reinforcement Learning with Pistonball

This project demonstrates a **Multi-Agent Reinforcement Learning (MARL)** simulation using the **Pistonball environment** from the [PettingZoo](https://www.pettingzoo.ml/) library. Multiple agents are trained using **Deep Q-Networks (DQN)** to coordinate and maximize rewards through exploration and learning.

## Features
- Simulates coordination among multiple agents in the Pistonball environment.
- Uses DQN for training agents.
- Tracks rewards per episode.
- Captures and saves training progress as a GIF animation.

## Requirements
- Python 3.8+
- PettingZoo
- Pygame
- Numpy
- Torch
- Imageio

Install dependencies:
```bash
pip install pettingzoo pygame numpy torch imageio

Usage-
Run the simulation:
python deep_marl_pistonball.py

Notes

- Training can be stopped manually with Ctrl+C.

- Ensure that the environment is created with render_mode="human" for proper visualization.

- Adjust hyperparameters like learning rate, epsilon, and network size for better performance.

License

This project is open-source and free to use for educational purposes.
