import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import heapq


# Define Deep Q-Network (DQN) Model
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done))

    def size(self):
        return len(self.buffer)


# Define DQN Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = ReplayBuffer(10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.batch_size = 64

        self.model = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.uniform(-1, 1, self.action_dim)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_values = self.model(state_tensor)
        return action_values.numpy()[0]

    def train(self):
        if self.memory.size() < self.batch_size:
            return

        state, action, reward, next_state, done = self.memory.sample(self.batch_size)

        state_tensor = torch.FloatTensor(state)
        action_tensor = torch.LongTensor(action)  # Actions should be indices for selection
        reward_tensor = torch.FloatTensor(reward)
        next_state_tensor = torch.FloatTensor(next_state)
        done_tensor = torch.FloatTensor(done)

        # Get Q-values from the model
        q_values = self.model(state_tensor)  # Shape: [batch_size, action_dim]
        next_q_values = self.model(next_state_tensor).detach()

        # Select the Q-value corresponding to the action taken
        q_value_selected = q_values.gather(1, action_tensor.unsqueeze(1))  # Shape: [batch_size, 1]

        # Calculate the target Q-values using the Bellman equation
        target_q_values = reward_tensor + (1 - done_tensor) * self.gamma * next_q_values.max(1)[0].unsqueeze(1)

        # Calculate the loss between predicted Q-values and target Q-values
        loss = self.criterion(q_value_selected, target_q_values)

        # Backpropagate the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# Training Loop
env = gym.make("CartPole-v1")  # Replace with your custom UAV environment if necessary
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0] if isinstance(env.action_space, gym.spaces.Box) else env.action_space.n
agent = DQNAgent(state_dim, action_dim)

episodes = 20  # Reduced number of episodes for testing
for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.memory.push(state, action, reward, next_state, done)
        agent.train()
        state = next_state
        total_reward += reward

    print(f"Episode {episode + 1}, Total Reward: {total_reward}")

env.close()
print("Training Complete.")


# A* Algorithm for UAV Navigation
def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def a_star_algorithm(start, goal):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        neighbors = [(current[0] + 1, current[1]), (current[0] - 1, current[1]),
                     (current[0], current[1] + 1), (current[0], current[1] - 1)]

        for neighbor in neighbors:
            tentative_g_score = g_score[current] + heuristic(current, neighbor)
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return []  # No path found


# Evaluate performance of DQN vs A*
def evaluate_performance():
    start = (0, 0)
    goal = (5, 5)

    # DQN Performance
    state = env.reset()
    total_reward_dqn = 0
    done = False
    while not done:
        action = agent.select_action(state)
        state, reward, done, _ = env.step(action)
        total_reward_dqn += reward

    # A* Performance
    path_a_star = a_star_algorithm(start, goal)
    cost_a_star = len(path_a_star)

    print(f"DQN Total Reward: {total_reward_dqn}")
    print(f"A* Path Cost: {cost_a_star}")


evaluate_performance()
