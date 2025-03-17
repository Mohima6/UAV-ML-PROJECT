import gym
import pybullet as p
import pybullet_data
import numpy as np
import time

class UAVEnv(gym.Env):
    def __init__(self):
        super(UAVEnv, self).__init__()

        # Connect to PyBullet (use GUI)
        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load a simple ground plane
        self.plane = p.loadURDF("plane.urdf")

        # Create a simple box to represent the UAV
        self.uav = p.loadURDF("r2d2.urdf", [0, 0, 1])

        # Simulation parameters
        self.max_steps = 1000
        self.step_count = 0

        # Define observation and action space
        self.observation_space = gym.spaces.Box(low=-10, high=10, shape=(6,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

    def step(self, action):
        """Move UAV based on action"""
        force = [action[0], action[1], action[2]]
        p.applyExternalForce(self.uav, -1, force, [0, 0, 0], p.WORLD_FRAME)
        p.stepSimulation()
        time.sleep(1/240)  # Simulate real-time

        # Get UAV position
        pos, ori = p.getBasePositionAndOrientation(self.uav)
        lin_vel, ang_vel = p.getBaseVelocity(self.uav)

        # Construct observation
        obs = np.array(pos + lin_vel)
        reward = -np.linalg.norm(np.array(pos[:2]))  # Reward for staying near origin
        done = self.step_count > self.max_steps

        return obs, reward, done, {}

    def reset(self):
        """Reset UAV position"""
        p.resetBasePositionAndOrientation(self.uav, [0, 0, 1], [0, 0, 0, 1])
        self.step_count = 0
        return np.zeros(6)

    def render(self, mode="human"):
        pass

    def close(self):
        p.disconnect()

# Create environment
env = UAVEnv()

# Run the simulation and collect data
data = []
for _ in range(100):
    action = env.action_space.sample()  # Random actions
    obs, reward, done, _ = env.step(action)
    data.append((obs.tolist(), reward))
    if done:
        env.reset()

env.close()

# Save collected data
import pandas as pd
df = pd.DataFrame(data, columns=["Observation", "Reward"])
df.to_csv("uav_data.csv", index=False)
print("Data collection complete. Saved as uav_data.csv.")
