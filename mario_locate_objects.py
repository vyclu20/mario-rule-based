import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros import actions

#######################################################
# Custom Wrappers

# So this wrapper I added because when I was debugging it, the reset() method does not accept a seed argument (I was using a random seed 
# at one point, or somewhere in the code seed was being used, so I've wrapped it to 
# simply check if the seed argument is present in the kwargs and removes it if it exists before calling the reset() method 
# (TypeError)
class RemoveSeedWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        kwargs.pop('seed', None)
        kwargs.pop('options', None)
        return super().reset(**kwargs)


class CustomMarioEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super(CustomMarioEnvWrapper, self).__init__(env)
        self.step_counter = 0
        self.prev_action = None

    def step(self, action):
        # Introduce a frame delay for the jump action
        if action == 5:
            self.env.step(action)
        else:
            self.prev_action = action
            self.step_counter += 1

        result = self.env.step(action)
        
        return result
    
# This one is for rewards for certain actions, specifically, if action 4 is taken, to give a reward (an incentive to do action 4)
class ActionRewardWrapper(gym.Wrapper):
    def __init__(self, env, action_reward_map):
        super(ActionRewardWrapper, self).__init__(env)
        self.action_reward_map = action_reward_map

    def step(self, action):
        obs, reward, done, _, info = self.env.step(action)
        if action in self.action_reward_map:
            reward += self.action_reward_map[action]  # (modify the reward based on the action taken)
        return obs, reward, done, _, info

# This wrapper I made because CnnPolicy expects the shape in the format: (channels * num_stacked_frames, height, width),
# so I had to alter the dimensions so it follows the format (problem is for Box(, , , ,..))
class PermuteObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super(PermuteObservation, self).__init__(env)
        assert len(env.observation_space.shape) == 4
        old_shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(old_shape[-1]*old_shape[0], old_shape[1], old_shape[2]), dtype=env.observation_space.dtype)

    def observation(self, observation):
        observation_np = np.asarray(observation)
        return observation_np.transpose(3, 0, 1, 2).reshape(-1, *observation_np.shape[1:3])
    
class ExtendedJumpWrapper(gym.Wrapper):
    def __init__(self, env, jump_action, jump_extension_frames):
        super(ExtendedJumpWrapper, self).__init__(env)
        self.jump_action = jump_action
        self.jump_extension_frames = jump_extension_frames

    def step(self, action):
        total_reward = 0
        total_done = False
        total_info = {}

        obs, reward, done, _, info = self.env.step(action)
        total_reward += reward

        # If the action taken is the jump action, repeat it for several frames
        if action == self.jump_action:
            for _ in range(self.jump_extension_frames):
                if done:  # If the episode ends during the extended jump, break
                    break
                obs, reward, done, _, info = self.env.step(action)
                total_reward += reward

        return obs, total_reward, done, _, total_info
    
######################################################

# Create a function to make the Mario environment
def make_mario_env():
    env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")

    # Frame skipping
    env = JoypadSpace(env, actions.SIMPLE_MOVEMENT)

    # Remove seed argument
    env = RemoveSeedWrapper(env)

    env = gym.wrappers.TransformObservation(env, lambda obs: obs.astype('float32') / 255.)
    
    # Reduce the number of stacked frames
    env = gym.wrappers.FrameStack(env, num_stack=2)

    env = PermuteObservation(env)
    
    # Use the custom action space wrapper
    env = CustomMarioEnvWrapper(env)

    modified_rewards = {4: 10.0}  # Add a reward of 10.0 for taking action 4
    env = ActionRewardWrapper(env, modified_rewards)    

    jump_action = 4 
    jump_extension_frames = 10  # Simulates a longer jump by pressing "jump" for 3 extra frames
    env = ExtendedJumpWrapper(env, jump_action, jump_extension_frames)

    return env

def main():
    print(f"Using a single environment for training")

    # Create the environment for training
    env = DummyVecEnv([make_mario_env])

    # Define the PPO agent
    model = PPO('CnnPolicy', env, n_steps=50, verbose=1)

    # Set the number of training iterations and total timesteps per iteration
    num_iterations = 2
    total_timesteps_per_iteration = 100
    print_interval = 100  # Print every 100 steps

    scores = []

    try:
        for iteration in range(num_iterations):
            print(f"Iteration {iteration + 1}")

            iteration_score = 0

            for t in range(0, total_timesteps_per_iteration, print_interval):
                model.learn(total_timesteps=print_interval, reset_num_timesteps=False)

                # Calculate the iteration score
                obs = env.reset()
                for step in range(print_interval):
                    action, _ = model.predict(obs)
                    obs, rewards, dones, info = env.step(action)
                    iteration_score += rewards

                print(f"Iteration {iteration + 1}, Step {t + print_interval}: Score = {iteration_score}")

            # Store the iteration score
            scores.append(iteration_score)

    except Exception as e:
        print(f"Error during training: {e}")

    # Print out the results for each iteration
    for i in range(len(scores)):
        print(f"Iteration {i + 1}: Score = {scores[i]}")

    # Calculate and print the total average time
    total_average_time = total_timesteps_per_iteration * num_iterations  # Assuming 1 step per second
    print(f"Total Average Time: {total_average_time} seconds")

    # Save the trained model
    model.save('ppo_mario')

    # Evaluate the trained model
    env_eval = DummyVecEnv([make_mario_env])
    mean_reward, std_reward = evaluate_policy(model, env_eval, n_eval_episodes=10)
    print(f'Mean reward: {mean_reward} +/- {std_reward}')

    # View the agent's performance
    obs = env_eval.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env_eval.step(action)
        env_eval.render()
    env_eval.close()

if __name__ == '__main__':
    main()