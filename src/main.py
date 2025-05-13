import gymnasium as gym
import highway_env    # registers the environments

def main():
    # Step 1: create env with render_mode only
    env = gym.make("highway-v0", render_mode="human")
    # Step 2: configure the underlying env
    env.unwrapped.configure({
        "duration": 1000,
        "offroad_terminal": False,
        "collision_reward": 0.0
    })

    obs, info = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            obs, info = env.reset()

    env.close()

if __name__ == "__main__":
    main()
