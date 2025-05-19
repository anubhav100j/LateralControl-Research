import gymnasium as gym
import highway_env    # registers the environments

def main():
    env = gym.make("highway-v0", render_mode="human")
    env.unwrapped.configure({
        "vehicles_count": 1,
        "controlled_vehicles": 1,
        "duration": 100,
        "offroad_terminal": False,
        "collision_reward": 0.0
    })

    # 1) Inspect the dict of actions
    actions = env.unwrapped.action_type.actions
    print("Action mapping:", actions)

    # 2) Find the integer code for "IDLE" (keep lane)
    lane_keep = next(k for k, v in actions.items() if v == "IDLE")
    print("Using action code", lane_keep, "for IDLE")

    obs, info = env.reset()
    for step in range(100):
        obs, reward, terminated, truncated, info = env.step(lane_keep)
        env.render()
        print(f"Step {step:3d} term={terminated}, trunc={truncated}")
        if terminated or truncated:
            break

    env.close()

if __name__ == "__main__":
    main()
