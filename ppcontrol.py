import gym
import highway_env
import numpy as np
import matplotlib.pyplot as plt
import time

def pure_pursuit_control(ego, look_ahead=10.0, L=2.5):
    """
    Compute steering and throttle using Pure Pursuit.
    ego: array [x, y, vx, vy, cos_h, sin_h]
    look_ahead: look-ahead distance (m)
    L: wheelbase (m)
    Returns: np.array([steer, throttle])
    """
    x, y, vx, vy, cos_h, sin_h = ego
    # Vehicle heading
    yaw = np.arctan2(sin_h, cos_h)
    # Define look-ahead point straight ahead in vehicle frame
    tx = x + look_ahead * cos_h
    ty = y + look_ahead * sin_h
    # Angle between heading and the line to the look-ahead point
    alpha = np.arctan2(ty - y, tx - x) - yaw
    # Pure Pursuit steering law
    delta = np.arctan2(2 * L * np.sin(alpha), look_ahead)
    # Normalize steering to [-1, 1] (env expects this)
    steer = float(delta / (np.pi / 2))
    # Constant throttle
    throttle = 0.5
    return np.array([steer, throttle])

def main():
    # 1. Create & configure the highway environment
    env = gym.make("highway-v0")
    env.configure({
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 5,
            "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"]
        },
        "action": {
            "type": "ContinuousAction"
        }
    })

    # 2. Reset environment
    obs = env.reset()
    done = False

    # 3. Simulation loop
    try:
        while not done:
            # Extract ego vehicle state
            ego = obs["vehicles"][:, :6][0]   # [x, y, vx, vy, cos_h, sin_h]

            # Compute control action
            action = pure_pursuit_control(ego, look_ahead=10.0, L=2.5)

            # Step the environment
            obs, reward, done, info = env.step(action)

            # Render to screen
            env.render()

            # Sleep to match real time (adjust if too slow/fast)
            time.sleep(0.05)

    finally:
        # 4. Clean up
        env.close()

if __name__ == "__main__":
    main()
