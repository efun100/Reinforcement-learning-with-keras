from maze_env import Maze
import time
import numpy as np

def get_action(observation):
    print(observation)
    if observation[1] < 0:
        return 1  #down
    elif observation[1] > 0:
        return 0  #up
    elif observation[0] < 0:
        return 2  #right
    else:
        return 3  #left

def run_maze():
    for episode in range(300):
        # initial observation
        observation = env.reset()

        while True:
            if np.random.rand() > 0.5:
                action = int(np.random.randint(0, 4))
            else:
                action = get_action(observation)
            print(action)

            # RL take action and get next observation and reward
            observation, reward, done = env.step(action)
            # fresh env
            env.render()
            time.sleep(1)

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = Maze()
    print(env.n_features)
    env.after(100, run_maze)
    env.mainloop()
