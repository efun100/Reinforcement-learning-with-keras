from maze_env import Maze
from RL_brain import DeepQNetwork

import time

def run_maze():
    for episode in range(300):
        # initial observation
        observation = env.reset()

        while True:
            # RL choose action based on observation
            action = RL.predict_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            observation = observation_

            # break while loop when end of this episode
            if done:
                break

            # fresh env
            env.render()

            time.sleep(3)

    # end of game
    print('game over')
    env.destroy()
    RL.save_model()

if __name__ == "__main__":
    # maze game
    env = Maze()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
    env.after(100, run_maze)
    env.mainloop()
