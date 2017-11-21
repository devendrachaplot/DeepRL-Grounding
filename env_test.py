import vizdoom
import argparse
import env as grounding_env
import numpy as np

parser = argparse.ArgumentParser(description='Grounding Environment Test')
parser.add_argument('-l', '--max-episode-length', type=int, default=30,
                    help='maximum length of an episode (default: 30)')
parser.add_argument('-d', '--difficulty', type=str, default="hard",
                    help="""Difficulty of the environment,
                    "easy", "medium" or "hard" (default: hard)""")
parser.add_argument('--living-reward', type=float, default=0,
                    help="""Default reward at each time step (default: 0,
                    change to -0.005 to encourage shorter paths)""")
parser.add_argument('--frame-width', type=int, default=300,
                    help='Frame width (default: 300)')
parser.add_argument('--frame-height', type=int, default=168,
                    help='Frame height (default: 168)')
parser.add_argument('-v', '--visualize', type=int, default=1,
                    help="""Visualize the envrionment (default: 1,
                    change to 0 for faster training)""")
parser.add_argument('--sleep', type=float, default=0,
                    help="""Sleep between frames for better
                    visualization (default: 0)""")
parser.add_argument('-t', '--use_train_instructions', type=int, default=1,
                    help="""0: Use test instructions, 1: Use train instructions
                    (default: 1)""")
parser.add_argument('--scenario-path', type=str, default="maps/room.wad",
                    help="""Doom scenario file to load
                    (default: maps/room.wad)""")
parser.add_argument('--interactive', type=int, default=0,
                    help="""Interactive mode enables human to play
                    (default: 0)""")
parser.add_argument('--all-instr-file', type=str,
                    default="data/instructions_all.json",
                    help="""All instructions file
                    (default: data/instructions_all.json)""")
parser.add_argument('--train-instr-file', type=str,
                    default="data/instructions_train.json",
                    help="""Train instructions file
                    (default: data/instructions_train.json)""")
parser.add_argument('--test-instr-file', type=str,
                    default="data/instructions_test.json",
                    help="""Test instructions file
                    (default: data/instructions_test.json)""")
parser.add_argument('--object-size-file', type=str,
                    default="data/object_sizes.txt",
                    help='Object size file (default: data/object_sizes.txt)')

if __name__ == '__main__':
    args = parser.parse_args()
    env = grounding_env.GroundingEnv(args)
    env.game_init()

    num_episodes = 0
    rewards_per_episode = []
    reward_sum = 0
    is_final = 1
    while num_episodes < 100:
        if is_final:
            (image, instruction), _, _, _ = env.reset()
            print("Instruction: {}".format(instruction))

        # Take a random action
        (image, instruction), reward, is_final, _ = \
            env.step(np.random.randint(3))
        reward_sum += reward

        if is_final:
            print("Total Reward: {}".format(reward_sum))
            rewards_per_episode.append(reward_sum)
            num_episodes += 1
            reward_sum = 0
            if num_episodes % 10 == 0:
                print("Avg Reward per Episode: {}".format(
                    np.mean(rewards_per_episode)))
