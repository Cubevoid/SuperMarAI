import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import numpy as np


if __name__ == "__main__":
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    state = env.reset()

    # env.step returns a tuple (state, reward, done, info)
    # info has an element keeping track of how many lives are left, among other things
    # note: the default reward function is something depending on how far along mario is
    # in the x position and how quickly he's gotten there. We could use the stuff in the info
    # dictionary to calculate our own reward function. There's things like x-pos, time, coins,
    # time, score, status (big, small, powerup, etc.)

    life = env.step(env.action_space.sample())[3]["life"]


    # experimenting with turning pixel values into grayscale
    # might be easier to deal with since each pixel can be one dimension instead of three
    # grayscale = []
    # for row in state:
        # grayscale.extend(list(map(lambda x: (0.299*x[0])+(0.587*x[1])+(0.144*x[2]), row)))

    # print(grayscale)
    # print(len(grayscale))

    done = False
    while not done:
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)

        # By default, the environment stops when mario dies 3 times
        # We want to train it to finish the level without dying so
        # we're considering it 'done' when life goes below 2 (the starting value)
        # so if it dies once, it's done.
        done = info["life"] < 2
        print(f"done: {done}, info: {info}")

        env.render()

