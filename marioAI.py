import multiprocessing
import os
import pickle

import neat
import numpy as np
import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace


runs_per_net = 2


# Use the NN network phenotype and the discrete actuator force function.
def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitnesses = []

    for runs in range(runs_per_net):
        env = gym_super_mario_bros.make("SuperMarioBros-v0")
        env = JoypadSpace(env, SIMPLE_MOVEMENT)


        state = env.reset()

        fitness = 0.0
        done = False
        while not done:
            # the following line is producing some weird numpy error and I'm not sure how to resolve it
            # net.activate returns a list of "activations" which is basically the network's "prediction" of
            # which action is the best one
            # np.argmax returns the index of the maximum element in a list
            # the error is happening in numpy within the activation function
            # and somewhere it's trying to find the "truth value" of an array but the array has multiple elements
            # so something like [true false true true false] and it doesn't know how to decide.
            # this can usually be solved by doing array.any() or array.all() but since the error is happening in the back end
            # I think our input might be the problem?
            # I'm guessing some comparison is happening in the back end either with the state values or the activation values that
            # is producing a boolean array or something? really not sure.
            action = np.argmax(net.activate(state))
            state, reward, done, info = env.step(action)
            fitness += reward

        fitnesses.append(fitness)

    return np.mean(fitnesses)


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run():
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat_config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = pop.run(pe.evaluate)

    # Save the winner.
    with open('winner-mario', 'wb') as f:
        pickle.dump(winner, f)

    print(winner)


if __name__ == '__main__':
    run()
