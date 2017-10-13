from population import Population
from genome import Genome
import numpy as np
from math import exp
from copy import deepcopy
from random import random
from PIL import Image
import gym
env = gym.make('CartPole-v1')

population_n = 150
fitness = [0 for i in range(population_n)]
node_n = 4+1

pop = [Genome(inp_n=4,out_n=1) for i in range(population_n)]
while True:
    for p in range(population_n):
        pop[p].build()
        obs = env.reset()
        done = False
        for step in range(1000):
            y = pop[p].forward(list(obs))
            action = 0
            if y[0] > 0.5:
                action = 1
            obs,reward,done,info = env.step(action)
            if done:
                fitness[p] = step
                break
    s_n = 0
    s_c = 0
    for p in pop:
        s_n += len(p.node)
        s_c += len(p.connection)
    print ('fit:',np.average(fitness),'n_len:',s_n/population_n,'c_len:',s_c/population_n)
    done = False
    obs = env.reset()
    step = 0
    while not done:
        break
        step += 1
        #rgb = env.render(mode='rgb_array')
        #if np.max(fitness) == 499:
        #rgb = Image.fromarray(rgb)
        #rgb.save('data_/CartPole'+str(step)+'.png')
        #env.env.saveScreenPNG('data_/CartPole'+str(step)+'.png')
        y = pop[np.argmax(fitness)].forward(list(obs))
        action = 0
        if y[0] > 0.5:
            action = 1
        obs,reward,done,info = env.step(action)
    #if np.max(fitness) == 499:
    #break
    #print pop[np.argmax(fitness)].forward([0,0]),pop[np.argmax(fitness)].forward([0,1]),pop[np.argmax(fitness)].forward([1,0]),pop[np.argmax(fitness)].forward([1,1])
    #print pop[np.argmax(fitness)].connection
    #print fitness
    #print fitness
    fitness = np.array(fitness,dtype=np.float32)
    fitness -= np.average(fitness)
    fitness = fitness / np.std(fitness)
    #print fitness
    fitness = np.exp(fitness)/np.sum(np.exp(fitness))
    #fitness = np.array(fitness)/np.sum(fitness)
    fitness = list(fitness)
    #print fitness
    new_pop = deepcopy(pop)
    for p in range(population_n):
        x = random()
        y = random()
        s = 0
        for i in range(population_n):
            s += fitness[i]
            if x < s and type(x) != int:
                x = i
            if y < s and type(y) != int:
                y = i
        new_pop[p] = deepcopy(pop[x])
        if x == y:
            continue
        for cy in pop[y].connection:
            exist = False
            for i,cx in enumerate(pop[x].connection):
                if cx[0] == cy[0] and cx[1] == cy[1]:
                    r = random()
                    exist = True
                    if r < 0.5:
                        new_pop[p].connection[i][2] = cy[2]
                    break
            if not exist and cy[0] in pop[x].node.keys() and cy[1] in pop[x].node.keys():
                new_pop[p].add_connection(cy[0],cy[1],cy[2],cy[3])
        r = random()
        if r < 0.4:
            if r < 0.04:
                #if len(new_pop[p].node) < 5:
                new_pop[p].mutate_add_node(node_n)
                node_n += 1
            elif r < 0.08:
                new_pop[p].mutate_delete_node()
            elif r < 0.24:
                new_pop[p].mutate_delete_connection()
            else:
                new_pop[p].mutate_add_connection()
    pop = deepcopy(new_pop)
