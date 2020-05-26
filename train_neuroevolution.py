import collections
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import manipulator as tor
from tqdm import tqdm


NOISE_STD = 0.01
POPULATION_SIZE = 250
PARENTS_COUNT = 10
WORKERS_COUNT = 10
SEEDS_PER_WORKER = POPULATION_SIZE // WORKERS_COUNT
MAX_SEED = 2**32 - 1

OBJ_SIZE = 10
EPOCHS = 1000
MAX_STEPS = 30
TEST_STEPS = 30


class Net(nn.Module):
    def __init__(self, obs_size, act_size, hid_size=64):
        super(Net, self).__init__()

        self.mu = nn.Sequential(
            nn.Linear(obs_size, hid_size),
            nn.Tanh(),
            nn.Linear(hid_size, hid_size),
            nn.Tanh(),
            nn.Linear(hid_size, act_size),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.mu(x)


def evaluate(env, net):
    obs = env.reset(returnable=True)
    reward = 0.0
    steps = 0
    while True:
        obs_v = torch.FloatTensor([obs])
        action_v = net(obs_v)
        obs, r, done = env.step(action_v.data.numpy()[0]*180)
        reward += r
        steps += 1
        if done or (steps == MAX_STEPS):
            break
    return reward, steps


def mutate_net(net, seed, copy_net=True):
    new_net = copy.deepcopy(net) if copy_net else net
    np.random.seed(seed)
    for p in new_net.parameters():
        noise_t = torch.tensor(np.random.normal(size=p.data.size()).astype(np.float32))
        p.data += NOISE_STD * noise_t
    return new_net


def build_net(seeds):
    torch.manual_seed(seeds[0])
    net = Net(obs_size=OBJ_SIZE*3, act_size=4)
    for seed in seeds[1:]:
        net = mutate_net(net, seed, copy_net=False)
    return net


OutputItem = collections.namedtuple('OutputItem', field_names=['seeds', 'reward', 'steps'])
cache = {}


def worker_func(input_queue, output_queue):
    env = tor.Environment(OBJ_SIZE)
    global cache
    cache = {}

    while True:
        parents = input_queue.get()
        if parents is None:
            break
        new_cache = {}
        for net_seeds in parents:
            if len(net_seeds) > 1:
                net = cache.get(net_seeds[:-1])
                if net is not None:
                    net = mutate_net(net, net_seeds[-1])
                else:
                    net = build_net(net_seeds)
            else:
                net = build_net(net_seeds)
            new_cache[net_seeds] = net
            reward, steps = evaluate(env, net)
            output_queue.put(OutputItem(seeds=net_seeds, reward=reward, steps=steps))
        cache = new_cache


if __name__ == "__main__":
    mp.set_start_method('spawn')

    input_queues = []
    output_queue = mp.Queue(maxsize=WORKERS_COUNT)
    workers = []
    for _ in range(WORKERS_COUNT):
        input_queue = mp.Queue(maxsize=1)
        input_queues.append(input_queue)
        w = mp.Process(target=worker_func, args=(input_queue, output_queue))
        w.start()
        seeds = [(np.random.randint(MAX_SEED),) for _ in range(SEEDS_PER_WORKER)]
        input_queue.put(seeds)

    gen_idx = 0
    elite = None
    for _ in range(EPOCHS):
        t_start = time.time()
        batch_steps = 0
        population = []
        for _ in tqdm(range(POPULATION_SIZE)):
            out_item = output_queue.get()
            population.append((out_item.seeds, out_item.reward))
            batch_steps += out_item.steps
        if elite is not None:
            population.append(elite)
        population.sort(key=lambda p: p[1], reverse=True)
        rewards = [p[1] for p in population[:PARENTS_COUNT]]
        reward_mean = np.mean(rewards)
        reward_max = np.max(rewards)
        reward_std = np.std(rewards)
        speed = batch_steps / (time.time() - t_start)
        print("%d: reward_mean=%.2f, reward_max=%.2f, reward_std=%.2f, speed=%.2f f/s" % (
            gen_idx, reward_mean, reward_max, reward_std, speed))

        elite = population[0]
        net = build_net(population[0][0])
        torch.save(net.state_dict(), 'net.pt')
        print('Elite salvo!')

        for worker_queue in input_queues:
            seeds = []
            for _ in range(SEEDS_PER_WORKER):
                parent = np.random.randint(PARENTS_COUNT)
                next_seed = np.random.randint(MAX_SEED)
                seeds.append(tuple(list(population[parent][0]) + [next_seed]))
            worker_queue.put(seeds)
        gen_idx += 1

    print('Seed: ', population[0][0])
    net = build_net(population[0][0])
    torch.save(net.state_dict(), 'net.pt')
    print('SAVED!')
