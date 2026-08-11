##                  -> counter -> node1
## source -> parser
##                  -> matcher -> node2
import os

# Simulation parameters
T_BURN_IN = 500
T_SIM     = 30000
DT_MONITOR = 3

LAMBDA_NORMAL = 5
LAMBDA_BURST  = 40
BURST_PROB    = 0.02   # 每个arrival有2%概率触发burst
BURST_DURATION = 50    # burst持续多少个time unit

MU = {
    'parser':  10,
    'counter': 6,
    'matcher': 5,
    'node1':   8,
    'node2':   8,
}

P_COUNTER = 0.5   # parser→counter的概率，1-P_COUNTER去matcher

import simpy
import random
import numpy as np

# todo: different capacity
env = simpy.Environment()
parser  = simpy.Resource(env, capacity=1) # single server for now
counter = simpy.Resource(env, capacity=1)
node1 = simpy.Resource(env, capacity=1)
node2 = simpy.Resource(env, capacity=1)
matcher = simpy.Resource(env, capacity=1)



def job(env, servers, record):
    t_arrive = env.now

    # # source
    # with servers['source'].request() as req_source:
    #     yield req_source
    #     yield env.timeout(random.expovariate(MU['source']))
    # time_after_source = env.now
    # source_delay = env.now - t_arrive

    # parser
    with servers['parser'].request() as req_parser:
        yield req_parser
        yield env.timeout(random.expovariate(MU['parser']))
        time_after_parser = env.now
        parser_delay = env.now - t_arrive

    # decide next step: counter or matcher
    if random.random() < P_COUNTER:
        with servers['counter'].request() as req_counter:
            yield req_counter
            yield env.timeout(random.expovariate(MU['counter']))
        time_after_counter = env.now
        counter_delay = env.now - time_after_parser

        with (servers['node1'].request()) as req_node1:
            yield req_node1
            yield env.timeout(random.expovariate(MU['node1']))
        time_after_node1 = env.now
        node1_delay = env.now - time_after_counter
    else:
        with servers['matcher'].request() as req_matcher:
            yield req_matcher
            yield env.timeout(random.expovariate(MU['matcher']))
        time_after_matcher = env.now
        matcher_delay = env.now - time_after_parser

        with (servers['node2'].request()) as req_node2:
            yield req_node2
            yield env.timeout(random.expovariate(MU['node2']))
        time_after_node2 = env.now
        node2_delay = env.now - time_after_matcher

def arrivals(env, servers, record):
    burst_end = -1
    while True:
        if env.now < burst_end:
            lam = LAMBDA_BURST
        else:
            if random.random() < BURST_PROB:
                burst_end = env.now + BURST_DURATION
                lam = LAMBDA_BURST
            else:
                lam = LAMBDA_NORMAL

        yield env.timeout(random.expovariate(lam))
        env.process(job(env, servers, record))

# todo: ave_delay for each node
def monitor(env, servers, record):
    while True:
        snapshot = []
        for node in MU.keys():
            s = servers[node]
            snapshot.append([
                len(s.queue) + s.count, # queue length
                s.count / s.capacity, # utilisation
            ])
        record.append(snapshot)
        yield env.timeout(DT_MONITOR)

if __name__ == '__main__':
    data = []
    servers = {
        'parser': parser,
        'counter': counter,
        'matcher': matcher,
        'node1': node1,
        'node2': node2,
    }
    env.process(arrivals(env, servers, data))
    env.process(monitor(env, servers, data))

    env.run(until=T_BURN_IN)
    data.clear()
    env.run(until=T_SIM+T_BURN_IN)
    arr = np.array(data)  # (T, N, F)
    print(arr.shape)
    os_path = os.path.join(os.path.dirname(__file__), 'mock_data.npy')
    np.save(os_path, arr)