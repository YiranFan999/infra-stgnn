##          -> counter -> node1
## -> parser
##          -> matcher -> node2
import os

# Simulation parameters
T_STABLE = 500
T_SIM     = 5 * 3000
DT_MONITOR = 5

LAMBDA_NORMAL = 5
LAMBDA_BURST  = 15
BURST_PROB    = 0.02   # 2% probability for burst
BURST_DURATION = 20    # duration of burst

MU = {
    'parser':  10,
    'counter': 6,
    'matcher': 5,
    'node1':   8,
    'node2':   8,
}

P_COUNTER = 0.5   # probability for parser→counter，1-P_COUNTER routing to matcher

import simpy
import random
import numpy as np

env = simpy.Environment()
parser  = simpy.Resource(env, capacity=4)
counter = simpy.Resource(env, capacity=4)
node1 = simpy.Resource(env, capacity=4)
node2 = simpy.Resource(env, capacity=4)
matcher = simpy.Resource(env, capacity=4)



def job(env, servers, record):
    t_arrive = env.now

    record['parser_arrivals'].append(env.now)
    # parser
    with servers['parser'].request() as req_parser:
        yield req_parser
        yield env.timeout(random.expovariate(MU['parser']))
    record['parser_delay'].append(env.now - t_arrive)
    t_arrive = env.now

    # decide next step: counter or matcher
    if random.random() < P_COUNTER:
        record['counter_arrivals'].append(env.now)
        with servers['counter'].request() as req_counter:
            yield req_counter
            yield env.timeout(random.expovariate(MU['counter']))
        record['counter_delay'].append(env.now - t_arrive)
        t_arrive = env.now

        record['node1_arrivals'].append(env.now)
        with (servers['node1'].request()) as req_node1:
            yield req_node1
            yield env.timeout(random.expovariate(MU['node1']))
        record['node1_delay'].append(env.now - t_arrive)
    else:
        record['matcher_arrivals'].append(env.now)
        with servers['matcher'].request() as req_matcher:
            yield req_matcher
            yield env.timeout(random.expovariate(MU['matcher']))
        record['matcher_delay'].append(env.now - t_arrive)
        t_arrive = env.now

        record['node2_arrivals'].append(env.now)
        with (servers['node2'].request()) as req_node2:
            yield req_node2
            yield env.timeout(random.expovariate(MU['node2']))
        record['node2_delay'].append(env.now - t_arrive)

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


def monitor(env, servers, record, snapshot=None):
    if snapshot is None:
        snapshot = []
    while True:
        data = []
        t_window_start = env.now - DT_MONITOR
        for node in MU.keys():
            s = servers[node]
            record[f'{node}_arrivals'] = [t for t in record[f'{node}_arrivals']
                                          if t >= t_window_start]
            arrival_rate = len(record[f'{node}_arrivals']) / DT_MONITOR
            avg_delay = np.mean(record[f'{node}_delay']) if record[f'{node}_delay'] else 0
            record[f'{node}_delay'].clear()

            data.append([
                len(s.queue) + s.count, # queue length
                s.count / s.capacity, # utilisation
                arrival_rate,
                avg_delay
            ])
        snapshot.append(data)
        if len(snapshot) % 300 == 0:
            print(f"snapshot {len(snapshot)}")
        yield env.timeout(DT_MONITOR)

if __name__ == '__main__':
    snapshot = []
    servers = {
        'parser': parser,
        'counter': counter,
        'matcher': matcher,
        'node1': node1,
        'node2': node2,
    }
    record = {
        'parser_arrivals': [], 'parser_delay': [],
        'counter_arrivals': [], 'counter_delay': [],
        'matcher_arrivals': [], 'matcher_delay': [],
        'node1_arrivals': [], 'node1_delay': [],
        'node2_arrivals': [], 'node2_delay': [],
    }

    env.process(arrivals(env, servers, record))
    env.process(monitor(env, servers, record, snapshot=snapshot))

    env.run(until=T_STABLE)
    for key in record:
        record[key].clear()
    snapshot.clear()
    env.run(until=T_SIM + T_STABLE)
    arr = np.array(snapshot)  # (T, N, F)
    print(arr.shape)
    os_path = os.path.join(os.path.dirname(__file__), 'mock_data.npy')
    np.save(os_path, arr)