from src import graph
import numpy as np
from numpy import linalg, random
from scipy.special import softmax

'''
simulation setup
'''
NUM_AGENTS = 6  # number of agents
NUM_ACTIONS = 3  # actions per agent
TOTAL_ACTIONS = np.power(NUM_ACTIONS, NUM_AGENTS)  # total actions
TOTAL_STATES = 20  # total states
DIM_PHI = TOTAL_ACTIONS * TOTAL_STATES  # |S||A|
DIM_FEATURES = 30  # value of d
VEC_EPSILON = random.rand(TOTAL_ACTIONS, 1)
INIT_STATE = random.randint(TOTAL_STATES)
TOTAL_SAMPLES = 1e6
STEPSIZE = 0.005
DISCOUNT = 0.5
'''
1. sample generation
'''
# 1.1 transition matrix
mtx_prob_ssa = np.ones((TOTAL_STATES, TOTAL_STATES, TOTAL_ACTIONS))/TOTAL_STATES
# 1.2 reward generation
reward_as = random.rand(DIM_PHI, NUM_AGENTS)
# 1.3 behavior policy
vec_prob_as = softmax(VEC_EPSILON)
cdf_prob_as = np.cumsum(vec_prob_as)
'''
2. mixing matrix generation
'''
mix_matrix, G = graph.gen_graph('ER', 10, 0.7)
subgraphs = graph.decompose(mix_matrix, G)
# ww_star = np.mean(init_weight, 2)
# 3. feature generation
feature = np.cos(np.pi*random.rand(DIM_FEATURES, DIM_PHI))
feature = feature / (np.ones((DIM_FEATURES, 1)) * linalg.norm(feature, ord=2))


def train(eta=STEPSIZE, gamma=DISCOUNT, BB=mix_matrix, phi=feature, temp_eta=(60, 80, 100)):
    # TODO
    init_state = INIT_STATE
    init_weight = np.zeros((DIM_FEATURES, NUM_AGENTS))
    init_reward = 0
    for xxx in range(len(temp_eta)):
        for ii in range(int(TOTAL_SAMPLES)):
            eta = temp_eta[xxx]/(ii + 3e5)/(1+gamma)

            # 4.1 find action
            rnd_tos = random.uniform(0, 1)
            idx_action_curr = np.argwhere(rnd_tos <= cdf_prob_as, 1)
            idx_state_curr = init_state

            # 4.2 find next state
            cdf_prob_ssa = np.cumsum(mtx_prob_ssa[:, idx_state_curr, idx_action_curr])
            rnd_tos = random.uniform(0, 1)
            idx_state_next = np.argwhere(rnd_tos <= cdf_prob_ssa, 1)

            # 4.3 find next action
            range_phi = idx_state_next * TOTAL_ACTIONS + np.asarray(range(1, TOTAL_ACTIONS))
            # for jj in range(NUM_AGENTS):


if __name__ == '__main__':
    mix_matrix, G = graph.gen_graph('ER', 10, 0.7)
    G.print_matrix()
    subgraphs = graph.decompose(mix_matrix, G)
    for i, graph in enumerate(subgraphs):
        print("Color: " + str(i + 1))
        graph.print_matrix()
    print("hello")
