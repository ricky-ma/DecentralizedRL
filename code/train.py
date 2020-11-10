from code import graph
from code.model import TD
import numpy as np
from numpy import linalg, random
from scipy.special import softmax

'''
simulation setup
'''
NUM_AGENTS = 6  # number of agents
NUM_ACTIONS = 3  # actions per agent
TOTAL_ACTIONS = int(np.power(NUM_ACTIONS, NUM_AGENTS))  # total actions
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
feature = np.cos(np.pi * random.rand(DIM_FEATURES, DIM_PHI))
feature_vec_norm = linalg.norm(feature, ord=2, axis=1)[:, np.newaxis]
# print(vec_norm.shape)
# print(np.ones((DIM_FEATURES, 1)).shape)
# temp = np.ones((DIM_FEATURES, 1)) @ vec_norm.T
# print(temp)
# feature = feature / (np.ones((DIM_FEATURES, 1)) @ feature_vec_norm.T)
feature = feature / feature_vec_norm


def train(eta=STEPSIZE, B=subgraphs, phi=feature):
    # TODO
    init_weight = np.zeros((DIM_FEATURES, NUM_AGENTS))
    init_state = INIT_STATE
    init_reward = 0
    init_phi = feature
    init_phip = 0

    model = TD(len(feature))
    model.w = init_weight

    N = subgraphs[0].size
    for subgraph in B:
        print(subgraph)
        # model.update(b=subgraph, phi=init_phi, phip=init_phip, r=init_reward, eta=eta)
        for n in range(N-1):
            print("iteration: " + str(n+1))

            # find action
            rnd_tos = random.uniform(0, 1)
            idx_action_curr = np.argwhere(rnd_tos <= cdf_prob_as)[0][0]
            idx_state_curr = init_state

            # 4.2 find next state
            cdf_prob_ssa = np.cumsum(mtx_prob_ssa[:, idx_state_curr, idx_action_curr])
            rnd_tos = random.uniform(0, 1)
            idx_state_next = np.argwhere(rnd_tos <= cdf_prob_ssa)[0][0]

            # 4.3 find next action
            range_phi = (idx_state_next-1) * TOTAL_ACTIONS + np.arange(0, TOTAL_ACTIONS)
            idx_action_pred = np.zeros((NUM_AGENTS, 1))
            print(range_phi.shape)
            print(feature[:, range_phi].conj().T.shape)
            print(init_weight.shape)
            # for jj in range(NUM_AGENTS):
            #     print(jj)
            #     idx_action_pred[jj] = np.argmax(feature[:, range_phi].conj().T @ init_weight[:, jj])
            idx_action_pred = np.argmax(feature[:, range_phi].conj().T @ init_weight)
            idx_feature_curr = (idx_state_curr-1)*TOTAL_ACTIONS + idx_action_curr
            idx_feature_pred = (idx_state_next-1)*TOTAL_ACTIONS + idx_action_pred

            print(idx_feature_curr.shape)
            print(idx_feature_curr)
            print(idx_action_pred.shape)
            print(idx_action_pred)
            print(idx_feature_pred.shape)
            print(idx_feature_pred)

            reward = reward_as[idx_feature_curr, n]
            phi = feature[:, idx_feature_curr.astype(int)]
            phip = feature[:, idx_feature_pred.astype(int)]
            model.update(b=subgraph, phi=phi, phip=phip, r=reward, eta=eta)


if __name__ == '__main__':
    mix_matrix, G = graph.gen_graph('ER', 10, 0.7)
    G.print_matrix()
    subgraphs = graph.decompose(mix_matrix, G)
    for i, graph in enumerate(subgraphs):
        print("Color: " + str(i + 1))
        graph.print_matrix()
    print("hello")

    train()
