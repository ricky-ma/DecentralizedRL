from code import graph
from code.model import TD
import numpy as np
import matplotlib.pyplot as plt


def train(model, num_agents, subgraphs, verbose=0, iterations=100):
    # for sample in range(int(model.TOTAL_SAMPLES)):
    for sample in range(iterations):
        init_state = model.INIT_STATE
        # eta = 100/(sample + 3e5)/(1+model.DISCOUNT)
        eta = 0.00001
        if verbose >= 1:
            print("Iteration: {}".format(sample))
            print("Stepsize: {}".format(eta))
        for k, subgraph in enumerate(subgraphs):
            if verbose >= 2:
                print("    Subgraph: {}".format(k))

            # find current action, state
            rnd_tos = np.random.uniform(0, 1)
            idx_action_curr = np.argwhere(rnd_tos <= model.cdf_prob_as)[0][0]
            idx_state_curr = init_state

            # find next state
            cdf_prob_ssa = np.cumsum(model.mtx_prob_ssa[:, idx_state_curr, idx_action_curr])
            rnd_tos = np.random.uniform(0, 1)
            idx_state_next = np.argwhere(rnd_tos <= cdf_prob_ssa)[0][0]

            # find next action
            range_phi = (idx_state_next - 1) * model.TOTAL_ACTIONS + np.arange(0, model.TOTAL_ACTIONS)
            idx_action_pred = np.empty(shape=(num_agents, 1))
            for agent in range(num_agents):
                idx_action_pred[agent] = np.argmax(model.feature[:, range_phi].conj().T @ model.WEIGHT[:, agent])

            idx_feature_curr = (idx_state_curr - 1) * model.TOTAL_ACTIONS + idx_action_curr
            idx_feature_pred = (idx_state_next - 1) * model.TOTAL_ACTIONS + idx_action_pred

            # print(idx_feature_curr.shape)
            # print(idx_feature_curr)
            # print(idx_action_pred.shape)
            # print(idx_action_pred)
            # print(idx_feature_pred.shape)
            # print(idx_feature_pred)

            # parameter updates
            model.update(subgraph=subgraph,
                         idx_feature_curr=idx_feature_curr,
                         idx_feature_pred=idx_feature_pred.astype(int),
                         eta=eta)
            if verbose >= 2:
                print("    Error: {}".format(model.ERROR[-1]))
            model.INIT_STATE = idx_state_next


if __name__ == '__main__':
    mix_matrix, G = graph.gen_graph('ER', 10, 0.7)
    G.print_matrix()
    SG = graph.decompose(mix_matrix, G)
    for i, graph in enumerate(SG):
        print("Color: {}".format(i + 1))
        graph.print_matrix()
    td_model = TD(SG)
    train(model=td_model, num_agents=G.size, subgraphs=SG, verbose=2, iterations=25)
    plt.plot(td_model.ERROR)
    plt.show()
