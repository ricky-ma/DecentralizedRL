from code import graph
from code.model import TD
import numpy as np

# TODO: implement error tracing
# ww_star = np.mean(init_weight, 2)


def train(num_agents, subgraphs, eta=0.005):
    model = TD(subgraphs)
    for sample in range(int(model.TOTAL_SAMPLES)):
        init_state = model.INIT_STATE
        for k, subgraph in enumerate(subgraphs):
            print(subgraph)
            print("subgraph: " + str(k))

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
            model.INIT_STATE = idx_state_next


if __name__ == '__main__':
    mix_matrix, G = graph.gen_graph('ER', 10, 0.7)
    G.print_matrix()
    subgraphs = graph.decompose(mix_matrix, G)
    for i, graph in enumerate(subgraphs):
        print("Color: " + str(i + 1))
        graph.print_matrix()
    train(num_agents=G.size, subgraphs=subgraphs)
