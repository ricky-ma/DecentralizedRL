from code import graph
from code.model import DecentralizedTD
import numpy as np
import matplotlib.pyplot as plt


def train(model, verbose=0):
    iteration = 0
    for k in range(model.EPOCHS):
        # eta = 100/(k + 3e5)/(1+model.DISCOUNT_GAMMA)
        eta = 1e-5
        if verbose >= 1:
            print("Subgraph epoch: {}".format(k))
            print("Stepsize: {}".format(eta))

        for m, subgraph in enumerate(model.SUBGRAPHS):
            if verbose >= 2:
                print("    Iteration: {}".format(iteration))
                print("    Subgraph: {}".format(m))

            # find next state
            cdf_prob_ssa = np.cumsum(model.mtx_prob_ssa[:, model.STATE])
            rnd_tos = np.random.uniform(0, 1)
            idx_state_next = np.where(rnd_tos <= cdf_prob_ssa)[0][0]

            # parameter updates
            error = model.update(
                iteration=iteration,
                subgraph=subgraph,
                phi=model.feature[:, model.STATE],
                phip=model.feature[:, idx_state_next],
                eta=eta
            )
            model.STATE = idx_state_next
            if verbose >= 2:
                print("    Error: {}".format(error))
            iteration += 1


if __name__ == '__main__':
    # first curve: is original decentralized TD (vanilla curve)
    # second curve: is our curve
    num_agent = 10
    mix_matrix, G = graph.gen_graph('ER', num_agent, 0.7)
    G.print_matrix()
    SG = graph.decompose(mix_matrix, G)
    # for i, graph in enumerate(SG):
    #     print("Color: {}".format(i + 1))
    #     graph.print_matrix()
    # SG = [mix_matrix]
    td_model = DecentralizedTD(SG)
    train(model=td_model, verbose=2)
    plt.plot(td_model.ERROR)
    plt.show()
