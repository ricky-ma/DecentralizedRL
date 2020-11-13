from code import graph
from code.model import DecentralizedTD
import numpy as np
import matplotlib.pyplot as plt


def train(model, verbose=0, iterations=100):
    for k in range(iterations):
        # eta = 100/(k + 3e5)/(1+model.DISCOUNT_GAMMA)
        eta = 0.00001
        if verbose >= 1:
            print("Subgraph epoch: {}".format(k))
            print("Stepsize: {}".format(eta))

        for m, subgraph in enumerate(model.SUBGRAPHS):
            if verbose >= 2:
                print("    Subgraph: {}".format(m))

            # find next state
            cdf_prob_ssa = np.cumsum(model.mtx_prob_ssa[:, model.STATE])
            rnd_tos = np.random.uniform(0, 1)
            idx_state_next = np.nonzero(rnd_tos <= cdf_prob_ssa)[0][0]

            phi = model.feature[:, model.STATE]
            phip = model.feature[:, idx_state_next]

            # parameter updates
            model.update(
                subgraph=subgraph,
                phi=phi,
                phip=phip,
                eta=eta
            )
            model.STATE = idx_state_next
            if verbose >= 2:
                print("    Error: {}".format(model.ERROR[-1]))


if __name__ == '__main__':
    mix_matrix, G = graph.gen_graph('ER', 50, 0.7)
    G.print_matrix()
    # SG = graph.decompose(mix_matrix, G)
    # for i, graph in enumerate(SG):
    #     print("Color: {}".format(i + 1))
    #     graph.print_matrix()
    # SG = [G]
    td_model = DecentralizedTD([G])
    train(model=td_model, verbose=2, iterations=100)
    plt.semilogx(td_model.ERROR)
    plt.show()
