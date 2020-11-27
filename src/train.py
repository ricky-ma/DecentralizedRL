from src import graph
from src.model import DecentralizedTD
import numpy as np
import matplotlib.pyplot as plt
seed = 0


def train(model, verbose=0):
    iteration = 1
    for k in range(model.EPOCHS):
        # eta = 100/(k + 3e5)/(1+model.DISCOUNT_GAMMA)
        eta = 1e-6
        if verbose >= 1:
            print("Subgraph epoch: {}".format(k))
            print("Stepsize: {}".format(eta))

        for m, subgraph in enumerate(model.SUBGRAPHS):
            if verbose >= 2:
                print("    Iteration: {}".format(iteration))
                print("    Subgraph: {}".format(m))

            # find next state
            cdf_prob_ssa = np.cumsum(model.mtx_prob_ssa[:, model.STATE])
            # np.random.seed(seed)
            rnd_tos = np.random.uniform(0, 1)
            idx_state_next = np.where(rnd_tos <= cdf_prob_ssa)[0][0]

            # parameter updates
            model.update(
                iteration=iteration,
                subgraph=subgraph,
                phi=model.feature[:, model.STATE],
                phip=model.feature[:, idx_state_next],
                eta=eta
            )
            if verbose >= 2:
                print("    Error: {}".format(model.ERROR[iteration]))
            model.STATE = idx_state_next
            iteration += 1


if __name__ == '__main__':
    # first curve: is original decentralized TD (vanilla curve)
    # second curve: is our curve
    num_agent = 10
    mix_matrix, G = graph.gen_graph('ER', num_agent, 0.7)

    decomposed = graph.decompose(mix_matrix, G)
    subgraphs = []
    for graph in decomposed:
        subgraphs.append(np.asarray(graph.adjMatrix))

    td_model_decomposed = DecentralizedTD(subgraphs)
    train(model=td_model_decomposed, verbose=2)

    td_model_vanilla = DecentralizedTD([mix_matrix])
    train(model=td_model_vanilla, verbose=2)

    iter_to_plot = td_model_vanilla.ERROR.shape[0]

    fig, axs = plt.subplots(2)
    fig.suptitle('Vertically stacked subplots')
    axs[0].plot(td_model_decomposed.ERROR, label="decomposed")
    axs[0].plot(td_model_vanilla.ERROR, label="vanilla")
    axs[1].plot(td_model_decomposed.REWARD, label="decomposed")
    axs[1].plot(td_model_vanilla.REWARD, label="vanilla")

    axs[0].set(xlabel="iteration", ylabel="error")
    axs[0].legend()
    axs[1].set(xlabel="epoch", ylabel="reward")
    axs[1].legend()

    fig.show()

    print(td_model_vanilla.WEIGHT)
    print(td_model_decomposed.WEIGHT)
