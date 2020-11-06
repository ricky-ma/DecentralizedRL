from graph import Graph
import matlab.engine
import numpy as np


def decompose(G):
    E = G.get_edges()
    max_degree = G.max_degree(E)
    print("Max degree: " + str(max_degree))
    coloured_edges = G.vizing_colouring(E, len(E))
    print("Coloured edges:")
    print(coloured_edges)

    decompositions = []
    colours = G.num_colours(E)
    for colour in range(1, colours+1):
        subgraph = G.copy()
        for edge in E:
            if edge[2] != colour:
                subgraph.remove_edge(edge[0], edge[1])
        decompositions.append(subgraph)
    return decompositions


def gen_graph():
    eng = matlab.engine.start_matlab()
    A = np.asarray(eng.gen_graph('ER', 10, 0.7))
    print(A)
    G = Graph(len(A))
    for row in range(len(A)):
        for col in range(len(A)):
            if A[row][col] > 0.0:
                G.add_edge(row, col)
    return G


def graph_test():
    g = Graph(8)
    g.add_edge(0, 1)
    g.add_edge(0, 4)
    g.add_edge(0, 7)
    g.add_edge(1, 2)
    g.add_edge(1, 3)
    g.add_edge(1, 5)
    g.add_edge(1, 7)
    g.add_edge(2, 3)
    g.add_edge(3, 6)
    g.add_edge(3, 7)
    g.add_edge(5, 6)
    g.add_edge(6, 7)

    subg = decompose(g)
    for i, g in enumerate(subg):
        print("Color: " + str(i + 1))
        g.print_matrix()

    h = Graph(4)
    h.add_edge(0, 1)
    h.add_edge(1, 2)
    h.add_edge(2, 3)
    h.add_edge(3, 0)
    h.add_edge(3, 1)
    subg = decompose(h)
    for i, g in enumerate(subg):
        print("Color: " + str(i + 1))
        g.print_matrix()


if __name__ == '__main__':
    G = gen_graph()
    G.print_matrix()
    subgraphs = decompose(G)
    for i, graph in enumerate(subgraphs):
        print("Color: " + str(i + 1))
        graph.print_matrix()
