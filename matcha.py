from graph import Graph, test


def decompose(G, E):
    decompositions = []
    colours = G.num_colours(E)
    for colour in range(1, colours+1):
        subgraph = G.copy()
        for edge in E:
            if edge[2] != colour:
                subgraph.remove_edge(edge[0], edge[1])
        decompositions.append(subgraph)
    return decompositions


if __name__ == '__main__':
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

    edges = test(g)
    subgraphs = decompose(g, edges)
    for i, graph in enumerate(subgraphs):
        print("Color: " + str(i+1))
        graph.print_matrix()

    h = Graph(4)
    h.add_edge(0, 1)
    h.add_edge(1, 2)
    h.add_edge(2, 3)
    h.add_edge(3, 0)
    h.add_edge(3, 1)
    edges = test(h)
    subgraphs = decompose(h, edges)
    for i, graph in enumerate(subgraphs):
        print("Color: " + str(i + 1))
        graph.print_matrix()