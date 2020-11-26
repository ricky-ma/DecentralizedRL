import matlab.engine
import numpy as np


def gen_graph(gtype, N, p):
    eng = matlab.engine.start_matlab()
    mix_matrix = np.asarray(eng.gen_graph(gtype, N, p))
    print(mix_matrix)
    G = Graph(len(mix_matrix))
    for row in range(len(mix_matrix)):
        for col in range(len(mix_matrix)):
            if mix_matrix[row][col] > 0.0:
                G.add_edge(row, col)
    return mix_matrix, G


def decompose(mix_matrix, G):
    E = G.get_edges()
    max_degree = G.max_degree(E)
    print("Max degree: " + str(max_degree))
    coloured_edges = vizing_colouring(E, len(E))
    print("Coloured edges:")
    print(coloured_edges)

    decompositions = []
    colours = num_colours(E)
    for colour in range(1, colours+1):
        subgraph = G.copy()
        for edge in E:
            if edge[2] != colour:
                subgraph.remove_edge(edge[0], edge[1])
        decompositions.append(subgraph)

    result = []
    for graph in decompositions:
        for v1 in range(len(mix_matrix)):
            for v2 in range(len(mix_matrix)):
                if graph.adjMatrix[v1][v2] != 0 or v1 == v2:
                    graph.adjMatrix[v1][v2] = mix_matrix[v1][v2]
        result.append(graph)
    return result


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

    subg = decompose(g, g)
    for i, g in enumerate(subg):
        print("Color: " + str(i + 1))
        g.print_matrix()

    h = Graph(4)
    h.add_edge(0, 1)
    h.add_edge(1, 2)
    h.add_edge(2, 3)
    h.add_edge(3, 0)
    h.add_edge(3, 1)
    subg = decompose(h, h)
    for i, g in enumerate(subg):
        print("Color: " + str(i + 1))
        g.print_matrix()


def goto(linenum):
    global line
    line = linenum


def vizing_colouring(edges, n_colours):
    # Assign a colour to every edge 'i'
    for i in range(n_colours):
        colour = 1
        global line
        line = 1
        while True:
            if line == 1:
                # Assign a colour and then check validity
                edges[i][2] = colour
                for j in range(n_colours):
                    if i == j:
                        continue
                    # If the colour of edges is adjacent to edge i
                    if (
                        edges[i][0] == edges[j][0] or
                        edges[i][1] == edges[j][0] or
                        edges[i][0] == edges[j][1] or
                        edges[i][1] == edges[j][1]
                    ):
                        # If colour matches
                        if edges[i][2] == edges[j][2]:
                            # Increment the colour, denotes change in colour
                            colour += 1
                            # Go back and assign next colour
                            goto(1)
                            break
                else:
                    goto(0)
                    break
    return edges


def num_colours(edges):
    max_colour = 1
    for edge in edges:
        if edge[2] > max_colour:
            max_colour = edge[2]
    return max_colour


class Graph(object):
    # Initialize the matrix
    def __init__(self, size):
        self.adjMatrix = []
        for i in range(size):
            self.adjMatrix.append([0 for i in range(size)])
        self.size = size

    # Add edges
    def add_edge(self, v1, v2):
        if v1 == v2:
            print("Same vertex %d and %d" % (v1, v2))
        # Assign uncoloured edges as -1
        self.adjMatrix[v1][v2] = -1
        self.adjMatrix[v2][v1] = -1

    # Remove edges
    def remove_edge(self, v1, v2):
        if self.adjMatrix[v1][v2] == 0:
            print("No edge between %d and %d" % (v1, v2))
            return
        self.adjMatrix[v1][v2] = 0
        self.adjMatrix[v2][v1] = 0

    def copy(self):
        copy = Graph(self.size)
        for row in range(self.size):
            for col in range(self.size):
                copy.adjMatrix[row][col] = self.adjMatrix[row][col]
        return copy

    def __len__(self):
        return self.size

    # Print the matrix
    def print_matrix(self):
        array = np.asarray(self.adjMatrix)
        print(array)

    def get_edges(self):
        edges = []
        for diag in range(0, len(self.adjMatrix)):
            for row in range(0, len(self.adjMatrix) - diag):
                col = row + diag
                colour = self.adjMatrix[row][col]
                if colour != 0:
                    edges.append([row, col, colour])
        return edges

    def max_degree(self, edges):
        # Map to store the degrees of every node
        leng = len(edges)
        n = self.size
        m = {}
        for i in range(leng):
            m[edges[i][0]] = 0
            m[edges[i][1]] = 0

        for i in range(leng):
            # Storing the degree for each node
            m[edges[i][0]] += 1
            m[edges[i][1]] += 1

        max_degree = 0
        for i in range(n):
            max_degree = max(max_degree, m[i])
        return max_degree
