

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

    def __len__(self):
        return self.size

    # Print the matrix
    def print_matrix(self):
        for row in self.adjMatrix:
            for val in row:
                print('{:4}'.format(val)),
            print()

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

    def colour_edges(self, edges, num_colours):
        # Assign a colour to every edge 'i'
        for i in range(num_colours):
            colour = 1
            global line
            line = 1
            while True:
                if line == 1:
                    # Assign a colour and then check validity
                    edges[i][2] = colour
                    for j in range(num_colours):
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
                                self.goto(1)
                                break
                    else:
                        self.goto(0)
                        break
        return edges

    @staticmethod
    def num_colours(edges):
        max_colour = 1
        for edge in edges:
            if edge[2] > max_colour:
                max_colour = edge[2]
        return max_colour

    @staticmethod
    def goto(linenum):
        global line
        line = linenum


def main():
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

    edges = g.get_edges()
    max_degree = g.max_degree(edges)
    print("Max degree: " + str(max_degree))
    coloured_edges = g.colour_edges(edges, len(edges))
    print(coloured_edges)
    # every simple undirected graph may be edge colored with d+1 colours given a graph with a max degree 'd'
    assert max_degree+1 >= g.num_colours(edges)

    h = Graph(4)
    h.add_edge(0, 1)
    h.add_edge(1, 2)
    h.add_edge(2, 3)
    h.add_edge(3, 0)
    h.add_edge(3, 1)

    edges = h.get_edges()
    max_degree = h.max_degree(edges)
    print("Max degree: " + str(max_degree))
    coloured_edges = h.colour_edges(edges, len(edges))
    print(coloured_edges)
    # every simple undirected graph may be edge colored with d+1 colours given a graph with a max degree 'd'
    assert max_degree+1 >= h.num_colours(edges)


if __name__ == '__main__':
    main()
