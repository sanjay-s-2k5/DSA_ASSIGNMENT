import networkx as nx
import matplotlib.pyplot as plt
import heapq

# Create a weighted undirected graph
graph = {
    'A': {'B': 1, 'C': 3},
    'B': {'A': 1, 'C': 1, 'D': 4},
    'C': {'A': 3, 'B': 1, 'D': 2},
    'D': {'B': 4, 'C': 2}
}

# Prim's Algorithm Implementation
def prim(graph, start):
    mst = []
    visited = set([start])
    edges = [(weight, start, to) for to, weight in graph[start].items()]
    heapq.heapify(edges)

    while edges:
        weight, frm, to = heapq.heappop(edges)
        if to not in visited:
            visited.add(to)
            mst.append((frm, to, weight))
            for to_next, weight in graph[to].items():
                if to_next not in visited:
                    heapq.heappush(edges, (weight, to, to_next))

    return mst

# Kruskal's Algorithm Implementation
class DisjointSet:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, item):
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, set1, set2):
        root1 = self.find(set1)
        root2 = self.find(set2)

        if root1 != root2:
            if self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            elif self.rank[root1] < self.rank[root2]:
                self.parent[root1] = root2
            else:
                self.parent[root2] = root1
                self.rank[root1] += 1

    def add(self, item):
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0

def kruskal(graph):
    edges = []
    for from_node, neighbors in graph.items():
        for to_node, weight in neighbors.items():
            edges.append((weight, from_node, to_node))
    edges = sorted(edges)

    ds = DisjointSet()
    mst = []

    for edge in edges:
        weight, frm, to = edge
        ds.add(frm)
        ds.add(to)

        if ds.find(frm) != ds.find(to):
            ds.union(frm, to)
            mst.append(edge)

    return mst

# Running Prim's Algorithm
prim_mst = prim(graph, 'A')
print("Prim's Minimum Spanning Tree:")
for edge in prim_mst:
    print(edge)

# Running Kruskal's Algorithm
kruskal_mst = kruskal(graph)
print("\nKruskal's Minimum Spanning Tree:")
for edge in kruskal_mst:
    print(edge)

# Visualization of the graph and Prim's MST
def visualize_prim(graph, prim_edges):
    G = nx.Graph()
    for node, edges in graph.items():
        for neighbor, weight in edges.items():
            G.add_edge(node, neighbor, weight=weight)

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', font_weight='bold')
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Highlight Prim's MST edges
    prim_graph = nx.Graph()
    for frm, to, weight in prim_edges:
        prim_graph.add_edge(frm, to, weight=weight)

    nx.draw(prim_graph, pos, edge_color='lightgreen', width=2, with_labels=True)
    plt.title("Graph Visualization with Prim's MST")
    plt.show()

# Visualization of the graph and Kruskal's MST
def visualize_kruskal(graph, kruskal_edges):
    G = nx.Graph()
    for node, edges in graph.items():
        for neighbor, weight in edges.items():
            G.add_edge(node, neighbor, weight=weight)

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', font_weight='bold')
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Highlight Kruskal's MST edges
    kruskal_graph = nx.Graph()
    for weight, frm, to in kruskal_edges:
        kruskal_graph.add_edge(frm, to, weight=weight)

    nx.draw(kruskal_graph, pos, edge_color='orange', width=2, with_labels=True)
    plt.title("Graph Visualization with Kruskal's MST")
    plt.show()

# Visualizing Prim's MST
visualize_prim(graph, prim_mst)

# Visualizing Kruskal's MST
visualize_kruskal(graph, kruskal_mst)
