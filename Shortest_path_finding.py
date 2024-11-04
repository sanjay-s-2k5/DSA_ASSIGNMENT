import networkx as nx
import matplotlib.pyplot as plt
import heapq

# Dijkstra's Algorithm
def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    priority_queue = [(0, start)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        
        if current_distance > distances[current_node]:
            continue
        
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances

# Bellman-Ford Algorithm
def bellman_ford(graph, start, vertices):
    distances = {node: float('inf') for node in vertices}
    distances[start] = 0

    for _ in range(len(vertices) - 1):
        for node in graph:
            for neighbor, weight in graph[node].items():
                if distances[node] != float('inf') and distances[node] + weight < distances[neighbor]:
                    distances[neighbor] = distances[node] + weight

    # Check for negative-weight cycles
    for node in graph:
        for neighbor, weight in graph[node].items():
            if distances[node] + weight < distances[neighbor]:
                return "Graph contains a negative-weight cycle"
                
    return distances

# Floyd-Warshall Algorithm
def floyd_warshall(graph, vertices):
    dist = {v: {u: float('inf') for u in vertices} for v in vertices}
    for v in vertices:
        dist[v][v] = 0
    for v in graph:
        for u, weight in graph[v].items():
            dist[v][u] = weight

    for k in vertices:
        for i in vertices:
            for j in vertices:
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    return dist

# Visualization Function
def visualize_graph(graph, results, algorithm_name, start_node=None):
    G = nx.DiGraph()
    for node, edges in graph.items():
        for neighbor, weight in edges.items():
            G.add_edge(node, neighbor, weight=weight)

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', font_weight='bold')
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Highlight shortest paths or distances
    if isinstance(results, dict):
        for node, distance in results.items():
            if distance != float('inf'):
                nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color='lightgreen')

    plt.title(f"{algorithm_name} Visualization from Node '{start_node}'" if start_node else f"{algorithm_name} All-Pairs Visualization")
    plt.show()

# Example Input Graphs (No negative edge weights)
graph_example = {
    'A': {'B': 1, 'C': 4},
    'B': {'C': 2, 'D': 5},
    'C': {'D': 1},
    'D': {}
}

graph_bellman_example = {
    'A': {'B': 4, 'C': 2},
    'B': {'C': 3, 'D': 2},
    'C': {'D': 2},
    'D': {}
}

graph_fw_example = {
    'A': {'B': 3, 'C': 8, 'D': 4},
    'B': {'D': 1, 'C': 2},
    'C': {'A': 4},
    'D': {'C': 7, 'A': 2}
}
vertices_example = ['A', 'B', 'C', 'D']

# Running and Visualizing Dijkstra's Algorithm
dijkstra_result = dijkstra(graph_example, 'A')
print("Dijkstra's Shortest Path from 'A':", dijkstra_result)
visualize_graph(graph_example, dijkstra_result, "Dijkstra's Algorithm", start_node='A')

# Running and Visualizing Bellman-Ford Algorithm
bellman_result = bellman_ford(graph_bellman_example, 'A', vertices_example)
print("Bellman-Ford Shortest Path from 'A':", bellman_result)
visualize_graph(graph_bellman_example, bellman_result, "Bellman-Ford Algorithm", start_node='A')

# Running and Visualizing Floyd-Warshall Algorithm
floyd_warshall_result = floyd_warshall(graph_fw_example, vertices_example)
print("Floyd-Warshall All-Pairs Shortest Path:", floyd_warshall_result)
visualize_graph(graph_fw_example, floyd_warshall_result, "Floyd-Warshall Algorithm")
