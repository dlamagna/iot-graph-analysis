import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.covariance import GraphicalLasso
from sklearn.preprocessing import StandardScaler
import cvxpy as cp
import os
import time  # For runtime tracking

# Ensure the 'plots' directory exists
os.makedirs('plots', exist_ok=True)

print("\n🚀 Starting the IoT Graph Analysis Project...")

# Load data
print("\n📥 Loading Node-Location Data...")
node_location = pd.read_csv('Node-Location.csv', delimiter=';', skipinitialspace=True)
print(f"✅ Loaded {len(node_location)} nodes.")

print("\n📥 Loading Sensor Data Matrix...")
data_matrix = pd.read_csv('data_matrix_12-stations.csv', delimiter=';', skipinitialspace=True)
data_matrix.columns = data_matrix.columns.str.strip()
data_matrix_numeric = data_matrix.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').dropna()
print(f"✅ Processed data matrix with {data_matrix_numeric.shape[0]} samples and {data_matrix_numeric.shape[1]} sensor nodes.")
# print(f"ℹ️ data_matrix_numeric shape: {data_matrix_numeric.shape}")  # DEBUG print

# Function to calculate Distance Matrix using Haversine formula
def calculate_distance_matrix(locations):
    print("\n📏 Calculating Distance Matrix using Haversine formula...")
    coords = locations[['Lat', 'Lon']].values
    n = len(coords)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            distance = np.linalg.norm(coords[i] - coords[j])
            distance_matrix[i, j] = distance_matrix[j, i] = distance
    print("✅ Distance matrix calculated.")
    return distance_matrix

# Function to compute key graph metrics
def compute_graph_metrics(G):
    num_nodes = len(G.nodes)
    num_edges = G.number_of_edges()

    if num_edges == 0:
        return {
            "Edges": num_edges,
            "Edge Density": 0,
            "Connected Components": num_nodes,
            "Clustering Coefficient": 0,
            "Average Path Length": None,
            "Diameter": None,
            "Assortativity": None,
            "Average Degree": 0,
            "Degree Variance": 0
        }

    edge_density = num_edges / (num_nodes * (num_nodes - 1))
    num_components = nx.number_connected_components(G)
    clustering_coeff = nx.average_clustering(G)
    assortativity = nx.degree_assortativity_coefficient(G)
    degrees = [d for _, d in G.degree()]
    avg_degree = sum(degrees) / num_nodes
    degree_variance = np.var(degrees)

    if nx.is_connected(G):
        avg_path_length = nx.average_shortest_path_length(G)
        diameter = nx.diameter(G)
    else:
        avg_path_length = None
        diameter = None

    return {
        "Edges": num_edges,
        "Edge Density": edge_density,
        "Connected Components": num_components,
        "Clustering Coefficient": clustering_coeff,
        "Average Path Length": avg_path_length,
        "Diameter": diameter,
        "Assortativity": assortativity,
        "Average Degree": avg_degree,
        "Degree Variance": degree_variance
    }

# 1. Distance-Based Graph Construction
def distance_based_graph(locations, theta_values=None, t_values=[1, 5, 10]):
    print("\n📌 Constructing Distance-Based Graphs...")
    distances = calculate_distance_matrix(locations)

    if theta_values is None:
        theta_std = np.std(distances[np.triu_indices_from(distances, k=1)])
        theta_values = [theta_std]
        print(f"ℹ️ Using θ = standard deviation of distances: {theta_std:.6f}")

    results = []
    graphs = {}

    for theta in theta_values:
        for T in t_values:
            start_time = time.time()
            
            W = np.exp(-distances / (2 * theta))
            W[distances > T] = 0
            np.fill_diagonal(W, 0) 
            G = nx.from_numpy_array(W) 
            
            run_time = time.time() - start_time

            metrics = compute_graph_metrics(G)
            print(f"  🔹 θ={theta:.6f}, T={T} -> Edges: {metrics['Edges']}, Edge Density: {metrics['Edge Density']:.6f}, Time: {run_time:.3f}s")

            filename = f"plots/distance_graph_theta_{theta:.6f}_T_{T}.png"
            graphs[filename] = G

            results.append({
                "Method": "Distance-Based",
                "Theta": theta,
                "T": T,
                **metrics,
                "Run Time (s)": run_time
            })

    return results, graphs

# 2. GLASSO Graph Construction
def graphical_lasso_graph(data, lambda_values):
    print("\n📌 Constructing GLASSO Graphs...")
    scaler = StandardScaler()
    X = scaler.fit_transform(data)
    # print(f"ℹ️ GLASSO input matrix shape: {X.shape}")  # DEBUG

    results = []
    graphs = {}

    for lambd in lambda_values:
        start_time = time.time()
        model = GraphicalLasso(alpha=lambd)
        model.fit(X)

        W = np.abs(model.precision_)
        np.fill_diagonal(W, 0)
        G = nx.from_numpy_array(W)
        
        run_time = time.time() - start_time

        metrics = compute_graph_metrics(G)
        print(f"  🔹 λ={lambd} -> Edges: {metrics['Edges']}, Edge Density: {metrics['Edge Density']:.6f}, Time: {run_time:.3f}s")

        filename = f"plots/glasso_graph_lambda_{lambd}.png"
        graphs[filename] = G

        results.append({
            "Method": "GLASSO",
            "Lambda": lambd,
            **metrics,
            "Run Time (s)": run_time
        })

    return results, graphs

# 3. Laplacian-Based Graph Construction
def laplacian_graph(data, alpha_values, beta_values, threshold=0.01):
    print("\n📌 Constructing Laplacian-Based Graphs...")
    scaler = StandardScaler()
    X = scaler.fit_transform(data)
    # print(f"ℹ️ Laplacian input matrix shape: {X.shape}")  # DEBUG

    results = []
    graphs = {}

    for alpha in alpha_values:
        for beta in beta_values:
            start_time = time.time()
            Y = cp.Variable((X.shape[0], X.shape[1]))

            obj = cp.Minimize(
                cp.norm(X - Y, 1)**2 +
                beta * cp.norm(Y, 1)**2
            )

            problem = cp.Problem(obj)
            problem.solve()

            if Y.value is not None:
                Y_opt = Y.value  # (samples x nodes) → shape (2170, 12)
                # # print(f"DEBUG: Laplacian Y_opt shape: {Y_opt.shape}")  # DEBUG

                Y_nodes = Y_opt.T  # Now (12, 2170), sensors as rows
                W = np.exp(-np.linalg.norm(Y_nodes[:, None, :] - Y_nodes[None, :, :], axis=2) / (2 * alpha))
                np.fill_diagonal(W, 0)
                W[W < threshold] = 0
                G = nx.from_numpy_array(W)
                
                run_time = time.time() - start_time

                metrics = compute_graph_metrics(G)
                print(f"  🔹 α={alpha}, β={beta} -> Edges: {metrics['Edges']}, Edge Density: {metrics['Edge Density']:.6f}, Time: {run_time:.3f}s")

                filename = f"plots/laplacian_graph_alpha_{alpha}_beta_{beta}.png"
                graphs[filename] = G

                results.append({
                    "Method": "Laplacian-Based",
                    "Alpha": alpha,
                    "Beta": beta,
                    **metrics,
                    "Run Time (s)": run_time
                })

    return results, graphs
