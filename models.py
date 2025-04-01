import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.covariance import GraphicalLasso
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import cvxpy as cp
import os
import time  # For runtime tracking
from matplotlib.patches import Patch



os.makedirs('plots', exist_ok=True)
print("-" *20)
print("\nStarting the IoT Graph Analysis Project...")

# Load data
print("\n-- Loading Node-Location Data...")
node_location = pd.read_csv('Node-Location.csv', delimiter=';', skipinitialspace=True)
print(f"-- Loaded {len(node_location)} nodes.")

print("\n-- Loading Sensor Data Matrix...")
data_matrix = pd.read_csv('data_matrix_12-stations.csv', delimiter=';', skipinitialspace=True)
data_matrix.columns = data_matrix.columns.str.strip()
data_matrix_numeric = data_matrix.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').dropna()
print(f"-- Processed data matrix with {data_matrix_numeric.shape[0]} samples and {data_matrix_numeric.shape[1]} sensor nodes.")
# print(f"data_matrix_numeric shape: {data_matrix_numeric.shape}")  # DEBUG print

def print_data_stats(data):
    print("\nSensor Data Statistics:")
    print(f"Shape: {data.shape}")
    print(f"Mean: {data.mean().mean():.4f}")
    print(f"Std Dev: {data.std().mean():.4f}")
    print(f"Min: {data.min().min():.4f}")
    print(f"Max: {data.max().max():.4f}")

print_data_stats(data_matrix_numeric)

def haversine_distance(coord1, coord2):
    # Convert degrees to radians
    lat1, lon1 = np.radians(coord1)
    lat2, lon2 = np.radians(coord2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    R = 6371  # Earth radius in kilometers
    return R * c

# Function to calculate Distance Matrix using Haversine formula
def calculate_distance_matrix(locations):
    print("\n Calculating Distance Matrix using Haversine formula...")
    coords = locations[['Lat', 'Lon']].values
    n = len(coords)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            distance = haversine_distance(coords[i], coords[j])
            distance_matrix[i, j] = distance_matrix[j, i] = distance
    print("-- Distance matrix calculated.")
    print(f"Number of nodes: {n}")
    # Only consider nonzero distances for stats:
    d_nonzero = distance_matrix[distance_matrix > 0]
    print(f"Pairwise distance - min: {d_nonzero.min():.4f}, max: {distance_matrix.max():.4f}, mean: {d_nonzero.mean():.4f}, std: {d_nonzero.std():.4f}")
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

def distance_based_graph(locations, theta_values=None, t_values=[0.5, 0.8, 1, 1.5, 2, 5, 10],verbose=True):
    print("\n+++ Constructing Distance-Based Graphs...")
    distances = calculate_distance_matrix(locations)
    # If no theta_values provided, compute dynamic theta based on std.
    if theta_values is None:
        theta_std = np.std(distances[np.triu_indices_from(distances, k=1)])
        multipliers = np.array([0.333, 0.5, 0.666, 0.8, 1.0, 1.2, 1.5, 1.75, 2.0], dtype=float)
        theta_values = list(multipliers*theta_std)
        print(f"Dynamically set theta_values: {theta_values}")
    results = []
    graphs = {}
    for theta in theta_values:
        for T in t_values:
            start_time = time.time()
            # Use squared distances in the Gaussian kernel
            W = np.exp(-np.power(distances, 2) / (2 * theta))
            W[distances > T] = 0
            np.fill_diagonal(W, 0)
            if verbose:
                print(f"\n[DEBUG] Weight Matrix for θ={theta:.6f}, T={T}:")
                print(W)
            G = nx.from_numpy_array(W)
            run_time = time.time() - start_time
            metrics = compute_graph_metrics(G)
            if verbose:    
                print(f"  ### θ={theta:.6f}, T={T} -> Edges: {metrics['Edges']}, Edge Density: {metrics['Edge Density']:.6f}, Time: {run_time:.3f}s")
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
def select_optimal_pca_components(data, auto_pca_lambda, min_components=2, max_components=None):
    """
    Evaluate different numbers of PCA components and choose the one that yields
    the maximum number of edges (using a baseline Graphical Lasso with auto_pca_lambda).
    """
    scaler = StandardScaler()
    X_std = scaler.fit_transform(data)
    if max_components is None:
        max_components = X_std.shape[1]
        
    best_components = max_components
    best_edges = -1
    print("\nEvaluating different PCA component choices:")
    for n in range(max_components, min_components - 1, -1):
        pca = PCA(n_components=n)
        X_pca = pca.fit_transform(X_std)
        S_pca = np.cov(X_pca, rowvar=False)
        cond_S_pca = np.linalg.cond(S_pca)
        
        # Run Graphical Lasso with the baseline lambda for this PCA reduction.
        model = GraphicalLasso(alpha=auto_pca_lambda, max_iter=500, tol=1e-3)
        model.fit(X_pca)
        W = np.abs(model.precision_)
        np.fill_diagonal(W, 0)
        # Count edges from the upper triangle
        num_edges = int(np.sum(W > 1e-5) / 2)
        
        print(f"  n_components={n}: Condition number: {cond_S_pca:.4f}, Edges (λ={auto_pca_lambda}): {num_edges}")
        
        if num_edges > best_edges:
            best_edges = num_edges
            best_components = n
            
    print(f"-- Selected optimal PCA components: {best_components} (Edges: {best_edges})")
    return best_components

def graphical_lasso_with_and_without_pca(data, lambda_values, auto_pca_lambda=0.000001, min_components=2, verbose=True):
    """
    Constructs Graphical Lasso graphs on:
      1. Original standardized data, and
      2. PCA-reduced data with an automatically chosen number of components.
    
    Returns a tuple (combined_results, combined_graphs) where:
      - combined_results is a list of result dictionaries, and
      - combined_graphs is a dictionary mapping filenames to graph objects.
    """
    print("\n+++ Constructing GLASSO Graphs (Original vs. Automatic PCA)...")
    
    # Standardize data once
    scaler = StandardScaler()
    X_std = scaler.fit_transform(data)
    
    # --- Original Data (No PCA) ---
    if verbose:    
        print("\n [Original Data] Without PCA:")
    S_orig = np.cov(X_std, rowvar=False)
    cond_S_orig = np.linalg.cond(S_orig)
    if verbose:    
        print(f"Condition number of original covariance matrix: {cond_S_orig:.4f}")
    
    results_orig = []
    graphs_orig = {}
    for lambd in lambda_values:
        start_time = time.time()
        model = GraphicalLasso(alpha=lambd, max_iter=500, tol=1e-3)
        model.fit(X_std)
        W = np.abs(model.precision_)
        np.fill_diagonal(W, 0)
        G = nx.from_numpy_array(W)
        run_time = time.time() - start_time
        metrics = compute_graph_metrics(G)
        if verbose:    
            print(f"  ### λ={lambd} -> Edges: {metrics['Edges']}, Edge Density: {metrics['Edge Density']:.6f}, Time: {run_time:.3f}s")
        filename = f"plots/glasso_graph_ORIG_lambda_{lambd}.png"
        graphs_orig[filename] = G
        results_orig.append({
            "Method": "GLASSO (Original)",
            "Lambda": lambd,
            **metrics,
            "Run Time (s)": run_time
        })
    
    # --- Automatic PCA Selection ---
    optimal_components = select_optimal_pca_components(data, auto_pca_lambda, min_components, max_components=data.shape[1])
    pca = PCA(n_components=optimal_components)
    X_pca = pca.fit_transform(X_std)
    S_pca = np.cov(X_pca, rowvar=False)
    cond_S_pca = np.linalg.cond(S_pca)
    if verbose:    
        print("\n [PCA-Reduced Data]:")
        print(f"Condition number after PCA: {cond_S_pca:.4f}")
        print(f"PCA reduced data shape: {X_pca.shape} (original shape was {X_std.shape})")
    
    results_pca = []
    graphs_pca = {}
    for lambd in lambda_values:
        start_time = time.time()
        model = GraphicalLasso(alpha=lambd, max_iter=500, tol=1e-3)
        model.fit(X_pca)
        W = np.abs(model.precision_)
        np.fill_diagonal(W, 0)
        G = nx.from_numpy_array(W)
        run_time = time.time() - start_time
        metrics = compute_graph_metrics(G)
        if verbose:    
            print(f"  ### λ={lambd} -> Edges: {metrics['Edges']}, Edge Density: {metrics['Edge Density']:.6f}, Time: {run_time:.3f}s")
            print(f"\n[DEBUG] Precision Matrix for λ={lambd}:")
            print(model.precision_)  # prints first 5 rows and columns
        filename = f"plots/glasso_graph_PCA_lambda_{lambd}.png"
        graphs_pca[filename] = G
        results_pca.append({
            "Method": "GLASSO (PCA)",
            "Lambda": lambd,
            **metrics,
            "Run Time (s)": run_time
        })
        
    
    # Combine results from both approaches
    combined_results = results_orig + results_pca
    combined_graphs = {}
    combined_graphs.update(graphs_orig)
    combined_graphs.update(graphs_pca)
    return combined_results, combined_graphs


def laplacian_graph(data, alpha_values, beta_values, threshold=0.01, verbose=True):
    print("\n+++ Constructing Laplacian-Based Graphs...")
    scaler = StandardScaler()
    X = scaler.fit_transform(data)
    results = []
    graphs = {}
    for alpha in alpha_values:
        for beta in beta_values:
            ratio = beta / alpha  # Calculate the beta/alpha ratio
            start_time = time.time()
            Y = cp.Variable((X.shape[0], X.shape[1]))
            obj = cp.Minimize(cp.norm(X - Y, 'fro')**2 + beta * cp.norm(Y, 'fro')**2)
            problem = cp.Problem(obj)
            problem.solve()
            if Y.value is not None:
                Y_opt = Y.value
                Y_nodes = Y_opt.T  # rows become nodes
                D_squared = np.sum(np.square(Y_nodes[:, None, :] - Y_nodes[None, :, :]), axis=2)
                W = np.exp(-D_squared / (2 * alpha))
                np.fill_diagonal(W, 0)
                W[W < threshold] = 0
                G = nx.from_numpy_array(W)
                run_time = time.time() - start_time
                metrics = compute_graph_metrics(G)
                if verbose:
                    print(f"  ### α={alpha}, β={beta} (ratio={ratio:.4f}) -> Edges: {metrics['Edges']}, Edge Density: {metrics['Edge Density']:.6f}, Time: {run_time:.3f}s")
                filename = f"plots/laplacian_graph_alpha_{alpha}_beta_{beta}.png"
                graphs[filename] = G
                results.append({
                    "Method": "Laplacian-Based",
                    "Alpha": alpha,
                    "Beta": beta,
                    "Ratio": ratio,
                    **metrics,
                    "Run Time (s)": run_time
                })
                if verbose:
                    print(f"\n[DEBUG] Laplacian-Based Weight Matrix for α={alpha}, β={beta}:")
                    print(W)

    return results, graphs
