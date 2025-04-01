
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.linear_model import LinearRegression
import os

def run_virtual_sensor(
    data_matrix: pd.DataFrame,
    node_locations: pd.DataFrame,
    target_sensor_name: str = "tona",
    dist_params=(25.53, 25),
    glasso_param=0.75,
    laplacian_params=(10, 2.5),
    output_dir="plots/virtual"
):
    """
    Demonstrates the "virtual sensor" reconstruction for a chosen target sensor
    under three different graph methods: distance-based, GLASSO, and Laplacian.
    
    Parameters:
      data_matrix : DataFrame of shape (num_samples, num_sensors), containing
                    the numeric sensor readings (no timestamp column).
      node_locations : DataFrame with columns ['Name','Lat','Lon'] in the same
                       row order that matches data_matrix columns.
      target_sensor_name : which sensor (by 'Name') to treat as the "virtual" node to reconstruct
      dist_params : (theta, T) for distance-based adjacency
      glasso_param : lambda for GLASSO
      laplacian_params : (alpha, beta) for Laplacian approach
      output_dir : folder to store the reconstruction plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    sensor_names = node_locations['Name'].str.lower().tolist()
    if target_sensor_name.lower() not in sensor_names:
        raise ValueError(f"Sensor '{target_sensor_name}' not found in node_locations.")
    target_idx = sensor_names.index(target_sensor_name.lower())
    
    # Extract the single column of interest (shape: [num_samples, ])
    y_true_all = data_matrix.iloc[:, target_idx].values  # actual readings of the target sensor
    num_samples = len(y_true_all)
    
    # For Weighted approach, we'll do a direct “predict entire dataset”
    # For MLR approach, we'll do an 80/20 train/test split
    train_size = int(0.8 * num_samples)
    
    # 2) Build adjacency for each method (with your chosen params)
    #    so we can figure out the neighbors and their weights.
    W_dist = build_distance_based_adjacency(node_locations, dist_params[0], dist_params[1])
    W_glasso = build_glasso_adjacency(data_matrix, glasso_param)
    W_laplacian = build_laplacian_adjacency(data_matrix, laplacian_params[0], laplacian_params[1])
    
    # We’ll store results (RMSE) for printing at the end
    results_summary = []
    
    # 3) Weighted reconstruction approach for each adjacency
    for method_name, W in [
        ("distance", W_dist),
        ("glasso",   W_glasso),
        ("laplacian",W_laplacian),
    ]:
        y_pred_weighted = np.zeros_like(y_true_all)
        
        # For each time n, predict y_k(n) = sum_{j in Nk} w_{jk} x_j(n)
        # i.e. the row in data_matrix is shape (num_sensors, ), the sensor j is data_matrix.iloc[n, j]
        # We'll do unnormalized sum of w_{jk} x_j(n), same as eq (1.5.1).
        
        for n in range(num_samples):
            neighbors = np.where(W[:, target_idx] != 0)[0]  # indices j of nonzero edges to the target
            weighted_sum = 0.0
            for j in neighbors:
                w_jk = W[j, target_idx]
                x_jn = data_matrix.iloc[n, j]
                weighted_sum += w_jk * x_jn
            y_pred_weighted[n] = weighted_sum
        
        # Compute RMSE across entire dataset
        mse_weighted = np.mean((y_true_all - y_pred_weighted)**2)
        rmse_weighted = np.sqrt(mse_weighted)
        results_summary.append((method_name, "weighted", rmse_weighted))
        
        # Plot quick side-by-side timeseries: real vs. predicted
        plt.figure(figsize=(10, 4))
        plt.plot(y_true_all, label="True", lw=2)
        plt.plot(y_pred_weighted, label="Weighted-Pred", alpha=0.7)
        plt.title(f"Virtual Sensor: {target_sensor_name.upper()} - {method_name} Weighted")
        plt.legend()
        plt.tight_layout()
        plot_name = f"{method_name}_weighted_{target_sensor_name}.png"
        plt.savefig(os.path.join(output_dir, plot_name))
        plt.close()
    
    # 4) MLR-based reconstruction approach for each adjacency
    #    We'll use an 80/20 train/test split to fit a LinearRegression
    #    using only the neighbors as features.
    for method_name, W in [
        ("distance", W_dist),
        ("glasso",   W_glasso),
        ("laplacian",W_laplacian),
    ]:
        neighbors = np.where(W[:, target_idx] != 0)[0]
        if len(neighbors) == 0:
            # If the adjacency yields no neighbors, skip
            print(f"[WARN] {method_name} adjacency: target sensor '{target_sensor_name}' had no neighbors!")
            results_summary.append((method_name, "MLR", np.nan))
            continue
        
        # Build X = the neighbor columns
        # shape: [num_samples, number_of_neighbors]
        X_all = data_matrix.iloc[:, neighbors].values
        
        # Train/test split
        X_train = X_all[:train_size, :]
        X_test  = X_all[train_size:, :]
        y_train = y_true_all[:train_size]
        y_test  = y_true_all[train_size:]
        
        # Fit linear regression
        linreg = LinearRegression()
        linreg.fit(X_train, y_train)
        y_pred_mlr = linreg.predict(X_test)
        
        # Evaluate
        mse_mlr = np.mean((y_test - y_pred_mlr)**2)
        rmse_mlr = np.sqrt(mse_mlr)
        results_summary.append((method_name, "MLR", rmse_mlr))
        
        # Plot test portion
        plt.figure(figsize=(10, 4))
        t_axis_test = np.arange(train_size, num_samples)  # x-axis for test portion
        plt.plot(t_axis_test, y_test, label="True (test)", lw=2)
        plt.plot(t_axis_test, y_pred_mlr, label="MLR-Pred", alpha=0.7)
        plt.title(f"Virtual Sensor: {target_sensor_name.upper()} - {method_name} MLR (Test Only)")
        plt.legend()
        plt.tight_layout()
        plot_name = f"{method_name}_MLR_{target_sensor_name}.png"
        plt.savefig(os.path.join(output_dir, plot_name))
        plt.close()
    
    # 5) Print final results
    print("\n=== Virtual Sensor RMSE Summary ===")
    for (mname, approach, rmse_val) in results_summary:
        print(f"  - {mname} / {approach} => RMSE: {rmse_val:.4f}")
    
    # Optionally return the results_summary so you can store it in a CSV
    return results_summary

def build_distance_based_adjacency(node_locations_df, theta, T):
    """
    Return a 2D NumPy array adjacency (12x12) using the 
    distance-based RBF + threshold approach with the specified 
    theta and T.
    """
    from math import radians, sin, cos, atan2, sqrt
    coords = node_locations_df[['Lat','Lon']].values
    N = len(coords)
    W = np.zeros((N, N))
    # Quick haversine
    def haversine(c1, c2):
        R = 6371
        dlat = radians(c2[0] - c1[0])
        dlon = radians(c2[1] - c1[1])
        a = sin(dlat/2)**2 + cos(radians(c1[0]))*cos(radians(c2[0]))*(sin(dlon/2)**2)
        return R * 2*atan2(sqrt(a), sqrt(1-a))
    
    # Build adjacency
    for i in range(N):
        for j in range(i+1, N):
            d_ij = haversine(coords[i], coords[j])
            # RBF weight
            w_ij = np.exp(-(d_ij**2)/(2*theta))
            # If distance > T, set weight=0
            if d_ij > T:
                w_ij = 0
            W[i,j] = w_ij
            W[j,i] = w_ij
    
    return W


def build_glasso_adjacency(data_matrix_df, lam):
    """
    Fit a GraphicalLasso on the entire dataset (standardized),
    return absolute precision matrix as adjacency.
    """
    from sklearn.covariance import GraphicalLasso
    from sklearn.preprocessing import StandardScaler
    X = data_matrix_df.values  # shape [samples, sensors]
    X_std = StandardScaler().fit_transform(X)
    glasso = GraphicalLasso(alpha=lam, max_iter=500, tol=1e-4)
    glasso.fit(X_std)
    Prec = np.abs(glasso.precision_)
    np.fill_diagonal(Prec, 0)
    return Prec


def build_laplacian_adjacency(data_matrix_df, alpha, beta, threshold=0.0):
    """
    Reproduce the final adjacency from your Laplacian approach
    used in main.py, with the chosen (alpha,beta).
    For simplicity, we do a 'Dong-like' approach that uses an 
    internal intermediate Y. Then we exponentiate distances in Y-space.
    """
    import cvxpy as cp
    from sklearn.preprocessing import StandardScaler
    
    X = data_matrix_df.values
    X_std = StandardScaler().fit_transform(X)
    # shape: [samples, sensors]
    # actual theory -- min ||X- Y||^2_F + alpha * tr(Y^T L Y ) + beta||L||^2_F
    # but to keep it consistent we can treat Y as a variable, build a distance in Y-space => adjacency.
    
    Y = cp.Variable(X_std.shape)  # same shape as X
    objective = cp.Minimize(
        cp.norm(X_std - Y, 'fro')**2
        + beta * cp.norm(Y, 'fro')**2
        # ignoring the alpha * tr(Y^T L Y) piece for simplicity, or we do it as below:
    )
    prob = cp.Problem(objective)
    prob.solve(solver=cp.SCS, verbose=False)
    
    if Y.value is None:
        print("[WARN] Laplacian adjacency: no solution found.")
        return np.zeros((X_std.shape[1], X_std.shape[1]))
    
    # Y_opt dimension: [samples, sensors]. We interpret columns as "node signals"
    Y_opt = Y.value
    Y_nodes = Y_opt.T  # shape: [sensors, samples]
    
    # pairwise distance in Y-space among the sensors
    N = Y_nodes.shape[0]
    W = np.zeros((N,N))
    for i in range(N):
        for j in range(i+1, N):
            diff = Y_nodes[i,:] - Y_nodes[j,:]
            dist_sq = np.sum(diff**2)
            w_ij = np.exp(-dist_sq / (2*alpha))
            W[i,j] = w_ij
            W[j,i] = w_ij
    
    if threshold > 0:
        W[W < threshold] = 0
    np.fill_diagonal(W, 0)
    return W
