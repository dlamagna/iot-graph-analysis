import pandas as pd
from models import (
    calculate_distance_matrix,
    node_location,
    distance_based_graph,
    graphical_lasso_with_and_without_pca,
    laplacian_graph,
    data_matrix_numeric,
    run_laplacian_interpolation_demo,
)
import matplotlib.pyplot as plt
import tqdm
import networkx as nx
import numpy as np
from visualization import plot_and_save_graphs, generate_aggregated_plots
import os

# Broadened Parameter Configurations
t_values = [1, 2, 4, 5, 7,5, 8.75, 10, 15, 25, 40, 50, 65, 80]
theta_values = [1, 5, 10, 25.53, 40, 50]
lambda_values = [1e-5, 5e-5, 0.0001, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.7, 0.75, 0.8, 1, 1.5, 2]
alpha_values = [10]
beta_values = [1,1.5,2,2.5,3,3.5,4,4.5,5]

results, graphs = [], {}

# Calculate distance matrix
distances = calculate_distance_matrix(node_location)
# Calculate theta as the standard deviation of the nonzero pairwise distances
theta_std = np.std(distances[np.triu_indices_from(distances, k=1)])
print(f"Standard deviation (θ) of pairwise distances: {theta_std:.6f} km")

# Run the graph construction models
print("\n +++ Running Distance-Based Graph Construction...")
r, g = distance_based_graph(node_location, theta_values, t_values)
results.extend(r)
graphs.update(g)

print("\n +++ Running GLASSO Graph Construction...")
r, g = graphical_lasso_with_and_without_pca(data_matrix_numeric, lambda_values)
results.extend(r)
graphs.update(g)

print("\n +++ Running Laplacian-Based Graph Construction...")
r, g = laplacian_graph(data_matrix_numeric, alpha_values, beta_values)
results.extend(r)
graphs.update(g)

# Save the metrics results
all_results = pd.DataFrame(results)
all_results.to_csv("full_metrics_comparison_table.csv")
print("\n-- All models executed, results concatenated into full_metrics_comparison_table.csv!")

node_locations_df = pd.read_csv("Node-Location.csv", delimiter=';', skipinitialspace=True)
print(f"-- Loaded {len(node_locations_df)} nodes.")
print(f"Found {len(graphs)} graphs to process.")

# 1. Mapping from dataframe index (0,1,2,...) to location Name
node_label_map_index = node_locations_df['Name'].to_dict()
# 2. Mapping from EOI (as string) to Name
node_label_map_eoi = dict(zip(node_locations_df['EOI'].astype(str), node_locations_df['Name']))

output_directory = "laplacian_graph_plots"  # for example
os.makedirs(output_directory, exist_ok=True)

# Load sensor data
sensor_data = pd.read_csv("data_matrix_12-stations.csv", delimiter=';', skipinitialspace=True)
station_columns = sensor_data.columns[1:]
sensor_means = sensor_data[station_columns].mean(axis=0)
sensor_values = {station.strip().lower(): mean for station, mean in sensor_means.items()}

# print(sensor_values)
# assert False
plot_and_save_graphs(
    graphs=graphs,
    node_locations_df=node_locations_df,
    output_dir=output_directory,
    label_offset=0.03,
    use_o3_colormap=True,  # Using YlOrRd, appropriate for O₃ data.
    show_progress=True,
    sensor_values=sensor_values  # Use actual sensor means for coloring.
)

print("\n +++ Running Laplacian Interpolation Demo...")
reconstructed = run_laplacian_interpolation_demo(graphs, node_locations_df, sensor_values)

generate_aggregated_plots(csv_file="full_metrics_comparison_table.csv",
                          output_dir="aggregated_plots",
                          show_plots=False)