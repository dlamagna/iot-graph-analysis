import pandas as pd
import models  # Importing from models.py
import matplotlib.pyplot as plt
import tqdm
import networkx as nx

# Expanded Parameter Configurations
t_values = [0.3, 0.5, 0.8, 1, 1.2, 1.5]
theta_values = [0.023, 0.057, 0.1136, 0.2273, 0.4546]
lambda_values = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2, 3, 4]
alpha_values = [0.01, 0.1, 0.5, 1, 2, 2.5, 3, 4, 5, 7.5, 10, 25, 50]
beta_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10]

# Run models
results, graphs = [], {}

print("\n📌 Running Distance-Based Graph Construction...")
r, g = models.distance_based_graph(models.node_location, theta_values, t_values)
results.extend(r)
graphs.update(g)

print("\n📌 Running GLASSO Graph Construction...")
r, g = models.graphical_lasso_graph(models.data_matrix_numeric, lambda_values)
results.extend(r)
graphs.update(g)

print("\n📌 Running Laplacian-Based Graph Construction...")
r, g = models.laplacian_graph(models.data_matrix_numeric, alpha_values, beta_values)
results.extend(r)
graphs.update(g)

all_results = pd.DataFrame(results)
all_results.to_csv("full_metrics_comparison_table.csv")

print("\n✅ All models executed, results concatenated into full_metrics_comparison_table.csv!")

# Save all graphs at once with progress bar
for filename, G in tqdm.tqdm(graphs.items(), desc="Saving Graphs"):
    if G.number_of_edges() > 5000:
        continue
    plt.figure(figsize=(8, 6))
    nx.draw(G, with_labels=True, node_color='skyblue', node_size=700, font_size=10)
    plt.savefig(filename)
    plt.close()
    # print("Saved graph to ", filename)

print("\n✅ Graphs saved successfully!")

