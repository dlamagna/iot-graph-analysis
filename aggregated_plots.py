import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load results CSV
df = pd.read_csv("full_metrics_comparison_table.csv")
print("Head of results:")
print(df.head())
print("\nDataFrame shape:", df.shape)

# Make sure parameter columns are numeric
numeric_cols = ["Theta", "T", "Lambda", "Alpha", "Beta", "Edges", "Edge Density", "Run Time (s)"]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# -----------------------------
# 1. Distance-Based Graphs
# -----------------------------
df_distance = df[df["Method"] == "Distance-Based"]
print("\nUnique Theta values (Distance-Based):", df_distance["Theta"].unique())

# Pivot table for Edge Count
pivot_distance_edges = df_distance.pivot(index="Theta", columns="T", values="Edges")
plt.figure(figsize=(8,6))
sns.heatmap(pivot_distance_edges, annot=True, fmt="g", cmap="viridis")
plt.title("Distance-Based Graph: Edge Count Heatmap")
plt.ylabel("Theta")
plt.xlabel("T")
plt.tight_layout()
plt.savefig("aggregated_plots/distance_based_edge_count_heatmap.png")
plt.show()

# Pivot table for Edge Density
pivot_distance_density = df_distance.pivot(index="Theta", columns="T", values="Edge Density")
plt.figure(figsize=(8,6))
sns.heatmap(pivot_distance_density, annot=True, fmt=".2f", cmap="magma")
plt.title("Distance-Based Graph: Edge Density Heatmap")
plt.ylabel("Theta")
plt.xlabel("T")
plt.tight_layout()
plt.savefig("aggregated_plots/distance_based_edge_density_heatmap.png")
plt.show()

# -----------------------------
# 2. GLASSO Graphs
# -----------------------------
df_glasso = df[df["Method"] == "GLASSO"]

plt.figure(figsize=(8,6))
sns.lineplot(data=df_glasso, x="Lambda", y="Edges", marker="o")
plt.title("GLASSO Graph: Edges vs Lambda")
plt.xlabel("Lambda")
plt.ylabel("Edges")
plt.tight_layout()
plt.savefig("aggregated_plots/glasso_edges_vs_lambda.png")
plt.show()

plt.figure(figsize=(8,6))
sns.lineplot(data=df_glasso, x="Lambda", y="Edge Density", marker="o")
plt.title("GLASSO Graph: Edge Density vs Lambda")
plt.xlabel("Lambda")
plt.ylabel("Edge Density")
plt.tight_layout()
plt.savefig("aggregated_plots/glasso_edge_density_vs_lambda.png")
plt.show()

plt.figure(figsize=(8,6))
sns.lineplot(data=df_glasso, x="Lambda", y="Run Time (s)", marker="o")
plt.title("GLASSO Graph: Run Time vs Lambda")
plt.xlabel("Lambda")
plt.ylabel("Run Time (s)")
plt.tight_layout()
plt.savefig("aggregated_plots/glasso_runtime_vs_lambda.png")
plt.show()

# -----------------------------
# 3. Laplacian-Based Graphs
# -----------------------------
df_laplacian = df[df["Method"] == "Laplacian-Based"]

# Pivot table for Edge Count (rows: Alpha, columns: Beta)
pivot_laplacian_edges = df_laplacian.pivot(index="Alpha", columns="Beta", values="Edges")
plt.figure(figsize=(8,6))
sns.heatmap(pivot_laplacian_edges, annot=True, fmt="g", cmap="coolwarm")
plt.title("Laplacian-Based Graph: Edge Count Heatmap")
plt.ylabel("Alpha")
plt.xlabel("Beta")
plt.tight_layout()
plt.savefig("aggregated_plots/laplacian_edge_count_heatmap.png")
plt.show()

# Pivot table for Edge Density
pivot_laplacian_density = df_laplacian.pivot(index="Alpha", columns="Beta", values="Edge Density")
plt.figure(figsize=(8,6))
sns.heatmap(pivot_laplacian_density, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Laplacian-Based Graph: Edge Density Heatmap")
plt.ylabel("Alpha")
plt.xlabel("Beta")
plt.tight_layout()
plt.savefig("aggregated_plots/laplacian_edge_density_heatmap.png")
plt.show()

# Pivot table for Run Time
pivot_laplacian_runtime = df_laplacian.pivot(index="Alpha", columns="Beta", values="Run Time (s)")
plt.figure(figsize=(8,6))
sns.heatmap(pivot_laplacian_runtime, annot=True, fmt=".3f", cmap="YlOrRd")
plt.title("Laplacian-Based Graph: Run Time Heatmap")
plt.ylabel("Alpha")
plt.xlabel("Beta")
plt.tight_layout()
plt.savefig("aggregated_plots/laplacian_runtime_heatmap.png")
plt.show()
