import os
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib as mpl


def plot_and_save_graphs(
    graphs,
    node_locations_df,
    output_dir=".",
    label_offset=0.03,
    use_o3_colormap=False,
    show_progress=False,
    sensor_values=None
):
    # show_progress=False
    """
    Plots and saves each NetworkX graph with formatted labels. Node colors are determined by either
    actual sensor measurements (if sensor_values is provided) or by node degree otherwise.
    
    Debug prints are added to help identify why node lookups may be failing.
    """
    # Optionally use a progress bar.
    if show_progress:
        from tqdm import tqdm
        graph_items = tqdm(graphs.items(), desc="Plotting Graphs")
    else:
        graph_items = graphs.items()
    
    # Build node label mappings:
    # 1. Mapping from the DataFrame index to the 'Name' column.
    node_label_map_index = node_locations_df['Name'].to_dict()
    # 2. Mapping from the 'EOI' column (converted to string) to 'Name'
    node_label_map_eoi = dict(zip(node_locations_df['EOI'].astype(str), node_locations_df['Name']))
    
    # Choose colormap: use YlOrRd if mimicking O₃ sensor data, else Set2.
    cmap = plt.cm.YlOrRd if use_o3_colormap else plt.cm.Set2
    
    # DEBUG: Print out some info about sensor_values keys
    if sensor_values is not None:
        all_sensor_keys = list(sensor_values.keys())
        # print(f"\n[DEBUG] sensor_values has {len(all_sensor_keys)} keys.")
        # Print the first 10 keys for inspection
        # print("[DEBUG] First few sensor_values keys:", all_sensor_keys[:10])
    
    for filename, G in graph_items:
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G)
    
        # Build a label dictionary for each node.
        labels = {}
        for node in G.nodes():
            if node in node_label_map_index:
                labels[node] = node_label_map_index[node]
            elif str(node) in node_label_map_eoi:
                labels[node] = node_label_map_eoi[str(node)]
            else:
                labels[node] = str(node)
    
        # Determine node values for coloring.
        if sensor_values is not None:
            # Collect sensor values for nodes present in this graph.
            local_values = []
            
            # print(f"\n[DEBUG] Graph '{filename}' has {G.number_of_nodes()} nodes.")
            
            for node in G.nodes():
                # Attempt multiple ways of looking up sensor data
                # 1) raw node
                # 2) str(node)
                # 3) label if it might match a known key
                raw_key = node
                str_key = str(node)
                label_key = labels[node]
                
                sensor_val = None
                if raw_key in sensor_values:
                    sensor_val = sensor_values[raw_key]
                elif str_key in sensor_values:
                    sensor_val = sensor_values[str_key]
                elif label_key in sensor_values:
                    sensor_val = sensor_values[label_key]
                
                if sensor_val is not None:
                    local_values.append(sensor_val)
                    # print(f"  Node {node} -> sensor value found: {sensor_val} (using key: {raw_key if raw_key in sensor_values else str_key if str_key in sensor_values else label_key})")
                else:
                    print(f"  Node {node} (label: {label_key}): No sensor data found under any key: {raw_key}, {str_key}, {label_key}")
            
            # If we found at least one valid sensor reading among the nodes:
            if local_values:
                vmin = min(local_values)
                vmax = max(local_values)
            else:
                vmin, vmax = 0, 1  # fallback if none of the nodes have sensor data
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    
            # Build node_colors, again trying all possible keys
            node_colors = []
            for node in G.nodes():
                raw_key = node
                str_key = str(node)
                label_key = labels[node]
                
                if raw_key in sensor_values:
                    value = sensor_values[raw_key]
                elif str_key in sensor_values:
                    value = sensor_values[str_key]
                elif label_key in sensor_values:
                    value = sensor_values[label_key]
                else:
                    value = vmin  # fallback
                
                node_colors.append(cmap(norm(value)))
    
            colorbar_label = "Mean O₃ Reading"
    
        else:
            # Fallback: use node degree.
            node_values = dict(G.degree())
            vmin = 0
            vmax = max(node_values.values()) if node_values else 1
            if vmax == 0:
                vmax = 1
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            node_colors = [cmap(norm(node_values[node])) for node in G.nodes()]
            colorbar_label = "Node Degree"
    
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700)
        nx.draw_networkx_edges(G, pos)
    
        # Offset labels so they appear just above the nodes.
        pos_labels = {n: (x, y + label_offset) for n, (x, y) in pos.items()}
        nx.draw_networkx_labels(G, pos_labels, labels=labels, font_size=10, font_color='black')
    
        # Create a ScalarMappable for the colorbar.
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        ax = plt.gca()
        cbar = plt.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label(colorbar_label, rotation=270, labelpad=15)
    
        # Create a title from the base filename (removing directories and extension).
        base_name = os.path.splitext(os.path.basename(filename))[0]
        plt.title(base_name)
        plt.axis('off')
    
        out_path = os.path.join(output_dir, base_name + ".png")
        plt.savefig(out_path)
        plt.close()


def plot_and_save_graphs_old(
    graphs,
    node_locations_df,
    output_dir=".",
    label_offset=0.03,
    use_o3_colormap=False,
    show_progress=False,
    sensor_values=None
):
    """
    Plots and saves each NetworkX graph with formatted labels. Node colors are determined by either
    actual sensor measurements (if sensor_values is provided) or by node degree otherwise.
    
    When sensor measurements are provided, the color scale is computed relative to the nodes present
    in each individual graph. This way, the legend (colorbar) distinguishes the differences in sensor
    values between nodes on that graph.
    
    Additionally, if sensor_values is provided, prints out the mean sensor value for each node.
    
    Parameters
    ----------
    graphs : dict
        Dictionary of {filename: nx.Graph} to be plotted and saved.
    node_locations_df : pd.DataFrame
        DataFrame containing columns 'Name' and 'EOI' for mapping node identifiers to location names.
    output_dir : str, optional
        Directory where the plots should be saved. Defaults to current directory.
    label_offset : float, optional
        Vertical offset for node labels so they don't overlap the nodes.
    use_o3_colormap : bool, optional
        If True, uses the YlOrRd colormap (often used for O₃ data); otherwise uses the Set2 colormap.
    show_progress : bool, optional
        If True, shows a tqdm progress bar while processing.
    sensor_values : dict, optional
        A mapping {node: mean_sensor_value} used to color the nodes. If provided, these raw values
        will determine the color scale relative to the nodes in each graph. If None, the node degree is used.
    
    Returns
    -------
    None
    """
    # Optionally use a progress bar.
    if show_progress:
        from tqdm import tqdm
        graph_items = tqdm(graphs.items(), desc="Plotting Graphs")
    else:
        graph_items = graphs.items()
    
    # Build node label mappings:
    # 1. Mapping from the DataFrame index to the 'Name' column.
    node_label_map_index = node_locations_df['Name'].to_dict()
    # 2. Mapping from the 'EOI' column (converted to string) to 'Name'
    node_label_map_eoi = dict(zip(node_locations_df['EOI'].astype(str), node_locations_df['Name']))
    
    # Choose colormap: use YlOrRd if mimicking O₃ sensor data, else Set2.
    cmap = plt.cm.YlOrRd if use_o3_colormap else plt.cm.Set2
    
    for filename, G in graph_items:
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G)
    
        # Build a label dictionary for each node.
        labels = {}
        for node in G.nodes():
            if node in node_label_map_index:
                labels[node] = node_label_map_index[node]
            elif str(node) in node_label_map_eoi:
                labels[node] = node_label_map_eoi[str(node)]
            else:
                labels[node] = str(node)
    
        # Determine node values for coloring.
        if sensor_values is not None:
            # Collect sensor values for nodes present in this graph.
            local_values = [sensor_values[node] for node in G.nodes() if node in sensor_values]
            if local_values:
                vmin = min(local_values)
                vmax = max(local_values)
            else:
                vmin, vmax = 0, 1  # Fallback if none of the nodes have sensor data.
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            node_colors = [cmap(norm(sensor_values.get(node, vmin))) for node in G.nodes()]
            colorbar_label = "Mean O₃ Reading"
            
            # Print the sensor mean for each node for debugging purposes.
            print(f"Graph '{filename}':")
            for node in G.nodes():
                value = sensor_values.get(node, None)
                label_text = labels.get(node, str(node))
                if value is not None:
                    print(f"  Node {node} (label: {label_text}): {value}")
                else:
                    print(f"  Node {node} (label: {label_text}): No sensor data")
        else:
            # Fallback: use node degree.
            node_values = dict(G.degree())
            vmin = 0
            vmax = max(node_values.values()) if node_values else 1
            if vmax == 0:
                vmax = 1
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            node_colors = [cmap(norm(node_values[node])) for node in G.nodes()]
            colorbar_label = "Node Degree"
    
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700)
        nx.draw_networkx_edges(G, pos)
    
        # Offset labels so they appear just above the nodes.
        pos_labels = {n: (x, y + label_offset) for n, (x, y) in pos.items()}
        nx.draw_networkx_labels(G, pos_labels, labels=labels, font_size=10, font_color='black')
    
        # Create a ScalarMappable for the colorbar.
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        ax = plt.gca()
        cbar = plt.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label(colorbar_label, rotation=270, labelpad=15)
    
        base_name = os.path.splitext(os.path.basename(filename))[0]
        plt.title(base_name)
        plt.axis('off')
    
        out_path = os.path.join(output_dir, base_name + ".png")
        plt.savefig(out_path)
        plt.close()

def generate_aggregated_plots(csv_file="full_metrics_comparison_table.csv",
                              output_dir="aggregated_plots",
                              show_plots=False):
    """
    Reads the results CSV and generates a set of aggregated plots for 
    Distance-Based, GLASSO, and Laplacian-Based graph methods.
    
    The generated plots include:
      - Distance-Based Graphs:
          • A heatmap showing the average Edge Count (Edges) vs T and Theta.
          • An overlay line plot showing Edges vs T for each Theta.
      - GLASSO Graphs:
          • A line plot of Edges vs Lambda.
          • A line plot of Run Time vs Lambda.
      - Laplacian-Based Graphs:
          • A heatmap showing Edge Count vs Alpha and Beta.
          • A heatmap showing Run Time vs Alpha and Beta.
    
    Parameters
    ----------
    csv_file : str, optional
        Path to the results CSV file. Defaults to "full_metrics_comparison_table.csv".
    output_dir : str, optional
        Directory to save the generated plots. Defaults to "aggregated_plots".
    show_plots : bool, optional
        If True, displays the plots interactively via plt.show(). Defaults to False.
    
    Returns
    -------
    None
    """
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results CSV
    df = pd.read_csv(csv_file)
    print("Head of results:")
    print(df.head())
    print("\nDataFrame shape:", df.shape)
    
    # Ensure parameter columns are numeric
    numeric_cols = ["Theta", "T", "Lambda", "Alpha", "Beta", "Edges", "Run Time (s)"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 1. Distance-Based Graphs
    df_distance = df[df["Method"] == "Distance-Based"]
    print("\nUnique Theta values (Distance-Based):", df_distance["Theta"].unique())
    
    # Heatmap for Edge Count (using 'Edges')
    pivot_distance_edges = df_distance.pivot_table(index="Theta", columns="T", values="Edges", aggfunc='mean')
    plt.figure(figsize=(8,6))
    sns.heatmap(pivot_distance_edges, annot=True, fmt="g", cmap="viridis")
    plt.title("Distance-Based Graph: Edge Count Heatmap")
    plt.ylabel("Theta")
    plt.xlabel("T")
    plt.tight_layout()
    filename = os.path.join(output_dir, "distance_based_edge_count_heatmap.png")
    plt.savefig(filename)
    if show_plots:
        plt.show()
    plt.close()
    
    # Overlaying line plot: Edges vs T for each Theta
    plt.figure(figsize=(8,6))
    for theta in sorted(df_distance["Theta"].unique()):
        subset = df_distance[df_distance["Theta"] == theta].sort_values("T")
        plt.plot(subset["T"], subset["Edges"], marker="o", label=f"Theta={theta}")
    plt.title("Distance-Based Graph: Edges vs T (Overlay for each Theta)")
    plt.xlabel("T")
    plt.ylabel("Edges")
    plt.legend(title="Theta")
    plt.tight_layout()
    filename = os.path.join(output_dir, "distance_based_edges_vs_T_overlay.png")
    plt.savefig(filename)
    if show_plots:
        plt.show()
    plt.close()
    
    # 2. GLASSO Graphs
    df_glasso = df[df["Method"] == "GLASSO (Original)"]
    print("\nGLASSO DataFrame shape:", df_glasso.shape)
    if df_glasso.empty:
        print("Warning: No GLASSO data found. Check the 'Method' column values. Available methods:", df["Method"].unique())
    else:
        print("Unique Lambda values in GLASSO DataFrame:", df_glasso["Lambda"].unique())
        print("Unique Edges values in GLASSO DataFrame:", df_glasso["Edges"].unique())
    
    # Line plot: Edges vs Lambda
    plt.figure(figsize=(8,6))
    sns.lineplot(data=df_glasso, x="Lambda", y="Edges", marker="o")
    plt.title("GLASSO Graph: Edges vs Lambda")
    plt.xlabel("Lambda")
    plt.ylabel("Edges")
    plt.tight_layout()
    filename = os.path.join(output_dir, "glasso_edges_vs_lambda.png")
    plt.savefig(filename)
    if show_plots:
        plt.show()
    plt.close()
    
    # Line plot: Run Time vs Lambda
    plt.figure(figsize=(8,6))
    sns.lineplot(data=df_glasso, x="Lambda", y="Run Time (s)", marker="o")
    plt.title("GLASSO Graph: Run Time vs Lambda")
    plt.xlabel("Lambda")
    plt.ylabel("Run Time (s)")
    plt.tight_layout()
    filename = os.path.join(output_dir, "glasso_runtime_vs_lambda.png")
    plt.savefig(filename)
    if show_plots:
        plt.show()
    plt.close()
    
    # 3. Laplacian-Based Graphs
    df_laplacian = df[df["Method"] == "Laplacian-Based"]
    
    # Heatmap for Edge Count (using 'Edges')
    pivot_laplacian_edges = df_laplacian.pivot(index="Alpha", columns="Beta", values="Edges")
    plt.figure(figsize=(8,6))
    sns.heatmap(pivot_laplacian_edges, annot=True, fmt="g", cmap="coolwarm")
    plt.title("Laplacian-Based Graph: Edge Count Heatmap")
    plt.ylabel("Alpha")
    plt.xlabel("Beta")
    plt.tight_layout()
    filename = os.path.join(output_dir, "laplacian_edge_count_heatmap.png")
    plt.savefig(filename)
    if show_plots:
        plt.show()
    plt.close()
    
    # Heatmap for Run Time
    pivot_laplacian_runtime = df_laplacian.pivot(index="Alpha", columns="Beta", values="Run Time (s)")
    plt.figure(figsize=(8,6))
    sns.heatmap(pivot_laplacian_runtime, annot=True, fmt=".3f", cmap="YlOrRd")
    plt.title("Laplacian-Based Graph: Run Time Heatmap")
    plt.ylabel("Alpha")
    plt.xlabel("Beta")
    plt.tight_layout()
    filename = os.path.join(output_dir, "laplacian_runtime_heatmap.png")
    plt.savefig(filename)
    if show_plots:
        plt.show()
    plt.close()
    
    print("Aggregated plots generated and saved in", output_dir)
