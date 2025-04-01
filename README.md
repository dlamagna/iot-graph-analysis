# iot-graph-analysis
This repository contains an implementation of data-driven techniques for constructing graph representations of sensor networks, with a focus on air pollution monitoring platforms. The project compares three approaches to graph topology inference:
  
1. **Distance-Based Graph Construction**  
2. **Graphical Lasso (GLASSO) Graph Construction**  
3. **Laplacian-Based (Smoothness-Based) Graph Construction**

These methods are used to build graphs from sensor data and evaluate them using key network metrics and signal reconstruction performance.

---

## Repository Structure

```
├── aggregated_plots
│   ├── ...              # output of aggregated_plots.py
|   .
├── aggregated_plots.py  # takes in experiment results and creates comparison plots
├── main.py              # Main script that runs experiments and aggregates result
├── models.py            # Contains functions for graph construction methods
├── plots                # output of main script, graph results
│   ├── ...
├── README.md            # this file
└── requirements.txt     # python dependencies
``` 
---

## Dependencies

The project requires the following Python packages:

- numpy
- pandas
- matplotlib
- networkx
- scikit-learn
- cvxpy
- tqdm

You can install all dependencies using pip:

```bash
python3 -m pip install -r requirements.txt
```

## Overview

### Distance-Based Graph Construction
- Uses the Haversine formula to calculate the pairwise distances between sensor nodes.
- Applies a Gaussian kernel to compute edge weights.
- Dynamically sets hyperparameters based on top–level distance statistics.

### Graphical Lasso
- Estimates a sparse precision matrix (inverse covariance matrix) from the sensor data.
- Constructs a graph by interpreting non-zero entries in the precision matrix as edges.
- Sensitive to multicollinearity; λ is tuned over a wide range.

### Laplacian-Based Graph Construction
- Learns a graph Laplacian that promotes signal smoothness.
- Uses an alternating minimization scheme to estimate a smoothed version of the data and a valid Laplacian.
- Hyperparameters α and β (or their ratio) control the trade-off between data fidelity, smoothness, and sparsity.

---

## How to Run

1. **Clone the Repository**

   ```bash
   git clone git@github.com:dlamagna/iot-graph-analysis.git
   cd iot-graph-analysis
   ```

2. **Install dependencies**

    ```bash
    python3 -m pip install -r requirements.txt
    ```

3. **Runt he main script**

    ```bash
    python3 main.py
    ```

    The script will:
    *Load the sensor and location data.
    *Run all three graph construction methods over a grid of hyperparameters.
    *Save the computed metrics to `full_metrics_comparison_table.csv.`
    *Save the generated graph plots to the `plots/` directory.


## Hyperparameter Configuration

The following are example hyperparameter grids used in the experiments (set in `main.py`):

- **Distance-Based Graphs:**
  - `theta_values = [0.023, 0.057, 0.1136, 0.2273, 0.4546]`
  - `t_values = [0.3, 0.5, 0.8, 1, 1.2, 1.5]`

- **Graphical Lasso:**
  - `lambda_values = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2, 3, 4]`

- **Laplacian-Based Graphs:**
  - `alpha_values = [0.01, 0.1, 0.5, 1, 2, 2.5, 3, 4, 5, 7.5, 10, 25, 50]`
  - `beta_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10]`

These ranges can be adjusted to explore a broader or more targeted set of parameter values. In our implementation, the distance-based method dynamically computes basic statistics (min, max, mean, and standard deviation) of the pairwise distances between nodes to potentially guide the selection of `theta_values`.

---

## Output

- **Metrics:**  
  A CSV file named `full_metrics_comparison_table.csv` is generated. This file contains key graph metrics (such as edge count, edge density, clustering coefficient, average degree, etc.) for each method and for every combination of hyperparameters tested.

- **Graph Visualizations:**  
  Graph plots are saved as PNG files in the `plots/` directory. These images provide a visual representation of the network topology constructed using each method.

- **Optional Visualizations:**  
  Additional visualizations (such as heatmaps showing edge counts vs. hyperparameters) can be generated to further analyze the effects of varying hyperparameters on the learned graph topology.

---

## Future Work

Potential extensions and improvements include:
- **Dynamic Hyperparameter Tuning:**  
  Refine the selection of hyperparameters by using more advanced methods (e.g., cross-validation) to minimize signal reconstruction error (e.g., RMSE).

- **Enhanced Visualization:**  
  Develop additional plots such as heatmaps, scatter plots, and clustering dendrograms to visualize the relationship between hyperparameters and graph metrics.

- **Robust Signal Reconstruction:**  
  Extend the framework to better handle missing data and improve robustness in signal reconstruction.
  - completed within virtual.py

- **Integration with Other Models:**  
  Explore incorporating other statistical priors or signal processing techniques to further enhance the graph learning process.

---

## Acknowledgments

This project builds on ideas and methodologies from:

- *Graph Learning Techniques Using Structured Data for IoT Air Pollution Monitoring Platforms* by Ferrer-Cid et al.
- *Learning Laplacian Matrix in Smooth Graph Signal Representations* by Dong et al.