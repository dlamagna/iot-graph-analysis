import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, atan2
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point
from PIL import Image

# Define directory for saving plots and create it if it doesn't exist
plot_dir = "plots/data-desc/"
os.makedirs(plot_dir, exist_ok=True)

# 1. Load and Explore Node-Location Data
node_loc = pd.read_csv('Node-Location.csv', delimiter=';', skipinitialspace=True)
print("Node-Location Data:")
print(node_loc.head())
print("Number of nodes:", len(node_loc))
print("Latitude range: {:.4f} - {:.4f}".format(node_loc['Lat'].min(), node_loc['Lat'].max()))
print("Longitude range: {:.4f} - {:.4f}".format(node_loc['Lon'].min(), node_loc['Lon'].max()))


# 2. Plot Node Locations on a Barcelona Basemap (rotated 90° clockwise)
# Create a GeoDataFrame with geometry from (Lon, Lat)
geometry = [Point(xy) for xy in zip(node_loc['Lon'], node_loc['Lat'])]
gdf = gpd.GeoDataFrame(node_loc, geometry=geometry, crs="EPSG:4326")
# Reproject to Web Mercator (EPSG:3857) for basemap tiles
gdf = gdf.to_crs(epsg=3857)

fig, ax = plt.subplots(figsize=(10, 10))
gdf.plot(ax=ax, color='red', markersize=50, label='Sensor Nodes')
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
ax.set_title("Node Locations in Barcelona")
ax.set_axis_off()
plt.legend()
basemap_path = os.path.join(plot_dir, "node_locations_barcelona.png")
plt.savefig(basemap_path, dpi=300, bbox_inches='tight')
plt.close()

# Rotate the saved map 90° clockwise using PIL and save it
im = Image.open(basemap_path)
im_rotated = im.rotate(-90, expand=True)  # negative angle rotates clockwise
rotated_path = os.path.join(plot_dir, "node_locations_barcelona_rotated.png")
im_rotated.save(rotated_path)
print("Saved rotated basemap image to:", rotated_path)

# 3. Compute Pairwise Distances using the Haversine Formula
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in kilometers
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))

n = len(node_loc)
dist_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i != j:
            dist_matrix[i, j] = haversine(node_loc.loc[i, 'Lat'], node_loc.loc[i, 'Lon'],
                                           node_loc.loc[j, 'Lat'], node_loc.loc[j, 'Lon'])
            
plt.figure()
plt.imshow(dist_matrix, cmap='viridis')
plt.colorbar(label='Distance (km)')
plt.title('Pairwise Distance Matrix')
plt.savefig(os.path.join(plot_dir, 'distance_matrix.png'))
plt.close()

# 4. Load and Explore Sensor Data Matrix
sensor_data = pd.read_csv('data_matrix_12-stations.csv', delimiter=';', skipinitialspace=True)
print("Sensor Data Matrix (first few rows):")
print(sensor_data.head())

# Convert the first column (timestamp) to datetime
timestamps = pd.to_datetime(sensor_data.iloc[:, 0])
# Convert sensor readings to numeric values
data_numeric = sensor_data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
print("Sensor data shape:", data_numeric.shape)

# Compute descriptive statistics
desc_stats = data_numeric.describe()
print("Descriptive Statistics of Sensor Data:")
print(desc_stats)
overall_mean = data_numeric.mean().mean()
overall_std = data_numeric.std().mean()
overall_min = data_numeric.min().min()
overall_max = data_numeric.max().max()
print(f"Overall sensor stats - Mean: {overall_mean:.2f}, Std: {overall_std:.2f}, Min: {overall_min:.2f}, Max: {overall_max:.2f}")

# 5. Plot Sensor Data Time Series
plt.figure()
for col in data_numeric.columns:
    plt.plot(timestamps, data_numeric[col], label=col)
plt.xlabel('Time')
plt.ylabel('Sensor Reading')
plt.title('Sensor Data Time Series')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'sensor_timeseries.png'))
plt.close()


# 6. Plot Histograms of Sensor Readings
plt.figure()
for col in data_numeric.columns:
    plt.hist(data_numeric[col].dropna(), bins=30, alpha=0.5, label=col)
plt.xlabel('Sensor Reading')
plt.ylabel('Frequency')
plt.title('Histogram of Sensor Readings')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'sensor_histograms.png'))
plt.close()


# 7. Plot and Print Correlation Matrix of Sensor Readings
corr_matrix = data_numeric.corr()
plt.figure()
plt.imshow(corr_matrix, cmap='hot', interpolation='nearest')
plt.colorbar(label='Correlation Coefficient')
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45)
plt.yticks(range(len(corr_matrix.index)), corr_matrix.index)
plt.title('Correlation Matrix of Sensor Readings')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'sensor_correlation.png'))
plt.close()

print("Correlation Matrix of Sensor Readings:")
print(corr_matrix)

# Identify the sensor pairs with highest and lowest correlations (excluding self-correlations)
corr_no_diag = corr_matrix.copy()
np.fill_diagonal(corr_no_diag.values, np.nan)
max_corr = corr_no_diag.max().max()
min_corr = corr_no_diag.min().min()
max_idx = np.unravel_index(np.nanargmax(corr_no_diag.values), corr_no_diag.shape)
min_idx = np.unravel_index(np.nanargmin(corr_no_diag.values), corr_no_diag.shape)

sensor_max1 = corr_matrix.columns[max_idx[0]]
sensor_max2 = corr_matrix.columns[max_idx[1]]
sensor_min1 = corr_matrix.columns[min_idx[0]]
sensor_min2 = corr_matrix.columns[min_idx[1]]

print(f"Highest correlation: {max_corr:.4f} between sensors {sensor_max1} and {sensor_max2}")
print(f"Lowest correlation: {min_corr:.4f} between sensors {sensor_min1} and {sensor_min2}")

# Overlay Histograms and Time Series for Selected Sensor Pairs


# Overlay histograms for the pair with highest correlation
plt.figure()
plt.hist(data_numeric[sensor_max1].dropna(), bins=30, alpha=0.5, label=sensor_max1)
plt.hist(data_numeric[sensor_max2].dropna(), bins=30, alpha=0.5, label=sensor_max2)
plt.xlabel('Sensor Reading')
plt.ylabel('Frequency')
plt.title(f'Overlay Histograms: {sensor_max1} vs {sensor_max2}')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'overlay_hist_highest.png'))
plt.close()

# Overlay time series for the pair with highest correlation
plt.figure()
plt.plot(timestamps, data_numeric[sensor_max1], label=sensor_max1)
plt.plot(timestamps, data_numeric[sensor_max2], label=sensor_max2)
plt.xlabel('Time')
plt.ylabel('Sensor Reading')
plt.title(f'Overlay Time Series: {sensor_max1} vs {sensor_max2}')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'overlay_timeseries_highest.png'))
plt.close()

# Overlay histograms for the pair with lowest correlation
plt.figure()
plt.hist(data_numeric[sensor_min1].dropna(), bins=30, alpha=0.5, label=sensor_min1)
plt.hist(data_numeric[sensor_min2].dropna(), bins=30, alpha=0.5, label=sensor_min2)
plt.xlabel('Sensor Reading')
plt.ylabel('Frequency')
plt.title(f'Overlay Histograms: {sensor_min1} vs {sensor_min2}')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'overlay_hist_lowest.png'))
plt.close()

# Overlay time series for the pair with lowest correlation
plt.figure()
plt.plot(timestamps, data_numeric[sensor_min1], label=sensor_min1)
plt.plot(timestamps, data_numeric[sensor_min2], label=sensor_min2)
plt.xlabel('Time')
plt.ylabel('Sensor Reading')
plt.title(f'Overlay Time Series: {sensor_min1} vs {sensor_min2}')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'overlay_timeseries_lowest.png'))
plt.close()
