# Required Libraries
import pandas as pd
from sklearn.cluster import DBSCAN
import folium

# Step 1: Load the Data from CSV
# Assuming your file is 'modified_synthetic_dataset.csv' in the working directory
df = pd.read_csv('RTA Final/filtered_dataset_7km.csv')

# Step 2: Data Preprocessing
# If there's an 'Hour' column already, we don't need to modify it
# Just ensure it's in string format for filtering
df['Hour'] = pd.to_datetime(df['Hour'],format='%H').dt.strftime('%H')

# Step 3: Function to filter data based on a given hour and perform clustering
def cluster_by_hour(hour_input):
    # Step 3.1: Filter the DataFrame by the provided hour (exact match)
    filtered_df = df[df['Hour'] == hour_input]

    # Check if there's data for that specific hour
    if filtered_df.empty:
        print(f"No data available for the hour {hour_input}")
        return

    # Step 3.2: Clustering the Latitude and Longitude for the filtered data
    coords = filtered_df[['StartLat', 'StartLon']].values

    # Adjust DBSCAN parameters to get more clusters
    dbscan = DBSCAN(eps=0.0005, min_samples=2, metric='euclidean').fit(coords)

    # Add cluster labels to the filtered DataFrame
    filtered_df['Cluster'] = dbscan.labels_

    # Step 3.3: Visualize Clusters on a Map for the specific hour
    map_center = [filtered_df['StartLat'].mean(), filtered_df['StartLon'].mean()]
    m = folium.Map(location=map_center, zoom_start=12)

    # Loop through each unique cluster and add markers to the map
    for cluster_label in filtered_df['Cluster'].unique():
        if cluster_label == -1:
            # Skip noise points labeled as -1
            continue

        # Filter data for the current cluster
        cluster_data = filtered_df[filtered_df['Cluster'] == cluster_label]

        # Get the average location (center of the cluster)
        cluster_center_lat = cluster_data['StartLat'].mean()
        cluster_center_lon = cluster_data['StartLon'].mean()

        # Count the number of points (people) in this cluster
        num_people = len(cluster_data)

        # Add a marker to the map for the cluster
        folium.Marker(
            location=[cluster_center_lat, cluster_center_lon],
            popup=f"Cluster {cluster_label}: {num_people} people",
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(m)

    # Step 4: Save the Map to an HTML file with the hour in the name
    map_file_name = f'clusters_map_hour_{hour_input}.html'
    m.save(map_file_name)
    print(f"Map saved as {map_file_name}")

# Example: To cluster for a specific hour, such as '08'
# Input hour in the format 'H' (24-hour format)
input_hour = "08"  # Replace with the desired hour for clustering
cluster_by_hour(input_hour)

