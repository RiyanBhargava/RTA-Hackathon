from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import r2_score, accuracy_score
import pickle
import folium

# Step 1: Load the Data from CSV
df = pd.read_csv("RTA Final/filtered_dataset_7km.csv")

# Preprocess Data
df['Hour'] = pd.to_datetime(df['Hour'], format='%H').dt.strftime('%H')

# Load the trained model
with open('trained_model.pkl', 'rb') as model_file:
    trained_model = pickle.load(model_file)

# Initialize Flask app
app = Flask(__name__)

# Helper function to calculate accuracy within a threshold
def within_threshold(y_true, y_pred, threshold=1):
    return np.mean(np.abs(y_true - y_pred) <= threshold)

# Function to predict the number of people based on latitude, longitude, and hour
def predict_num_people(lat, lon, hour):
    input_data = pd.DataFrame({
        'StartLat': [lat],
        'StartLon': [lon],
        'Hour': [int(hour)]
    })
    prediction = trained_model.predict(input_data)
    return int(prediction[0])

# Function to plot the predicted cluster and nearby clusters on a map using folium
def get_nearby_clusters(lat, lon, hour, num_nearby_clusters=3):
    # Filter data for the specified hour
    filtered_df = df[df['Hour'] == hour]

    if filtered_df.empty:
        return None

    # Perform clustering (AgglomerativeClustering)
    coords = filtered_df[['StartLat', 'StartLon']].values
    agglomerative = AgglomerativeClustering(n_clusters=min(70, len(filtered_df))).fit(coords)
    filtered_df['Cluster'] = agglomerative.labels_

    # Calculate the distances from the input location (lat, lon)
    filtered_df['Distance'] = np.sqrt((filtered_df['StartLat'] - lat) ** 2 + (filtered_df['StartLon'] - lon) ** 2)

    # Get the nearby clusters (sorted by distance)
    nearby_clusters = filtered_df.groupby('Cluster').apply(lambda x: x['Distance'].min()).sort_values().head(num_nearby_clusters + 1).index

    return filtered_df[filtered_df['Cluster'].isin(nearby_clusters)][['StartLat', 'StartLon', 'Cluster']]

# Define the route for the prediction
@app.route('/predict', methods=['GET'])
def predict():
    # Get parameters from the URL
    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)
    hour = request.args.get('hour', type=str)

    if lat is None or lon is None or hour is None:
        return jsonify({"error": "Please provide valid lat, lon, and hour parameters."}), 400

    # Predict the number of people
    predicted_people = predict_num_people(lat, lon, hour)

    # Get nearby clusters
    nearby_clusters = get_nearby_clusters(lat, lon, hour)

    if nearby_clusters is None or nearby_clusters.empty:
        return jsonify({"message": "No nearby clusters found for the given hour."}), 404

    # Format the response
    response = {
        "predicted_people": predicted_people,
        "nearby_clusters": nearby_clusters.to_dict(orient='records')
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
