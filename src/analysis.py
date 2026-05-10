import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import sys
import os
import folium
import geopandas as gpd
import folium

sys.path.append(os.path.abspath(".."))

def plot_cost_history(cost_history):
    """
    Visualize the convergence of the gradient descent algorithm.
    
    Args:
        cost_history (list): List of cost values recorded during training.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(cost_history)), cost_history, color='#2c3e50', linewidth=2)
    plt.title("Model Convergence: Cost Function over Time", fontsize=14)
    plt.xlabel("Iterations (per 100 steps)", fontsize=12)
    plt.ylabel("Log-Loss Cost", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def get_feature_importance(weights, feature_names):
    """
    Map the trained weights back to their feature names to show impact.
    
    Args:
        weights (ndarray): Trained weights (including bias at index 0).
        feature_names (list): Names of the demographic features.
    """
    # Convert weights to a flat array if it's a list or nested array
    weights = np.array(weights).flatten()
    
    # Check if we have a bias term (N+1 weights for N features)
    if len(weights) == len(feature_names) + 1:
        # Ignore the first weight (bias) to focus on demographic features
        model_weights = weights[1:]
    elif len(weights) == len(feature_names):
        # No bias term detected, use all weights
        model_weights = weights
    else:
        # Trigger a helpful error if they still don't match
        raise ValueError(f"Weight mismatch! Model has {len(weights)} weights, "
                         f"but you provided {len(feature_names)} feature names.")

    # Create the sorted importance dataframe
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Weight': model_weights
    }).sort_values(by='Weight', ascending=False)

    # Visualization code...
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Weight', y='Feature', data=importance_df, palette='viridis', hue='Feature', legend=False)
    plt.title("Feature Importance (Impact on Healthcare Desert Status)")
    plt.tight_layout()
    plt.show()
    
    return importance_df

def plot_confusion_matrix_visual(y_true, y_pred):
    """
    Create a professional-grade heatmap of the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Adequate', 'Underserved'], 
                yticklabels=['Adequate', 'Underserved'])
    plt.title("Confusion Matrix: Healthcare Desert Predictions", fontsize=14)
    plt.ylabel('Actual Status')
    plt.xlabel('Predicted Status')
    plt.show()

def summarize_deserts(df, target_col):
    """
    Provide a high-level summary of the findings.
    """
    total_zips = len(df)
    underserved_count = df[target_col].sum()
    pct = (underserved_count / total_zips) * 100
    
    print("--- Executive Summary ---")
    print(f"Total ZIP Codes Analyzed: {total_zips}")
    print(f"Identified Healthcare Deserts: {underserved_count} ({pct:.1f}%)")
    
    # Show top 5 highest-risk features based on simple mean differences
    summary = df.groupby(target_col).mean(numeric_only=True)
    return summary

def create_healthcare_desert_map(df, shapefile_path, target_col='is_desert'):
    """
    Reads the Houston Shapefile and joins it with the project dataframe 
    to create a geographic visualization of healthcare deserts.
    """
    # 1. Load the Shapefile (GeoPandas can read directly from the .zip)
    gdf = gpd.read_file(shapefile_path)

    # 2. Standardize the Join Keys (Ensure ZIP codes are strings and match in format)
    gdf['ZIPCODE'] = gdf['ZIPCODE'].astype(str)
    df['zip_code'] = df['zip_code'].astype(str)

    # 3. Merge your model results into the Geographic Data
    merged_gdf = gdf.merge(df, left_on='ZIPCODE', right_on='zip_code')

    # 4. Initialize Folium Map
    m = folium.Map(location=[29.7604, -95.3698], zoom_start=10, tiles='cartodbpositron')

    # 5. Add the Choropleth layer using the merged GeoDataFrame
    folium.Choropleth(
        geo_data=merged_gdf,
        name="Healthcare Deserts",
        data=merged_gdf,
        columns=['zip_code', target_col],
        key_on="feature.properties.zip_code", 
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name="Desert Classification (1 = Underserved)"
    ).add_to(m)

    return m
