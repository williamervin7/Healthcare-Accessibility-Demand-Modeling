import os
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath("."))

def get_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads raw census and healthcare facility datasets from the local data directory.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the raw Census 
        DataFrame and the raw Provider of Services (POS) DataFrame.
        
    Raises:
        FileNotFoundError: If the required CSV files are missing from the data folder.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_folder = os.path.join(base_dir, "data")
    
    census_path = os.path.join(data_folder, "raw_census_data.csv")
    healthcare_path = os.path.join(data_folder, "raw_pos.csv")
    
    if not os.path.exists(census_path) or not os.path.exists(healthcare_path):
        raise FileNotFoundError(f"Raw data files not found in: {data_folder}")

    census_df = pd.read_csv(census_path)
    healthcare_df = pd.read_csv(healthcare_path)
    
    return census_df, healthcare_df

def merge_data(census_df: pd.DataFrame, healthcare_df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes ZIP code formats and merges demand (Census) and supply (Healthcare) data.

    Args:
        census_df (pd.DataFrame): Raw demographic data.
        healthcare_df (pd.DataFrame): Raw healthcare facility data.

    Returns:
        pd.DataFrame: A merged dataset joined on a 5-digit ZIP code string.
    """
    # Standardize ZIP code formats to 5-digit strings
    census_df['zip_code'] = census_df['zip_code'].astype(str).str.strip().str.zfill(5)
    healthcare_df['zip_cd'] = healthcare_df['zip_cd'].astype(str).str.strip().str.zfill(5)
    
    # Group healthcare facilities by ZIP to get a frequency count
    healthcare_counts = healthcare_df.groupby('zip_cd').size().reset_index(name='facility_count')
    
    # Left merge to keep all Census ZIPs; fill missing facility counts with 0
    merged_df = pd.merge(census_df, healthcare_counts, left_on='zip_code', right_on='zip_cd', how='left')
    merged_df['facility_count'] = merged_df['facility_count'].fillna(0)

    merged_df.drop(columns=['zip_cd'], inplace=True)  # Remove redundant ZIP column after merge
    
    return merged_df

def calculate_metrics(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs data cleaning, median imputation, and calculates target variables.

    Calculates facility density (continuous) and defines 'is_desert' (binary) 
    for classification tasks.

    Args:
        merged_df (pd.DataFrame): The raw merged dataset from merge_data().

    Returns:
        pd.DataFrame: A cleaned DataFrame ready for regional filtering or modeling.
    """
    # Remove records missing core demographic predictors
    df_clean = merged_df.dropna(subset=['median_age', 'total_population', 'median_household_income']).copy()
    
    # Ensure no division by zero for density calculations
    df_clean = df_clean[df_clean['total_population'] > 0].copy()
    
    # Impute missing secondary features with the median to preserve sample size
    impute_cols = [
        'unemployed', 'uninsured_adults', 'no_vehicle_households', 
        'with_disability', 'bachelors_degree', 'masters_degree', 'below_poverty'
    ]
    for col in impute_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # Calculate Facility Density per 1,000 residents and cap at 99th percentile
    df_clean['facility_density'] = (df_clean['facility_count'] / df_clean['total_population']) * 1000
    df_clean['facility_density'] = df_clean['facility_density'].clip(upper=df_clean['facility_density'].quantile(0.99))
    
    # Define Classification Target: 1 if 0 facilities exist, else 0
    df_clean['is_desert'] = (df_clean['facility_count'] == 0).astype(int)
    
    return df_clean

def filter_by_city(df: pd.DataFrame, city_name: str) -> pd.DataFrame:
    """
    Filters the dataset into city-specific cohorts based on ZIP prefixes.

    Args:
        df (pd.DataFrame): The preprocessed global DataFrame.
        city_name (str): The name of the city ('houston', 'dallas', or 'austin').

    Returns:
        pd.DataFrame: A subset of the data corresponding to the selected metro area.
    """
    prefixes = {
        'houston': ('770', '772', '773', '774', '775'),
        'dallas': ('750', '751', '752', '753'),
        'austin': ('786', '787')
    }
    
    target_prefixes = prefixes.get(city_name.lower())
    if not target_prefixes:
        raise ValueError("Unsupported city. Please choose 'houston', 'dallas', or 'austin'.")
        
    return df[df['zip_code'].str.startswith(target_prefixes)].copy()

def preprocess_data() -> pd.DataFrame:
    """
    Main execution function to orchestrate the preprocessing pipeline.

    Returns:
        pd.DataFrame: A fully processed dataset ready for regional subsetting.
    """
    census_df, healthcare_df = get_data()
    merged_df = merge_data(census_df, healthcare_df)
    final_df = calculate_metrics(merged_df)
    
    # Define the save path relative to this script
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_path = os.path.join(base_dir, "data", "processed_data.csv")
    
    # Save the cleaned data
    final_df.to_csv(save_path, index=False)
    print(f"File successfully saved to: {save_path}")

    return final_df

if __name__ == "__main__":
    # Test run to verify
    df = preprocess_data()
    print(f"Preprocessing complete. Rows: {df.shape[0]}")
    print(f"Deserts found in Texas: {df['is_desert'].sum()}")
    houston_df = filter_by_city(df, 'houston')
    print(f"Deserts in Houston: {houston_df.shape[0]}")
    dallas_df = filter_by_city(df, 'dallas')
    print(f"Deserts in Dallas: {dallas_df.shape[0]}")
    austin_df = filter_by_city(df, 'austin')
    print(f"Deserts in Austin: {austin_df.shape[0]}")