import argparse
import logging
import os
import time
from typing import Optional

import pandas as pd
import requests

from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("CENSUSKEY")

print(api_key)
print("API key loaded successfully.")

CENSUS_BASE_URL = "https://api.census.gov/data/2022/acs/acs5"

ACS_VARIABLES: dict[str, str] = {
    "B01002_001E": "median_age",
    "B01001_001E": "total_population",
    "B19013_001E": "median_household_income",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)



def fetch_census_data() -> pd.DataFrame:
    """
    Fetch ACS 5-Year Estimates for all U.S. ZCTAs and filter for Texas.

    The Census API does not allow state-level filtering for ZCTAs (ZIP Code Tabulation Areas).
    This function retrieves national data for key demographic indicators and filters 
    locally for Texas prefixes (75-79).

    Returns
    -------
    pd.DataFrame
        A DataFrame indexed by ZIP code containing:
        - zip_code : str
        - median_age : float
        - total_population : int
        - median_household_income : float
        
    Notes
    -----
    - Sentinel values (e.g., -666666666) are converted to NaN.
    - Household income and age are coerced to numeric types.
    - Data is sourced from the 2022 ACS 5-Year Estimates.
    """
    variable_str = ",".join(ACS_VARIABLES.keys())
    params = {
        "get": variable_str,
        "for": "zip code tabulation area:*",
        "key": api_key,
    }

    logger.info("Fetching Census ACS data for Texas…")

    try:
        response = requests.get(CENSUS_BASE_URL, params=params, timeout=30)
        response.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        logger.error("Census API HTTP error: %s", exc)
        raise
    except requests.exceptions.ConnectionError as exc:
        logger.error("Census API connection error: %s", exc)
        raise

    raw_json         = response.json()
    headers, rows    = raw_json[0], raw_json[1:]
    df               = pd.DataFrame(rows, columns=headers)

    rename_map: dict[str, str] = {
        **ACS_VARIABLES,
        "zip code tabulation area": "zip_code",
    }
    df.rename(columns=rename_map, inplace=True)
    # Filter for Texas ZIP codes (Texas ZIPs start with 75, 76, 77, 78, 79)
    df = df[df['zip_code'].astype(str).str.startswith(('75', '76', '77', '78', '79'))].copy()

    df.drop(columns=["state"], errors="ignore", inplace=True)

    numeric_cols = ["median_age", "total_population", "median_household_income"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        # Replace Census sentinel values with NaN
        df[col] = df[col].where(df[col] > 0, other=pd.NA)

    df = df[["zip_code"] + numeric_cols].copy()
    logger.info("Census fetch complete — %d ZIP codes retrieved.", len(df))
    return df

def acquire_all_data() -> pd.DataFrame:
    
    census_df = fetch_census_data()
    zip_codes = census_df["zip_code"].dropna().unique().tolist()

    return census_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Acquire data for healthcare accessibility demand modeling.")
    parser.add_argument("--state_fips", type=str, default="48", help="Two-digit FIPS code for the state (default: 48 for Texas).")
    args = parser.parse_args()

    start_time = time.time()
    df = acquire_all_data()
    elapsed_time = time.time() - start_time
    logger.info("Data acquisition completed in %.2f seconds.", elapsed_time)

    output_path = "data/raw_census_data.csv"
    df.to_csv(output_path, index=False)
    logger.info("Data saved to %s.", output_path)