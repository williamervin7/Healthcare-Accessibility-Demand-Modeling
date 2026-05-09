import argparse
import logging
import os
import time

import pandas as pd
import requests

from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("CENSUSKEY")

CENSUS_BASE_URL = "https://api.census.gov/data/2022/acs/acs5"

ACS_VARIABLES: dict[str, str] = {
    "B01002_001E": "median_age",
    "B01001_001E": "total_population",
    "B19013_001E": "median_household_income",
     # Education
    "B15003_022E": "bachelors_degree",
    "B15003_023E": "masters_degree",

    # Poverty
    "B17001_002E": "below_poverty",

    # Employment
    "B23025_005E": "unemployed",

    # Insurance
    "B27010_017E": "uninsured_adults",

    # Disability
    "C18108_001E": "with_disability",

    # Transportation
    "B08201_002E": "no_vehicle_households",
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

    Retrieves national data for a comprehensive set of demographic and socio-economic
    indicators (Age, Income, Education, Poverty, Employment, Insurance, Disability,
    and Transportation) and filters locally for Texas prefixes (75-79).

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the following features per ZIP code:
        - zip_code                : str
        - median_age              : float
        - total_population        : int
        - median_household_income : float
        - bachelors_degree        : int
        - masters_degree          : int
        - below_poverty           : int
        - unemployed              : int
        - uninsured_adults        : int
        - with_disability         : int
        - no_vehicle_households   : int

    Notes
    -----
    - API constraint: ZCTA-level queries do not support state-level filtering.
    - Data quality: Sentinel values (e.g., -666666666) are coerced to NaN.
    - Year: Utilizing 2022 ACS 5-Year Estimates.
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

    # Map Census codes to readable names defined in ACS_VARIABLES
    rename_map: dict[str, str] = {
        **ACS_VARIABLES,
        "zip code tabulation area": "zip_code",
    }
    df.rename(columns=rename_map, inplace=True)
    # Filter for Texas ZIP codes (Texas ZIPs start with 75, 76, 77, 78, 79)
    df = df[df['zip_code'].astype(str).str.startswith(('75', '76', '77', '78', '79'))].copy()

    df.drop(columns=["state"], errors="ignore", inplace=True)

    # Automatically identify all value columns to convert to numeric
    numeric_cols = list(ACS_VARIABLES.values())
 
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        # Replace Census sentinel values with NaN
        df[col] = df[col].where(df[col] > 0, other=pd.NA)

    df = df[["zip_code"] + numeric_cols].copy()
    logger.info("Census fetch complete — %d ZIP codes retained with %d features.", 
                len(df), len(numeric_cols))
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Acquire US Census data for healthcare accessibility demand modeling.")
    args = parser.parse_args()

    start_time = time.time()
    # Call the data acquisition function and save the results
    df = fetch_census_data()
    elapsed_time = time.time() - start_time
    logger.info("Data acquisition completed in %.2f seconds.", elapsed_time)

    output_path = "data/raw_census_data.csv"
    df.to_csv(output_path, index=False)
    logger.info("Data saved to %s.", output_path)