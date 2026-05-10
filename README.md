# Healthcare Accessibility & Demand Modeling: Identifying "Healthcare Deserts"

## 1. Project Overview
This project identifies underserved geographic areas ("healthcare deserts") by modeling the relationship between healthcare facility density and demographic demand indicators. This analysis simulates a real-world use case for a **Healthcare Real Estate Investment Trust (REIT)** seeking to optimize facility placement in high-demand areas.

### 1.1 Research Question
How does the density of healthcare facilities correlate with demographic demand indicators (median age, population, and income) across ZIP codes, and which areas can be identified as "healthcare deserts" based on this relationship?

## 2. Technical Rigor: Logistic Regression from Scratch
A primary objective of this project was to demonstrate mathematical and engineering rigor by implementing a Logistic Regression model using only **NumPy** for matrix operations, rather than relying on "black-box" libraries.

### 2.1 Core Mathematical Functions
* **Sigmoid Function:** Maps real-valued linear outputs into a probability value between 0 and 1.
* **Binary Cross-Entropy (Log Loss):** Used as the cost function to penalize incorrect classifications.
* **Gradient Descent:** An optimization algorithm that iteratively calculates partial derivatives to update weights and minimize error.

## 3. Project Structure
The project is organized as a modular Python pipeline following PEP 8 standards:

* `src/data_acquisition.py`: Handles Census API calls with error handling.
* `src/preprocessing.py`: Manages data cleaning, merging.
* `src/model.py`: Contains the custom NumPy Logistic Regression implementation, Sklearn comparison and feature engineering (scaling and bias integration).
* `src/analysis.py`: Evaluates model performance via confusion matrices and cost history visualization.
* `results.ipynb`: The end-to-end notebook for presentation and stakeholder visualization.

## 4. Key Findings & Strategic Results
The model successfully identified high-priority investment areas based on the intersection of aging populations and low healthcare supply.

### 4.1 Identified "Healthcare Deserts"
Based on the final model output, the following ZIP codes in the Houston and surrounding area are identified as high-priority for new facility investment:

| ZIP Code | Median Age | Median Household Income | Priority Rank |
| :--- | :--- | :--- | :--- |
| **77431** | 88.2 | $16,513 | 1 (Critical) |
| **77359** | 72.2 | $55,331 | 2 (High) |
| **77475** | 70.1 | $58,255 | 3 (High) |

### 4.2 Conclusion
By quantifying the demand-supply gap, this project provides a data-driven roadmap for the REIT to expand its infrastructure where it will have the highest community impact and demand fulfillment.