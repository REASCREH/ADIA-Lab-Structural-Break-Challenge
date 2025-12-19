# <span style="color:#2E86C1">1. Project Aim & Prediction Goal</span>

The primary objective of this project is to detect **Structural Breaks** in univariate time series data for the **ADIA Lab Structural Break Challenge**. 

In time series analysis, a structural break occurs when the underlying process governing the data generation changes abruptly. Our goal is to classify each given time series based on a specific "boundary point":

* <span style="color:#E74C3C">**Structural Break (Class 1):**</span> The data behavior (mean, variance, or trend) changed significantly after the boundary point.
* <span style="color:#3498DB">**No Break (Class 0):**</span> The data behavior remained consistent across the boundary point.

### <span style="color:#2E86C1">Model Performance Summary</span>
The solution utilizes a **LightGBM** model, which achieved the following results during local evaluation:
* **Training ROC-AUC:** <span style="color:#27AE60">**0.9224**</span> (Excellent class separation).
* **Local Test ROC-AUC:** **0.6653** (Generalization on unseen data).
* **Precision (Break Detection):** **94%** (High reliability in "Break" predictions).
* **Accuracy:** **82%** (Overall correct classification rate).



---

# <span style="color:#2E86C1">2. Dataset Overview</span>

The dataset is comprised of synthetic univariate time series designed to mimic real-world scenarios across multiple domains:

* **Total Data Volume:** Over **23.7 Million rows** of raw data.
* **Samples:** **10,001 unique time series IDs** for training.
* **Time Steps:** Each series contains approximately **1,000 to 5,000** observations.
* **Domains Simulated:** Finance, Climatology, Industrial Sensors, and Healthcare monitoring.

---

# <span style="color:#2E86C1">3. The Challenge in this Dataset</span>

1. **Signal vs. Noise:** Structural changes are often subtle and easily confused with random noise.
2. **Computational Scale:** Processing 23 million rows requires efficient memory management and optimized feature extraction.
3. **Generalization:** The model must learn universal statistical signatures of "change" that apply across all simulated domains.

---

# <span style="color:#2E86C1">4. How the Code Solves the Challenge</span>

This notebook implements an end-to-end pipeline that handles data engineering and predictive modeling:

### <span style="color:#27AE60">A. Advanced Feature Engineering</span>
Instead of using raw values, we extract **49 distinct statistical features** for each ID:
* **Statistical Moments:** Mean, Standard Deviation, Skewness, and Kurtosis.
* **Frequency Domain (FFT):** Uses Fast Fourier Transform to detect shifts in the data's "rhythm."
* **Complexity Analysis (Entropy):** Measures how chaotic or unpredictable the series is.
* **Trend Slopes:** Uses `polyfit` to identify consistent upward or downward movement.

### <span style="color:#27AE60">B. Efficient Modeling with LightGBM</span>
* **Gradient Boosting:** We use **LightGBM** for its speed and ability to handle large-scale data.
* **Regularization:** Uses `lambda_l1` and `lambda_l2` to prevent the model from overfitting to noise.
* **Determinism:** Ensures consistent outputs for competition eligibility.

---

# <span style="color:#2E86C1">5. Final Result Summary Table</span>

| Category | Metric / Detail | Value |
| :--- | :--- | :--- |
| **Performance** | **Training ROC-AUC** | <span style="color:#27AE60">**0.9224**</span> |
| **Performance** | **Validation ROC-AUC** | **0.6653** |
| **Processing** | **Total Rows Analyzed** | **23,715,734** |
| **Engineering** | **Features Extracted** | **49** |
| **Reliability** | **Precision (Class 1)** | **0.94** |
| **Stability** | **Determinism Check** | <span style="color:#27AE60">**Passed**</span> |

---
*Note: This solution follows the official ADIA Lab & CrunchDAO submission template requirements.*
