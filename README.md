# Data Cleaning & Visualization Project: Palmer Penguins Analysis

This repository contains a complete data processing, cleaning, and visualization workflow for the Palmer Penguins dataset. The project fulfills the requirements for Task 1 of the data track, highlighting automated cleaning, outlier mitigation, and visual storytelling.

## 📊 Dashboard Preview
The workflow completely processes the raw variables and exports a high-resolution, multi-panel insights dashboard saved as `data_insights_dashboard.png`.

---

## 🧹 Data Cleaning & Preprocessing Workflow
The pipeline automates several fundamental data engineering steps to ready the data for downstream machine learning applications:

*   **Duplicate Elimination:** Scans the data to purge any exact duplicate records.
*   **Missing Value Imputation:** 
    *   **Numeric Columns:** Structurally handles missing features (such as missing physical measurements) by imputing values with the target column's **median**.
    *   **Categorical Columns:** Fills structural gaps (such as missing sex data) utilizing the column **mode** (most frequent occurrence).
*   **Outlier Mitigation:** Applies the mathematical **Interquartile Range (IQR) method** to detect anomalies in critical variables like `flipper_length_mm` and dynamically caps boundaries to a mathematically sound interval of `[155.50, 247.50]`.

---

## 📈 Key Insights & Data Storytelling

Based on the generated dashboard visualizations, three major analytical conclusions were uncovered:

1.  **Bimodal Biological Distributions:** The flipper length distribution shows a distinct bimodal curve with two clear peaks around 195mm and 215mm. This suggests the data holds prominent underlying sub-populations.
2.  **Significant Species Morphometrics:** There is a distinct mass disparity across species groups. The **Gentoo** species displays a substantially higher median body mass (~5,000g) compared to the tightly clustered distributions of the **Adelie** and **Chinstrap** species (~3,700g).
3.  **Powerful Covariation:** The feature correlation matrix reveals an incredibly strong positive linear correlation (**0.87**) between a penguin's flipper length and its overall body mass, identifying it as a dominant metric for any predictive modeling tasks.

---

## 🚀 How to Run the Project

1. **Install dependencies:**
```bash
   pip install -r requirements.txt
