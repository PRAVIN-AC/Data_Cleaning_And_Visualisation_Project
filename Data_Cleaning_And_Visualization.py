import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. SETUP & DATA LOADING
# ==========================================
print("📥 Fetching the Palmer Penguins dataset...")
url = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv"
df = pd.read_csv(url)

print(f"Dataset loaded successfully. Initial Shape: {df.shape}")
print("\n--- Initial Missing Value Counts ---")
print(df.isnull().sum())

# ==========================================
# 2. DATA CLEANING
# ==========================================
print("\n🧹 Starting Data Cleaning Phase...")

# Step A: Remove exact duplicate rows (if any)
initial_count = len(df)
df = df.drop_duplicates()
print(f"Removed {initial_count - len(df)} duplicate rows.")

# Step B: Handle missing values
# For numeric columns: fill with the median
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

# For categorical columns: fill with the mode (most frequent value)
categorical_cols = df.select_dtypes(include=[object, 'category']).columns
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)

# Step C: Outlier Detection and Handling (using IQR Method on 'flipper_length_mm')
# We'll identify and cap outliers to keep the data robust
target_numeric = 'flipper_length_mm'
Q1 = df[target_numeric].quantile(0.25)
Q3 = df[target_numeric].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Cap values outside the bounds
df[target_numeric] = np.clip(df[target_numeric], lower_bound, upper_bound)
print(f"Capped outliers for '{target_numeric}' using boundaries: [{lower_bound:.2f}, {upper_bound:.2f}]")
print(f"Remaining Missing Values in Dataset: {df.isnull().sum().sum()}")

# ==========================================
# 3. DATA VISUALIZATION & DASHBOARD
# ==========================================
print("\n📊 Generating Dashboard Visualizations...")

# Set a professional, clean visual style
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Data Insights Dashboard: Palmer Penguins Analysis', fontsize=16, fontweight='bold', y=1.05)

# Plot 1: Distribution of Flipper Length (Histogram + KDE)
sns.histplot(ax=axes[0], data=df, x='flipper_length_mm', kde=True, color='royalblue', bins=20)
axes[0].set_title('Distribution of Flipper Length', fontsize=12, fontweight='semibold')
axes[0].set_xlabel('Flipper Length (mm)')
axes[0].set_ylabel('Count')

# Plot 2: Relationship Between Species and Body Mass (Boxplot)
sns.boxplot(ax=axes[1], data=df, x='species', y='body_mass_g', palette='Set2', hue='species', legend=False)
axes[1].set_title('Body Mass Distribution by Species', fontsize=12, fontweight='semibold')
axes[1].set_xlabel('Species')
axes[1].set_ylabel('Body Mass (g)')

# Plot 3: Feature Correlation Heatmap
# Filter to only numeric columns for correlation matrix
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(ax=axes[2], data=numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, cbar=True)
axes[2].set_title('Feature Correlation Matrix', fontsize=12, fontweight='semibold')

# Adjust layout and save the dashboard image
plt.tight_layout()
output_filename = 'data_insights_dashboard.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Success! Dashboard saved as a high-resolution image: '{output_filename}'")
plt.show()

# ==========================================
# 4. DATA STORYTELLING REPORT
# ==========================================
print("\n📝 ==========================================")
print("          DATA STORYTELLING REPORT            ")
print("==============================================")
print(f"• Final Cleaned Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns.")
print("• Cleaning Actions Taken: Missing structural values filled using median/mode strategies.")
print("• Key Observation 1: Flipper lengths show a distinct bimodal distribution pattern.")
print("• Key Observation 2: The Gentoo species exhibits a significantly higher median body mass")
print("  compared to Adelie and Chinstrap species.")
print("• Key Observation 3: Strong positive correlations exist between flipper length and body mass.")
print("==============================================")