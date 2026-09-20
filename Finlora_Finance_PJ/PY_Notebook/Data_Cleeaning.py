#------------------------------------------
## IMPORTING LIBRARIES
#------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import probplot
import os
import joblib
import plotly.express as px


#-------------------------------------------
## LOAD DATASET
#-------------------------------------------
# 1. Load the first CSV file
account_data = pd.read_csv(
    r"C:\Users\User\Documents\AMDARI_IMS\Finlora_Finance_PJ\Dataset\Raw_data\finlora_accounts.csv")

# Data Summary
account_summary = pd.DataFrame({
    "Data Type": account_data.dtypes.astype(str),
    "Missing Values": account_data.isna().sum(),
    "% Missing Values": (account_data.isna().mean() * 100).round(2),
    "Unique Values": account_data.nunique(),
    "Duplicate Values": [account_data.duplicated(subset=[col]).sum() for col in account_data.columns]
}).sort_values(by="% Missing Values", ascending=False)

account_summary


# 2. Load the second CSV file
transaction_data = pd.read_csv(
    r"C:\Users\User\Documents\AMDARI_IMS\Finlora_Finance_PJ\Dataset\Raw_data\finlora_transactions.csv")

# Data Summary
transaction_summary = pd.DataFrame({
    "Data Type": transaction_data.dtypes.astype(str),
    "Missing Values": transaction_data.isna().sum(),
    "% Missing Values": (transaction_data.isna().mean() * 100).round(2),
    "Unique Values": transaction_data.nunique(),
    "Duplicate Values": [transaction_data.duplicated(subset=[col]).sum() for col in transaction_data.columns]
}).sort_values(by="% Missing Values", ascending=False)

transaction_summary


# 3. Merging the two datasets
# Remove duplicate account-level columns from Dataset 2
transaction_data = transaction_data.drop(
    columns=["account_type", "kyc_tier", "currency", "home_country"])

# Merge datasets
merged_data = pd.merge(account_data, transaction_data, on="account_id", how="inner")

# Display columns
print("Merged Dataset Columns:")
print(merged_data.columns.tolist())

# Merged Data Summary
merged_summary = pd.DataFrame({
    "Data Type": merged_data.dtypes.astype(str),
    "Missing Values": merged_data.isna().sum(),
    "% Missing Values": (merged_data.isna().mean() * 100).round(2),
    "Unique Values": merged_data.nunique(),
    "Duplicate Values": [merged_data.duplicated(subset=[col]).sum() for col in merged_data.columns]
}).sort_values(by="% Missing Values", ascending=False)

merged_summary


# 4. Save the merged dataset as a new CSV file
output_folder = r"C:\Users\User\Documents\AMDARI_IMS\Finlora_Finance_PJ\Dataset\Merged_data"

os.makedirs(output_folder, exist_ok=True)

# Save the merged dataset
output_file = os.path.join(
    output_folder,
    "finlora_merged.csv"
)

merged_data.to_csv(
    output_file,
    index=False
)

# 5. Confirm the result
print("Merged dataset created successfully!")
print("Rows and columns:", merged_data.shape)
print("File saved as: finlora_merged.csv")


## INVESTIGATING THE TARGET VARIABLE
print(merged_data["is_fraud"].value_counts())
print("="*20)
fraud_distribution = (merged_data["is_fraud"].value_counts(normalize=True).mul(100).round(2))
print(fraud_distribution)


# Visulization of the target variable
fraud_distribution = (merged_data["is_fraud"].value_counts(normalize=True).mul(100).round(2).reset_index())

fraud_distribution.columns = ["is_fraud", "proportion"]

# Convert fraud codes to meaningful labels
fraud_distribution["is_fraud"] = fraud_distribution["is_fraud"].map({0: "Not Fraud", 1: "Fraud"})

fig = px.sunburst(fraud_distribution, path=["is_fraud"], values="proportion", title="Fraud Distribution")

fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", showlegend=False,  width=800, height=500,)

fig.show()
