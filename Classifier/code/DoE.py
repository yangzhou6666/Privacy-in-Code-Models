import os
import re
import pandas as pd
import itertools
from collections import defaultdict
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np

# ---------------------------
# Data Extraction and Saving
# ---------------------------

# Parameters for the experiment
seed_corpus = [30, 59, 98]

victims = [
    "codeparrot-small",
    "CodeGPT-small-java-adaptedGPT2",
    "PolyCoder-0.4B",
    "codegen-350M-multi",
    "gpt-neo-125m",
    "PolyCoder-160M"
]

surrogates = [
    "transformer",
    "gpt2",
    "rnn",
    "CodeGPT-small-java-adaptedGPT2"
]

epochs = [1, 2, 3, 4]

ratios = [10, 20]

# Base path for logs
base_path = "../classifier_save/PTM3/javaCorpus"

# Regular expression to match TPR and FPR values
tpr_fpr_pattern = re.compile(r'"TPR":\s*([\d.]+),\s*\n\s*"FPR":\s*([\d.]+)')

# Dictionary to store intermediate data for averaging
intermediate_data = defaultdict(list)

# Generate all combinations of surrogate models, ratios, victims, and epochs
combinations = itertools.product(surrogates, ratios, victims, epochs)

# Iterate over each combination (ignoring seed for now)
for surrogate, ratio, victim, epoch in combinations:
    # For each combination, accumulate TPR and FPR values over seeds
    tpr_values = []
    fpr_values = []

    # Iterate over seeds and read corresponding log files
    for seed in seed_corpus:
        log_path = os.path.join(
            base_path,
            surrogate,
            str(ratio),
            str(seed),
            f"log_eval_{victim}_{epoch}.txt"
        )

        if os.path.exists(log_path):
            # Open and read the log file
            with open(log_path, 'r') as log_file:
                log_content = log_file.read()

                # Search for the TPR and FPR using the regular expression
                matches = tpr_fpr_pattern.findall(log_content)

                if matches:
                    # Extract the last match of TPR and FPR (most recent in log)
                    tpr, fpr = matches[-1]  # get the most recent TPR and FPR values

                    # Append to the seed-specific lists for averaging
                    tpr_values.append(float(tpr))
                    fpr_values.append(float(fpr))
        else:
            print(f"Log file does not exist: {log_path}")

    if tpr_values and fpr_values:
        # Calculate average TPR and FPR across seeds
        avg_tpr = sum(tpr_values) / len(tpr_values)
        avg_fpr = sum(fpr_values) / len(tpr_values)

        # Store the averaged results
        intermediate_data[(surrogate, ratio, victim, epoch)] = {
            'Surrogate': surrogate,
            'Ratio': ratio,
            'Victim': victim,
            'Epoch': epoch,
            'Avg_TPR': avg_tpr,
            'Avg_FPR': avg_fpr
        }

# Convert the intermediate data to a DataFrame for further use
extracted_data = pd.DataFrame(intermediate_data.values())

# Save the DataFrame to a CSV file
csv_file_path = 'extracted_data.csv'
extracted_data.to_csv(csv_file_path, index=False)
print(f"Data saved to {csv_file_path}")

# Optionally, display the results
print(extracted_data.head())

# ---------------------------
# ANOVA Analysis with Interactions
# ---------------------------

# Load the data
data = pd.read_csv(csv_file_path)

# Ensure that the factors are treated as categorical variables
categorical_variables = ['Surrogate', 'Ratio', 'Victim', 'Epoch']
for var in categorical_variables:
    data[var] = data[var].astype('category')

# Check for NaNs
print("\nChecking for NaNs in data:")
print(data.isnull().sum())

# Check for infinite values
print("\nChecking for infinite values in data:")
print(np.isinf(data.select_dtypes(include=[np.number])).sum())

# Remove rows with NaNs or infinite values
data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna()
# Ensure that data is not empty after cleaning
if data.empty:
    raise ValueError("No data available after removing NaNs and infinite values.")

# Build the ANOVA model with interactions up to three-way interactions
formula = 'Avg_TPR ~ (C(Surrogate) + C(Ratio) + C(Victim) + C(Epoch)) ** 3'

# Perform the ANOVA
model = ols(formula, data=data).fit()

# Calculate Degrees of Freedom
n_observations = data.shape[0]
n_parameters = model.df_model + 1  # +1 for the intercept
df_resid = model.df_resid

print(f"\nNumber of observations: {n_observations}")
print(f"Number of parameters: {n_parameters}")
print(f"Degrees of freedom for residuals: {df_resid}")

if df_resid <= 0:
    raise ValueError("Degrees of freedom for residuals is zero or negative. Simplify the model.")

anova_results = sm.stats.anova_lm(model, typ=2)

# Calculate the Sum of Squares Total (SST) and the percentage for each factor
anova_results['SST %'] = (anova_results['sum_sq'] / anova_results['sum_sq'].sum()) * 100
anova_results['Degrees of Freedom'] = anova_results['df']

# Reorder columns for better readability
anova_results = anova_results[['Degrees of Freedom', 'sum_sq', 'SST %', 'F', 'PR(>F)']]

# Print the ANOVA results
print("\nANOVA Results:")
print(anova_results)