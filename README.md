# Data Imputation Project

This project demonstrates iterative techniques for imputing missing values in a dataset. The main methods used are Random Forest for discrete values and Linear Regression for continuous values. This project includes multiple approaches for filling missing values in a dataset, leveraging machine learning models for predictive imputation.

## Features
- **Iterative Imputation for Discrete Columns**: Missing values in categorical columns are filled using RandomForestClassifier.
- **Iterative Imputation for Continuous Columns**: Missing values in continuous columns are filled using Linear Regression.
- **Initial Imputation Methods**: Continuous columns are initially imputed using mean, median, or mode.
- **Comparison Function**: Compares the original and imputed values to visualize the filled data.

## Methods
1. **Iterative Cleaning for Discrete Values**: This method uses RandomForestClassifier to predict missing values for discrete columns. The process is repeated for multiple iterations to ensure robustness.
2. **Iterative Regression Imputation**: This method uses Linear Regression to predict and fill missing values in continuous columns iteratively.
3. **Initial Fill for Continuous Values**: Before the iterative imputation, missing values in continuous columns are initially filled using the method specified in a configuration dictionary (mean, median, or mode).

## Installation

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/your-username/data-imputation-project.git
