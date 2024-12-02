# %%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression


# %%
data = pd.read_csv("processed_car_data.csv")

# %%
data['wheelbase'].mean()

# %%
data.head()

# %% [markdown]
# random_null :  a function which helps to randomly change values to null according to the percentage we mention 

# %%
def random_null(df, column_list, percentage):
    df_copy = df.copy()  # Create a copy of the dataframe
    percentage = percentage / 100
    total_rows = df_copy.shape[0]
    
    for column_name in column_list:
        num_nulls = int(total_rows * percentage)
        null_indices = np.random.choice(df_copy.index, num_nulls, replace=False)
        
        # Convert integer columns to nullable integers
        if pd.api.types.is_integer_dtype(df_copy[column_name]):
            df_copy[column_name] = df_copy[column_name].astype("Int64")
        
        # Assign NaN values
        df_copy.loc[null_indices, column_name] = np.nan
        
    return df_copy


# %%
data_columns = list(data.columns)

# %%
data_columns

# %%
data_with_null_values = random_null(data, data_columns, 10)

# %%
data_with_null_values.isnull().sum()

# %%
data_with_null_values

# %%
data_with_null_values["fueltype"][201]

# %%
null_indices_dict = {
    column: data_with_null_values.index[data_with_null_values[column].isnull()].tolist()
    for column in data_with_null_values.columns
}

# %%
null_indices_dict

# %%
data_null_places = data_with_null_values.isnull()

# %%
x = np.arange(len(data_with_null_values))  # Indices of the data
null_indices = np.where(data_null_places)[0]

# %%
null_indices

# %%
data_null_places

# %%
def iterative_Cleaning_discrete_values(data, null_place_df,columns_list, number_of_loops):
    for i in range(1,number_of_loops+1):
        for column_name in columns_list:
            data_without_null_values = data[null_place_df[column_name]== False]
            data_with_null_values = data[null_place_df[column_name] == True]
        
            x_train = data_without_null_values.drop(columns = [column_name])
            y_train = data_without_null_values[column_name]
        
            model = RandomForestClassifier()
            model.fit(x_train, y_train)
        
            x_test = data_with_null_values.drop(columns= [column_name])
        
            predicted = model.predict(x_test)
        
            data.loc[data_with_null_values.index , column_name] = predicted
    
    return data    
        
        
        
    

# %%
def iterative_Cleaning_continous_value(data, null_place_df,columns_list, number_of_loops):
    
    for i in range(1, number_of_loops +1):
        for column_name in columns_list:
            data_without_null_values = data[null_place_df[column_name]== False]
            data_with_null_values = data[null_place_df[column_name] == True]
        
            x_train = data_without_null_values.drop(columns = [column_name])
            y_train = data_without_null_values[column_name]
        
            model = LinearRegression()
            model.fit(x_train, y_train)
        
            x_test = data_with_null_values.drop(columns= [column_name])
        
            predicted = model.predict(x_test)
        
            data.loc[data_with_null_values.index , column_name] = predicted
    
    return data 

# %%
data_traditional_cleaning = data_with_null_values

# %%
# Categorical columns: Fill with mode
categorical_columns = [
    'symboling', 'fueltype', 'aspiration', 'doornumber', 'carbody', 
    'drivewheel', 'enginelocation', 'enginetype', 'cylindernumber', 
    'fuelsystem', 'company_name'
]

for col in categorical_columns:
    data_traditional_cleaning[col].fillna(data_traditional_cleaning[col].mode()[0], inplace=True)

# Numerical columns: Fill based on assumptions (mean for normal, median for skewed)
numerical_columns = {
    'wheelbase': 'mean', 'carlength': 'median', 'carwidth': 'mean',
    'carheight': 'median', 'curbweight': 'mean', 'enginesize': 'median',
    'boreratio': 'mean', 'stroke': 'median', 'compressionratio': 'mean',
    'horsepower': 'median', 'peakrpm': 'median', 'citympg': 'mean',
    'highwaympg': 'mean', 'price': 'median'
}

for col, method in numerical_columns.items():
    # Compute the fill value
    if method == 'mean':
        fill_value = data_traditional_cleaning[col].mean()
    elif method == 'median':
        fill_value = data_traditional_cleaning[col].median()
    
    # Preserve original data type
    if pd.api.types.is_integer_dtype(data_traditional_cleaning[col]):
        data_traditional_cleaning[col].fillna(int(round(fill_value)), inplace=True)
    else:
        data_traditional_cleaning[col].fillna(fill_value, inplace=True)


# %%
data_traditional_cleaning

# %%
def overalliterative(data_columns_discrete, data_columns_continuos,number_of_loops, data, null_place_df):
    for i in range(1, number_of_loops +1):
        for column_name in data_columns_continuos:
            data_without_null_values = data[null_place_df[column_name]== False]
            data_with_null_values = data[null_place_df[column_name] == True]
        
            x_train = data_without_null_values.drop(columns = [column_name])
            y_train = data_without_null_values[column_name]
        
            model = LinearRegression()
            model.fit(x_train, y_train)
        
            x_test = data_with_null_values.drop(columns= [column_name])
        
            predicted = model.predict(x_test)
        
            data.loc[data_with_null_values.index , column_name] = predicted
            
        for column_name in data_columns_discrete:
            data_without_null_values = data[null_place_df[column_name]== False]
            data_with_null_values = data[null_place_df[column_name] == True]
        
            x_train = data_without_null_values.drop(columns = [column_name])
            y_train = data_without_null_values[column_name]
        
            model = RandomForestClassifier()
            model.fit(x_train, y_train)
        
            x_test = data_with_null_values.drop(columns= [column_name])
        
            predicted = model.predict(x_test)
        
            data.loc[data_with_null_values.index , column_name] = predicted
            
    return data
        

# %%
def overalliterativedataype(data_columns_discrete, data_columns_continuos, number_of_loops, data, null_place_df):
    for i in range(1, number_of_loops + 1):
        for column_name in data_columns_continuos:
            data_without_null_values = data[null_place_df[column_name] == False]
            data_with_null_values = data[null_place_df[column_name] == True]
            
            x_train = data_without_null_values.drop(columns=[column_name])
            y_train = data_without_null_values[column_name]
            
            model = LinearRegression()
            model.fit(x_train, y_train)
            
            x_test = data_with_null_values.drop(columns=[column_name])
            predicted = model.predict(x_test)
            
            
            
            # Preserve data type
            if pd.api.types.is_integer_dtype(data[column_name]):
                predicted = np.round(predicted).astype(int)
            data.loc[data_with_null_values.index, column_name] = predicted
        
        for column_name in data_columns_discrete:
            data_without_null_values = data[null_place_df[column_name] == False]
            data_with_null_values = data[null_place_df[column_name] == True]
            
            x_train = data_without_null_values.drop(columns=[column_name])
            y_train = data_without_null_values[column_name]
            
            model = RandomForestClassifier()
            model.fit(x_train, y_train)
            
            x_test = data_with_null_values.drop(columns=[column_name])
            predicted = model.predict(x_test)
            
            
            # Preserve data type
            if pd.api.types.is_integer_dtype(data[column_name]):
                predicted = predicted.astype(int)
            data.loc[data_with_null_values.index, column_name] = predicted
            
    return data

# %%
def overalliterativedataypestorage(data_columns_discrete, data_columns_continuos, number_of_loops, data, null_place_df):
    iteration_dict = {col : [] for col in data_columns_continuos+data_columns_discrete}
    for i in range(1, number_of_loops + 1):
        for column_name in data_columns_continuos:
            data_without_null_values = data[null_place_df[column_name] == False]
            data_with_null_values = data[null_place_df[column_name] == True]
            
            x_train = data_without_null_values.drop(columns=[column_name])
            y_train = data_without_null_values[column_name]
            
            model = LinearRegression()
            model.fit(x_train, y_train)
            
            x_test = data_with_null_values.drop(columns=[column_name])
            predicted = model.predict(x_test)
            
            iteration_dict[column_name].append(predicted)
            
            
            if pd.api.types.is_integer_dtype(data[column_name]):
                predicted = np.round(predicted).astype(int)
            data.loc[data_with_null_values.index, column_name] = predicted
        
        for column_name in data_columns_discrete:
            data_without_null_values = data[null_place_df[column_name] == False]
            data_with_null_values = data[null_place_df[column_name] == True]
            
            x_train = data_without_null_values.drop(columns=[column_name])
            y_train = data_without_null_values[column_name]
            
            model = RandomForestClassifier()
            model.fit(x_train, y_train)
            
            x_test = data_with_null_values.drop(columns=[column_name])
            predicted = model.predict(x_test)
            
            iteration_dict[column_name].append(predicted)
            
            
            if pd.api.types.is_integer_dtype(data[column_name]):
                predicted = predicted.astype(int)
            data.loc[data_with_null_values.index, column_name] = predicted
            
        
            
    return data, iteration_dict

# %%
data_traditional_cleaning_missing = data_with_null_values

# %%
data_with_null_values.isnull().sum()

# %%


# Numerical columns: Fill based on assumptions (mean for normal, median for skewed)
numerical_columns = {
    'wheelbase': 'mean', 'carlength': 'median', 'carwidth': 'mean',
    'carheight': 'median', 'curbweight': 'mean', 'enginesize': 'median',
    'boreratio': 'mean', 'stroke': 'median', 'compressionratio': 'mean',
    'horsepower': 'median', 'peakrpm': 'median', 'citympg': 'mean',
    'highwaympg': 'mean', 'price': 'median'
}

for col, method in numerical_columns.items():
    # Compute the fill value
    if method == 'mean':
        fill_value = data_traditional_cleaning_missing[col].mean()
    elif method == 'median':
        fill_value = data_traditional_cleaning_missing[col].median()
    
    # Preserve original data type
    if pd.api.types.is_integer_dtype(data_traditional_cleaning_missing[col]):
        data_traditional_cleaning_missing[col].fillna(int(round(fill_value)), inplace=True)
    else:
        data_traditional_cleaning_missing[col].fillna(fill_value, inplace=True)


# %%
data_traditional_cleaning_missing.isnull().sum()

# %%
def overalliterativedataypestorage_withm_withmissingvalues(data_columns_discrete, data_columns_continuos, number_of_loops, data, null_place_df):
    iteration_dict = {col : [] for col in data_columns_continuos+data_columns_discrete}
    for i in range(1, number_of_loops + 1):
        for column_name in data_columns_discrete:
            data_without_null_values = data[null_place_df[column_name] == False]
            data_with_null_values = data[null_place_df[column_name] == True]
            
            x_train = data_without_null_values.drop(columns=[column_name])
            y_train = data_without_null_values[column_name]
            
            model = RandomForestClassifier()
            model.fit(x_train, y_train)
            
            x_test = data_with_null_values.drop(columns=[column_name])
            predicted = model.predict(x_test)
            
            iteration_dict[column_name].append(predicted)
            
            # Preserve data type
            if pd.api.types.is_integer_dtype(data[column_name]):
                predicted = predicted.astype(int)
            data.loc[data_with_null_values.index, column_name] = predicted
            
            
        for column_name in data_columns_continuos:
            data_without_null_values = data[null_place_df[column_name] == False]
            data_with_null_values = data[null_place_df[column_name] == True]
            
            x_train = data_without_null_values.drop(columns=[column_name])
            y_train = data_without_null_values[column_name]
            
            model = LinearRegression()
            model.fit(x_train, y_train)
            
            x_test = data_with_null_values.drop(columns=[column_name])
            predicted = model.predict(x_test)
            
            iteration_dict[column_name].append(predicted)
            
            # Preserve data type
            if pd.api.types.is_integer_dtype(data[column_name]):
                predicted = np.round(predicted).astype(int)
            data.loc[data_with_null_values.index, column_name] = predicted
        
        
        
            
    return data, iteration_dict

# %%
final_data_missing, iteration_data_missing = overalliterativedataypestorage_withm_withmissingvalues(discrete_columns,continuos_columns, 5, data_traditional_cleaning, data_null_places )

# %%
def overall_iterative_data_type_storage_with_m_with_missing_values(
    data_columns_discrete, data_columns_continuous, number_of_loops, data, null_place_df, pre_fill_values
):
    
    iteration_dict = {col: [] for col in data_columns_continuous + data_columns_discrete}

    for column_name in data_columns_discrete + data_columns_continuous:
        
        null_indices = null_place_df[column_name]
        pre_filled_values = pre_fill_values[column_name][null_indices]
        iteration_dict[column_name].append(pre_filled_values)  
    
    for i in range(1, number_of_loops + 1):
        for column_name in data_columns_discrete:
            data_without_null_values = data[null_place_df[column_name] == False]
            data_with_null_values = data[null_place_df[column_name] == True]
            
            x_train = data_without_null_values.drop(columns=[column_name])
            y_train = data_without_null_values[column_name]
            
            model = RandomForestClassifier()
            model.fit(x_train, y_train)
            
            x_test = data_with_null_values.drop(columns=[column_name])
            predicted = model.predict(x_test)
            
            iteration_dict[column_name].append(predicted)
            
            
            if pd.api.types.is_integer_dtype(data[column_name]):
                predicted = predicted.astype(int)
            data.loc[data_with_null_values.index, column_name] = predicted
            
        for column_name in data_columns_continuous:
            data_without_null_values = data[null_place_df[column_name] == False]
            data_with_null_values = data[null_place_df[column_name] == True]
            
            x_train = data_without_null_values.drop(columns=[column_name])
            y_train = data_without_null_values[column_name]
            
            model = LinearRegression()
            model.fit(x_train, y_train)
            
            x_test = data_with_null_values.drop(columns=[column_name])
            predicted = model.predict(x_test)
            
            iteration_dict[column_name].append(predicted)
            
            
            if pd.api.types.is_integer_dtype(data[column_name]):
                predicted = np.round(predicted).astype(int)
            data.loc[data_with_null_values.index, column_name] = predicted
        
    return data, iteration_dict


# %%
def grphical_cleaning_process(orginal_data, iterative_data,null_space_data, iterations, column):
    fig, axes = plt.subplots(nrows= iterations, ncols= 1, figsize = (10, 6* iterations))
    
    line_colors = plt.cm.viridis(np.linspace(0,1,iterations))
    
    for i, (iterativeData, color) in enumerate(zip(iterative_data, line_colors)):
        
        ax = axes[i] if iterations > 1 else axes
        
        ax.plot(null_space_data, orginal_data)
    
    

# %%
import matplotlib.pyplot as plt
import numpy as np

# Sample original data and iterative dictionary (simulating your setup)
original_data = {'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]}
iterative_dict = {
    'A': [2.5, 3.5],  # Iterative imputed values for column 'A'
    'B': [25, 35]     # Iterative imputed values for column 'B'
}
null_indices_dict = {
    'A': [1, 3],      # Null indices for column 'A'
    'B': [2, 3]       # Null indices for column 'B'
}

# Function to plot graph for a specific column
def plot_iteration_graph(column_name, original_data, iterative_dict, null_indices_dict):
    null_indices = null_indices_dict[column_name]  # Get null indices for the column
    iterative_values = iterative_dict[column_name]  # Get iterative values for the column

    # Check if sizes match
    if len(null_indices) != len(iterative_values):
        print(f"Mismatch: {len(iterative_values)} iterative values but {len(null_indices)} null indices.")
        return

    # Extract original null values for the plot
    original_null_values = [original_data[column_name][idx] for idx in null_indices]

    # Plotting
    plt.figure(figsize=(8, 6))

    # Original data points (black)
    plt.scatter(null_indices, original_null_values, color='black', label='Original Data (Null Values)', s=100)

    # Iterative values (blue)
    plt.scatter(null_indices, iterative_values, color='blue', label='Iterative Values', s=100)

    # Connect original and iterative values with a line
    for idx, (original, iterative) in enumerate(zip(original_null_values, iterative_values)):
        plt.plot([null_indices[idx], null_indices[idx]], [original, iterative], 'r--', label='Correction' if idx == 0 else "")

    plt.title(f"Iterative Cleaning for Column '{column_name}'", fontsize=14)
    plt.xlabel("Index", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.show()

# Example usage
plot_iteration_graph('A', original_data, iterative_dict, null_indices_dict)


# %%
data.dtypes

# %%
for column_name in data_columns:
    if data[column_name].dtype == "int64":
        print(column_name ,data[column_name].unique())
        print("")

# %%
discrete_columns = ["company_name","symboling", "fueltype" , "aspiration", "doornumber", "carbody", "drivewheel", "enginelocation", "enginetype", "cylindernumber", "fuelsystem"]

# %%
continuos_columns = ["horsepower","peakrpm","citympg","highwaympg","enginesize","curbweight","wheelbase", "carlength", "carwidth", "carheight", "boreratio","stroke", "compressionratio","price" ]

# %%
data_traditional_cleaning.dtypes

# %%
overalliterativedataype(discrete_columns, continuos_columns, 5, data_traditional_cleaning, data_null_places)

# %%
final_data , iteration_data = overalliterativedataypestorage(discrete_columns, continuos_columns, 5, data_traditional_cleaning, data_null_places)

# %%
final_data

# %%
iteration_data['symboling']

# %%
def plot_iteration_graph(column_name, original_data, iterative_dict, null_indices_dict):
    null_indices = null_indices_dict[column_name]  # Get null indices for the column
    iterative_values = iterative_dict[column_name]  # Get iterative values for the column

    # Check if sizes match
    if len(null_indices) != len(iterative_values):
        print(f"Mismatch: {len(iterative_values)} iterative values but {len(null_indices)} null indices.")
        return

    # Extract original null values for the plot
    original_null_values = [original_data[column_name][idx] for idx in null_indices]

    # Plotting
    plt.figure(figsize=(8, 6))

    # Original data points (black)
    plt.scatter(null_indices, original_null_values, color='black', label='Original Data (Null Values)', s=100)

    # Iterative values (blue)
    plt.scatter(null_indices, iterative_values, color='blue', label='Iterative Values', s=100)

    # Connect original and iterative values with a line
    for idx, (original, iterative) in enumerate(zip(original_null_values, iterative_values)):
        plt.plot([null_indices[idx], null_indices[idx]], [original, iterative], 'r--', label='Correction' if idx == 0 else "")

    plt.title(f"Iterative Cleaning for Column '{column_name}'", fontsize=14)
    plt.xlabel("Index", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.show()

# %%
def plot_iteration_graph(column_name, original_data, iteration_data, null_indices_dict):
    """
    Plot the original null values and iterative values (with line graphs) for a specified column.

    Args:
        column_name (str): Name of the column to visualize.
        original_data (pd.DataFrame): Original dataset with all values.
        iteration_data (dict): Dictionary where keys are column names and values are lists of lists (iterations).
        null_indices_dict (dict): Dictionary where keys are column names and values are lists of null indices.
    """
    # Step 1: Extract null indices for the given column
    null_indices = null_indices_dict.get(column_name, [])
    if not null_indices:
        print(f"No null indices found for column '{column_name}'.")
        return
    
    # Step 2: Get original values at null indices
    original_null_values = [original_data[column_name][idx] for idx in null_indices]
    
    # Step 3: Get iterative cleaning data for the column
    column_iteration_data = iteration_data.get(column_name, [])
    num_iterations = len(column_iteration_data)
    
    if num_iterations == 0:
        print(f"No iterative data available for column '{column_name}'.")
        return
    
    # Step 4: Initialize the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    x_positions = range(len(null_indices))  # X-axis positions for indices
    
    # Step 5: Plot original null values
    ax.scatter(
        x_positions, 
        original_null_values, 
        color='black', 
        label='Original Null Values (Points)', 
        s=100, 
        zorder=3
    )
    ax.plot(
        x_positions, 
        original_null_values, 
        color='black', 
        label='Original Null Values (Line)', 
        linewidth=2, 
        linestyle='-',
        zorder=3
    )
    
    # Step 6: Plot iterative values for each iteration
    colors = plt.cm.viridis(np.linspace(0, 1, num_iterations))  # Color map for iterations
    
    for i, iteration_values in enumerate(column_iteration_data):
        # Align iteration values with null indices
        if len(iteration_values) != len(null_indices):
            print(f"Mismatch in iteration {i + 1}: {len(iteration_values)} values for {len(null_indices)} null indices.")
            continue
        
        # Plot iterative values as points
        ax.scatter(
            x_positions, 
            iteration_values, 
            color=colors[i], 
            label=f'Iteration {i + 1} (Points)', 
            s=70, 
            zorder=2
        )
        
        # Plot iterative values as a line
        ax.plot(
            x_positions, 
            iteration_values, 
            color=colors[i], 
            label=f'Iteration {i + 1} (Line)', 
            linewidth=1.5, 
            linestyle='-',
            zorder=2
        )
        
        # Draw correction lines from original to iterative values
        for idx, (orig_val, iter_val) in enumerate(zip(original_null_values, iteration_values)):
            ax.plot(
                [x_positions[idx], x_positions[idx]], 
                [orig_val, iter_val], 
                color=colors[i], 
                linestyle='--', 
                linewidth=1, 
                alpha=0.7,
                label='Correction' if idx == 0 and i == 0 else ""  # Only label the first correction line
            )
    
    ax.set_title(f"Iterative Cleaning for Column '{column_name}'", fontsize=16)
    ax.set_xlabel("Null Index Position", fontsize=14)
    ax.set_ylabel("Value", fontsize=14)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True)
    plt.xticks(x_positions, labels=null_indices, rotation=45)  
    plt.tight_layout()
    plt.show()

# %%
plot_iteration_graph("wheelbase", data, iteration_data, null_indices_dict)

# %%
def plot_iteration_graph_per_iteration(column_name, original_data, iteration_data, null_indices_dict):
    """
    Plot original null values and iterative values for a specified column, with one graph per iteration.

    Args:
        column_name (str): Name of the column to visualize.
        original_data (pd.DataFrame): Original dataset with all values.
        iteration_data (dict): Dictionary where keys are column names and values are lists of lists (iterations).
        null_indices_dict (dict): Dictionary where keys are column names and values are lists of null indices.
    """
    # Step 1: Extract null indices for the given column
    null_indices = null_indices_dict.get(column_name, [])
    if not null_indices:
        print(f"No null indices found for column '{column_name}'.")
        return
    
    # Step 2: Get original values at null indices
    original_null_values = [original_data[column_name][idx] for idx in null_indices]
    
    # Step 3: Get iterative cleaning data for the column
    column_iteration_data = iteration_data.get(column_name, [])
    num_iterations = len(column_iteration_data)
    
    if num_iterations == 0:
        print(f"No iterative data available for column '{column_name}'.")
        return
    
    # Step 4: Plot for each iteration
    colors = plt.cm.viridis(np.linspace(0, 1, num_iterations))  # Color map for iterations
    
    for i, iteration_values in enumerate(column_iteration_data):
        # Align iteration values with null indices
        if len(iteration_values) != len(null_indices):
            print(f"Mismatch in iteration {i + 1}: {len(iteration_values)} values for {len(null_indices)} null indices.")
            continue
        
        # Initialize the plot for the current iteration
        fig, ax = plt.subplots(figsize=(10, 6))
        x_positions = range(len(null_indices))  # X-axis positions for indices
        
        # Plot original null values
        ax.scatter(
            x_positions, 
            original_null_values, 
            color='black', 
            label='Original Null Values (Points)', 
            s=100, 
            zorder=3
        )
        ax.plot(
            x_positions, 
            original_null_values, 
            color='black', 
            label='Original Null Values (Line)', 
            linewidth=2, 
            linestyle='-',
            zorder=3
        )
        
        # Plot iterative values for the current iteration
        ax.scatter(
            x_positions, 
            iteration_values, 
            color=colors[i], 
            label=f'Iteration {i + 1} (Points)', 
            s=70, 
            zorder=2
        )
        ax.plot(
            x_positions, 
            iteration_values, 
            color=colors[i], 
            label=f'Iteration {i + 1} (Line)', 
            linewidth=1.5, 
            linestyle='-',
            zorder=2
        )
        
        # Draw correction lines from original to iterative values
        for idx, (orig_val, iter_val) in enumerate(zip(original_null_values, iteration_values)):
            ax.plot(
                [x_positions[idx], x_positions[idx]], 
                [orig_val, iter_val], 
                color=colors[i], 
                linestyle='--', 
                linewidth=1, 
                alpha=0.7,
                label='Correction' if idx == 0 else ""  # Only label the first correction line
            )
        
        # Finalize the current plot
        ax.set_title(f"Iterative Cleaning for Column '{column_name}' - Iteration {i + 1}", fontsize=16)
        ax.set_xlabel("Null Index Position", fontsize=14)
        ax.set_ylabel("Value", fontsize=14)
        ax.legend(fontsize=12, loc='upper right')
        ax.grid(True)
        plt.xticks(x_positions, labels=null_indices, rotation=45)  # Show actual null indices on the x-axis
        plt.tight_layout()
        plt.show()


# %%
iteration_data["wheelbase"]

# %%
def plot_iteration_graph_per_iteration(column_name, original_data, iteration_data, null_indices_dict):
    """
    Plot original null values and iterative values for a specified column, with one graph per iteration,
    ensuring consistent scaling across all iterations.

    Args:
        column_name (str): Name of the column to visualize.
        original_data (pd.DataFrame): Original dataset with all values.
        iteration_data (dict): Dictionary where keys are column names and values are lists of lists (iterations).
        null_indices_dict (dict): Dictionary where keys are column names and values are lists of null indices.
    """
    # Step 1: Extract null indices for the given column
    null_indices = null_indices_dict.get(column_name, [])
    if not null_indices:
        print(f"No null indices found for column '{column_name}'.")
        return
    
    # Step 2: Get original values at null indices
    original_null_values = [original_data[column_name][idx] for idx in null_indices]
    
    # Step 3: Get iterative cleaning data for the column
    column_iteration_data = iteration_data.get(column_name, [])
    num_iterations = len(column_iteration_data)
    
    if num_iterations == 0:
        print(f"No iterative data available for column '{column_name}'.")
        return
    
    # Step 4: Determine the global y-axis range for consistent scaling
    all_values = original_null_values[:]
    for iteration_values in column_iteration_data:
        all_values.extend(iteration_values)
    y_min, y_max = min(all_values), max(all_values)
    
    # Step 5: Plot for each iteration
    colors = plt.cm.viridis(np.linspace(0, 1, num_iterations))  # Color map for iterations
    
    for i, iteration_values in enumerate(column_iteration_data):
        # Align iteration values with null indices
        if len(iteration_values) != len(null_indices):
            print(f"Mismatch in iteration {i + 1}: {len(iteration_values)} values for {len(null_indices)} null indices.")
            continue
        
        # Initialize the plot for the current iteration
        fig, ax = plt.subplots(figsize=(10, 6))
        x_positions = range(len(null_indices))  # X-axis positions for indices
        
        # Plot original null values
        ax.scatter(
            x_positions, 
            original_null_values, 
            color='black', 
            label='Original Null Values (Points)', 
            s=100, 
            zorder=3
        )
        ax.plot(
            x_positions, 
            original_null_values, 
            color='black', 
            label='Original Null Values (Line)', 
            linewidth=2, 
            linestyle='-',
            zorder=3
        )
        
        # Plot iterative values for the current iteration
        ax.scatter(
            x_positions, 
            iteration_values, 
            color=colors[i], 
            label=f'Iteration {i + 1} (Points)', 
            s=70, 
            zorder=2
        )
        ax.plot(
            x_positions, 
            iteration_values, 
            color=colors[i], 
            label=f'Iteration {i + 1} (Line)', 
            linewidth=1.5, 
            linestyle='-',
            zorder=2
        )
        
        # Draw correction lines from original to iterative values
        for idx, (orig_val, iter_val) in enumerate(zip(original_null_values, iteration_values)):
            ax.plot(
                [x_positions[idx], x_positions[idx]], 
                [orig_val, iter_val], 
                color=colors[i], 
                linestyle='--', 
                linewidth=1, 
                alpha=0.7,
                label='Correction' if idx == 0 else ""  # Only label the first correction line
            )
        
        # Finalize the current plot
        ax.set_title(f"Iterative Cleaning for Column '{column_name}' - Iteration {i + 1}", fontsize=16)
        ax.set_xlabel("Null Index Position", fontsize=14)
        ax.set_ylabel("Value", fontsize=14)
        ax.set_ylim(y_min - 1, y_max + 1)  # Ensure the same y-axis range across all plots
        ax.legend(fontsize=12, loc='upper right')
        ax.grid(True)
        plt.xticks(x_positions, labels=null_indices, rotation=45)  # Show actual null indices on the x-axis
        plt.tight_layout()
        plt.show()


# %%
data.columns

# %%
plot_iteration_graph_per_iteration("doornumber", data, iteration_data, null_indices_dict)

# %%
# Remove null indices that have been filled
filled_indices = null_indices[:len(null_indices_dict)]
null_indices_dict['A'] = null_indices[len(null_indices_dict):]


# %%
print("Null Indices:", null_indices)
print("Iterative Values:", iteration_data)
print("Length of Null Indices:", len(null_indices))
print("Length of Iterative Values:", len(iteration_data))


# %%
data.columns

# %%
plot_iteration_graph_per_iteration("aspiration", data, iteration_data_missing, null_indices_dict)

# %%


