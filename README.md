# BlackFridaySales
This Python script performs an analysis of Black Friday sales data. It includes data visualization and regression modeling.

# Prerequisites
Make sure you have the following libraries installed:

* numpy
* pandas
* matplotlib
* seaborn
* sklearn
* xgboost

You can install the required packages using the following command:
pip install numpy pandas matplotlib seaborn scikit-learn xgboost

# Usage
1. Clone the repository:
   git clone https://github.com/Dcomforter/BlackFridaySalesAnalysis.git

2. Navigate to the project directory:
   cd BlackFridaySalesAnalysis

3. Run the script:
   python black_friday_sales_analysis.py

# File Structure
"black_friday_sales_analysis.py": Contains the main Python script.
"customer.csv": Input data file in CSV format.

# Scripts Explanation
The script performs the following tasks:

1. Loading Data: Loads the Black Friday sales data from the "customer.csv" file.
2. Data Exploration:
    Displays basic information about the dataset, such as shape, data types, and missing values.
    Visualizes the distribution of the 'Purchase' column.
    Visualizes the boxplot of 'Purchase'.
    Visualizes the distribution of 'Gender'.
    Visualizes the distribution of 'Marital_Status'.
    Visualizes the distribution of 'Occupation'.
    Visualizes the distribution of 'City_Category'.
    Visualizes the distribution of 'Stay_In_Current_City_Years'.
    Visualizes the distribution of 'Age'.
    Visualizes the distribution of 'Product_Category_1', 'Product_Category_2', and 'Product_Category_3'.
    Displays a heatmap of the correlation matrix.
3. Data Preprocessing:
    Drops unnecessary columns ('User_ID' and 'Product_ID').
    Handles missing values in 'Product_Category_2' and 'Product_Category_3'.
    Encodes categorical variables.
    Splits the data into features (X) and target variable (y).
    Splits the data into training and testing sets.
4. Modeling:
    Trains a Linear Regression model.
    Evaluates the Linear Regression model.
    Trains a Decision Tree Regression model.
    Evaluates the Decision Tree Regression model.
    Trains a Random Forest Regression model.
    Evaluates the Random Forest Regression model.
    Trains an XGBoost Regression model.
    Evaluates the XGBoost Regression model.

# License
This project is licensed under the MIT License - see the LICENSE file for details.

