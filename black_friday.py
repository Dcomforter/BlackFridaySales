import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

class BlackFridaySalesAnalysis:
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file)
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def summarize_data(self):
        """Displays basic information about the dataset."""
        print("Data Shape:", self.data.shape)
        print("\nData Info:")
        print(self.data.info())
        print("\nMissing Values:\n", self.data.isnull().sum())
        print("\nUnique Values:\n", self.data.nunique())
        print("\nSummary Statistics:\n", self.data.describe())

    def visualize_purchase_distribution(self):
        sns.displot(self.data["Purchase"], color='r')
        plt.title("Purchase Distribution")
        plt.show()

    def visualize_purchase_boxplot(self):
        sns.boxplot(x=self.data["Purchase"])
        plt.title("Boxplot of Purchase")
        plt.show()

    def visualize_categorical_distributions(self):
        """Visualizes various categorical distributions and purchase correlations."""
        plt.figure(figsize=(15, 6))
        sns.countplot(x=self.data['Gender'])
        plt.title('Gender Distribution')
        plt.show()

        # Age Distribution
        plt.figure(figsize=(8, 5))
        sns.countplot(data=self.data, x='Age', order=sorted(self.data['Age'].unique()))
        plt.title("Age Distribution")
        plt.xlabel("Age Groups")
        plt.show()

        # Marital Status Distribution
        plt.figure(figsize=(15, 6))
        sns.countplot(x=self.data['Marital_Status'])
        plt.title('Marital Status Distribution')
        plt.show()

        plt.figure(figsize=(14, 5))
        sns.countplot(data=self.data, x='Occupation')
        plt.title("Occupation Distribution")
        plt.xlabel("Occupation Code")
        plt.show()

        plt.figure(figsize=(6, 4))
        sns.countplot(data=self.data, x='City_Category')
        plt.title("City Category Distribution")
        plt.xlabel("City Category (A, B, C)")
        plt.show()

        plt.figure(figsize=(8, 5))
        sns.countplot(data=self.data, x='Stay_In_Current_City_Years')
        plt.title("Stay In Current City Years Distribution")
        plt.xlabel("Years in Current City")
        plt.show()

        # Add more as needed...

        # Product Category 1 Distribution
        plt.figure(figsize=(14, 5))
        sns.countplot(data=self.data, x='Product_Category_1')
        plt.title("Product Category 1 Distribution")
        plt.xlabel("Product Category 1")
        plt.show()

        # Product Category 2 Distribution
        plt.figure(figsize=(14, 5))
        sns.countplot(data=self.data, x='Product_Category_2')
        plt.title("Product Category 2 Distribution")
        plt.xlabel("Product Category 2")
        plt.show()

        # Product Category 3 Distribution
        plt.figure(figsize=(14, 5))
        sns.countplot(data=self.data, x='Product_Category_3')
        plt.title("Product Category 3 Distribution")
        plt.xlabel("Product Category 3")
        plt.show()

    def visualize_heatmap(self):
        # Select only numeric columns
        numeric_data = self.data.select_dtypes(include=[np.number])
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_data.corr(), annot=True)
        plt.title("Feature Correlation Heatmap")
        plt.show()

    def preprocess_data(self):
        """Preprocesses the data by encoding, filling missing values, and splitting."""
        self.data = self.data.drop(["User_ID", "Product_ID"], axis=1)
        self.data['Product_Category_2'].fillna(0, inplace=True)
        self.data['Product_Category_3'].fillna(0, inplace=True)

        # Encoding categorical variables
        label_encoder = LabelEncoder()
        self.data['Gender'] = label_encoder.fit_transform(self.data['Gender'])
        self.data['Age'] = label_encoder.fit_transform(self.data['Age'])
        self.data['City_Category'] = label_encoder.fit_transform(self.data['City_Category'])
        self.data = pd.get_dummies(self.data, columns=['Stay_In_Current_City_Years'], drop_first=True)

        # Separating features and target variable
        self.X = self.data.drop("Purchase", axis=1)
        self.y = self.data["Purchase"]

        # Splitting data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=123)

    def fit_and_evaluate(self, model):
        """Fits and evaluates a model, returning performance metrics."""
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        metrics = {
            'MAE': mean_absolute_error(self.y_test, y_pred),
            'MSE': mean_squared_error(self.y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(self.y_test, y_pred)),
            'R-squared': r2_score(self.y_test, y_pred)
        }
        return metrics

def main():
    data_file = "sales_record.csv"
    analysis = BlackFridaySalesAnalysis(data_file)

    # Display data summary
    analysis.summarize_data()

    # Visualize data distributions
    analysis.visualize_purchase_distribution()
    analysis.visualize_purchase_boxplot()
    analysis.visualize_categorical_distributions()
    analysis.visualize_heatmap()

    # Preprocess the data
    analysis.preprocess_data()

    # Train and evaluate models
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=0),
        "Random Forest": RandomForestRegressor(random_state=0),
        "XGBoost": XGBRegressor(learning_rate=1.0, max_depth=6, min_child_weight=40, seed=0)
    }

    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        metrics = analysis.fit_and_evaluate(model)
        print(f"{model_name} Metrics:\n", metrics)

if __name__ == "__main__":
    main()
