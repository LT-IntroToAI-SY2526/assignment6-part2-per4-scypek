"""
Assignment 6 Part 2: House Price Prediction (Multivariable Regression)

This assignment predicts house prices using MULTIPLE features.
Complete all the functions below following the in-class car price example.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


def load_and_explore_data(filename):
    """
    Load the house price data and explore it
    
    Args:
        filename: name of the CSV file to load
    
    Returns:
        pandas DataFrame containing the data
    """
    # Load the CSV file using pandas
    data = pd.read_csv(filename)

    # Print basic exploration info
    print("=== House Price Data ===")
    print("\nFirst 5 rows:")
    print(data.head())

    print(f"\nDataset shape: {data.shape[0]} rows, {data.shape[1]} columns")

    print("\nBasic statistics:")
    print(data.describe())

    print(f"\nColumn names: {list(data.columns)}")

    return data


def visualize_features(data):
    """
    Create 4 scatter plots (one for each feature vs Price)
    
    Args:
        data: pandas DataFrame with features and Price
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('House Features vs Price', fontsize=16, fontweight='bold')

    # Plot 1: SquareFeet vs Price
    axes[0, 0].scatter(data['SquareFeet'], data['Price'], color='blue', alpha=0.6)
    axes[0, 0].set_xlabel('SquareFeet')
    axes[0, 0].set_ylabel('Price ($)')
    axes[0, 0].set_title('SquareFeet vs Price')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Bedrooms vs Price
    axes[0, 1].scatter(data['Bedrooms'], data['Price'], color='green', alpha=0.6)
    axes[0, 1].set_xlabel('Bedrooms')
    axes[0, 1].set_ylabel('Price ($)')
    axes[0, 1].set_title('Bedrooms vs Price')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Bathrooms vs Price
    axes[1, 0].scatter(data['Bathrooms'], data['Price'], color='red', alpha=0.6)
    axes[1, 0].set_xlabel('Bathrooms')
    axes[1, 0].set_ylabel('Price ($)')
    axes[1, 0].set_title('Bathrooms vs Price')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Age vs Price
    axes[1, 1].scatter(data['Age'], data['Price'], color='orange', alpha=0.6)
    axes[1, 1].set_xlabel('Age (years)')
    axes[1, 1].set_ylabel('Price ($)')
    axes[1, 1].set_title('Age vs Price')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('feature_plots.png', dpi=300, bbox_inches='tight')
    print("\n✓ Feature plots saved as 'feature_plots.png'")
    plt.show()


def prepare_features(data):
    """
    Separate features (X) from target (y)
    
    Args:
        data: pandas DataFrame with all columns
    
    Returns:
        X - DataFrame with feature columns
        y - Series with target column
    """
    feature_columns = ['SquareFeet', 'Bedrooms', 'Bathrooms', 'Age']
    X = data[feature_columns]
    y = data['Price']

    print(f"\n=== Feature Preparation ===")
    print(f"Features (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")
    print(f"\nFeature columns: {list(X.columns)}")

    return X, y


def split_data(X, y):
    """
    Split data into training and testing sets
    
    Args:
        X: features DataFrame
        y: target Series
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"\n=== Data Split ===")
    print(f"Training set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, feature_names):
    """
    Train a multivariable linear regression model
    
    Args:
        X_train: training features (scaled)
        y_train: training target values
        feature_names: list of feature column names
    
    Returns:
        trained LinearRegression model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)

    print(f"\n=== Model Training Complete ===")
    print(f"Intercept: ${model.intercept_:.2f}")
    print(f"\nCoefficients:")
    for name, coef in zip(feature_names, model.coef_):
        print(f"  {name}: {coef:.2f}")

    # Build readable equation
    equation = "Price = "
    for i, (name, coef) in enumerate(zip(feature_names, model.coef_)):
        sign = " + " if coef >= 0 and i > 0 else ""
        equation += f"{sign}{coef:.2f}×{name}"
    equation += f" + {model.intercept_:.2f}"
    print(f"\nEquation:")
    print(equation)

    return model


def evaluate_model(model, X_test, y_test, feature_names):
    """
    Evaluate the model's performance
    
    Args:
        model: trained model
        X_test: testing features (scaled)
        y_test: testing target values
        feature_names: list of feature names
    
    Returns:
        predictions array
    """
    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    print(f"\n=== Model Performance ===")
    print(f"R² Score: {r2:.4f}")
    print(f"  → Model explains {r2*100:.2f}% of price variation")

    print(f"\nRoot Mean Squared Error: ${rmse:.2f}")
    print(f"  → On average, predictions are off by ${rmse:.2f}")

    # Feature importance (absolute value of coefficients)
    print(f"\n=== Feature Importance ===")
    feature_importance = list(zip(feature_names, np.abs(model.coef_)))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    for i, (name, importance) in enumerate(feature_importance, 1):
        print(f"{i}. {name}: {importance:.2f}")

    return predictions


def compare_predictions(y_test, predictions, num_examples=5):
    """
    Show side-by-side comparison of actual vs predicted prices
    
    Args:
        y_test: actual prices
        predictions: predicted prices
        num_examples: number of examples to show
    """
    print(f"\n=== Prediction Examples ===")
    print(f"{'Actual Price':<15} {'Predicted Price':<18} {'Error':<12} {'% Error'}")
    print("-" * 60)

    for i in range(min(num_examples, len(y_test))):
        actual = y_test.iloc[i]
        predicted = predictions[i]
        error = actual - predicted
        pct_error = (abs(error) / actual) * 100

        print(f"${actual:>13.2f}   ${predicted:>13.2f}   ${error:>10.2f}   {pct_error:>6.2f}%")


def make_prediction(model, sqft, bedrooms, bathrooms, age):
    """
    Make a prediction for a specific house
    
    Args:
        model: trained LinearRegression model
        sqft: square footage
        bedrooms: number of bedrooms
        bathrooms: number of bathrooms
        age: age of house in years
    
    Returns:
        predicted price
    """
    feature_columns = ['SquareFeet', 'Bedrooms', 'Bathrooms', 'Age']
    house = pd.DataFrame([[sqft, bedrooms, bathrooms, age]], columns=feature_columns)
    predicted_price = model.predict(house)[0]

    print(f"\n=== New Prediction ===")
    print(f"House specs: {sqft} sqft, {bedrooms} beds, {bathrooms} baths, {age} years")
    print(f"Predicted price: ${predicted_price:,.2f}")

    return predicted_price


if __name__ == "__main__":
    print("=" * 70)
    print("HOUSE PRICE PREDICTION - YOUR ASSIGNMENT")
    print("=" * 70)
    
    # Step 1: Load and explore
    data = load_and_explore_data('house_prices.csv')

    # Step 2: Visualize features
    visualize_features(data)

    # Step 3: Prepare features
    X, y = prepare_features(data)

    # Step 4: Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 5: Train model
    model = train_model(X_train, y_train, X.columns)

    # Step 6: Evaluate model
    predictions = evaluate_model(model, X_test, y_test, X.columns)

    # Step 7: Compare predictions
    compare_predictions(y_test.reset_index(drop=True), predictions, num_examples=10)

    # Step 8: Make a new prediction
    # Example: 3000 sqft, 6 bedrooms, 4 bathrooms, 5 years old
    make_prediction(model, 3000, 6, 4, 5)
    
    print("\n" + "=" * 70)
    print("✓ Assignment complete! Check your saved plots.")
    print("Don't forget to complete a6_part2_writeup.md!")
    print("=" * 70)