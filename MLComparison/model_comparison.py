import argparse
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Regression Models
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Metrics
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score


# ==================================================
# Command Line Arguments
# ==================================================

parser = argparse.ArgumentParser(description="ML Model Comparison Tool")

parser.add_argument("dataset_path", type=str,
                    help="Path to CSV dataset")

parser.add_argument("task_type", type=str,
                    choices=["classification", "regression"],
                    help="Type of ML task")

args = parser.parse_args()


# ==================================================
# Check Dataset File
# ==================================================

if not os.path.exists(args.dataset_path):
    print("Error: Dataset file not found.")
    exit()


# ==================================================
# Load Dataset
# ==================================================

try:
    df = pd.read_csv(args.dataset_path)

except Exception as e:
    print("Error loading dataset:", e)
    exit()


# ==================================================
# Features and Target
# ==================================================

df = df.dropna(subset=[df.columns[-1]])

X = df.iloc[:, :-1].copy()
y = df.iloc[:, -1].copy()


# ==================================================
# Handle Missing Values
# ==================================================

# Numerical columns
num_cols = X.select_dtypes(include=np.number).columns

for col in num_cols:
    X[col] = X[col].fillna(X[col].mean())

# Categorical columns
cat_cols = X.select_dtypes(include=["object", "string"]).columns

for col in cat_cols:
    X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else "Missing")


# ==================================================
# Drop High Cardinality Columns
# ==================================================

for col in cat_cols:
    if X[col].nunique() > 20:
        X = X.drop(columns=[col])


# ==================================================
# One-Hot Encoding
# ==================================================

X = pd.get_dummies(X, drop_first=True)


# ==================================================
# Encode Target (Classification)
# ==================================================

if args.task_type == "classification":

    if y.dtype == "object":
        le = LabelEncoder()
        y = le.fit_transform(y)


# ==================================================
# Final NaN Check
# ==================================================

X = X.fillna(0)


# ==================================================
# Train/Test Split
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==================================================
# Feature Scaling
# ==================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ==================================================
# Models
# ==================================================

if args.task_type == "classification":

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier()
    }

else:

    models = {
        "Linear Regression": LinearRegression(),
        "SVR": SVR(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor()
    }


# ==================================================
# Train and Evaluate
# ==================================================

results = []

print("\nTraining Models...\n")

for name, model in models.items():

    try:
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        # Classification
        if args.task_type == "classification":

            accuracy = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions, average="weighted")

            results.append([name, accuracy, f1])

        # Regression
        else:

            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

            results.append([name, mse, r2])

        print(f"{name} completed.")

    except Exception as e:
        print(f"Error with {name}: {e}")


# ==================================================
# Results Table
# ==================================================

if not results:
    print("Error: All models failed to train.")
    exit()

if args.task_type == "classification":

    results_df = pd.DataFrame(
        results,
        columns=["Model", "Accuracy", "F1 Score"]
    )

else:

    results_df = pd.DataFrame(
        results,
        columns=["Model", "MSE", "R2 Score"]
    )


print("\n=============== RESULTS ===============\n")
print(results_df)


# ==================================================
# Best Model (Bonus)
# ==================================================

if args.task_type == "classification":

    best_model = results_df.loc[
        results_df["Accuracy"].idxmax()
    ]

    print("\nBest Model:", best_model["Model"])

else:

    best_model = results_df.loc[
        results_df["R2 Score"].idxmax()
    ]

    print("\nBest Model:", best_model["Model"])


# ==================================================
# Save Report
# ==================================================

report_file = "model_report.txt"

with open(report_file, "w") as file:

    file.write("ML MODEL COMPARISON REPORT\n")
    file.write("=" * 40 + "\n\n")

    file.write(f"Dataset: {args.dataset_path}\n")
    file.write(f"Task Type: {args.task_type}\n\n")

    file.write(results_df.to_string(index=False))

    file.write("\n\n")

    file.write(f"Best Model: {best_model['Model']}\n")

print(f"\nReport saved as '{report_file}'")