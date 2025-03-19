import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
file_path = "D:\\Machine Learning\\287_t_audioexcel.xlsx"
df = pd.read_excel(file_path)

# Convert categorical columns using Label Encoding
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])  # Convert categorical to numerical
    label_encoders[col] = le  # Store encoders for future reference

# Fix target label range (normalize classes to 0,1,2,...)
df.iloc[:, -1] -= df.iloc[:, -1].min()

# Splitting features and target
X = df.iloc[:, :-1]  # Features
y = df.iloc[:, -1]   # Target variable

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42, k_neighbors=1)
X, y = smote.fit_resample(X, y)

# Split data (Stratified to balance classes)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Define classifiers and hyperparameters for RandomizedSearchCV
classifiers = {
    "RandomForest": (RandomForestClassifier(), {"n_estimators": [10, 50, 100], "max_depth": [None, 10, 20]}),
    "AdaBoost": (AdaBoostClassifier(), {"n_estimators": [50, 100, 200]}),
    "SVM": (SVC(), {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}),
    "DecisionTree": (DecisionTreeClassifier(), {"max_depth": [None, 5, 10]}),
    "XGBoost": (XGBClassifier(), {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]}),
    "NaiveBayes": (GaussianNB(), {}),  # No hyperparameters for tuning
    "MLP": (MLPClassifier(), {"hidden_layer_sizes": [(50,), (100,), (50, 50)]}),
}

# Train models and compare performance
results = []
for name, (model, params) in classifiers.items():
    # Use StratifiedKFold to handle imbalanced data
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
   
    search = RandomizedSearchCV(model, params, n_iter=3, cv=skf, n_jobs=-1, random_state=42) if params else model
    search.fit(X_train, y_train)
   
    best_model = search.best_estimator_ if params else search
    y_pred = best_model.predict(X_test)
   
    acc = accuracy_score(y_test, y_pred)
    results.append((name, acc))
    print(f"\n{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

# Convert results to DataFrame for better comparison
results_df = pd.DataFrame(results, columns=["Model", "Accuracy"])
print("\nComparison of Classifier Performance:")
print(results_df.sort_values(by="Accuracy", ascending=False))


