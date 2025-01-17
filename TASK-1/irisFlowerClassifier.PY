# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Load the dataset
file_path = 'TASK-1\Iris.csv'
data = pd.read_csv(file_path)

# Explore the dataset
print("First 5 rows of the dataset:")
print(data.head())
print("\nDataset Info:")
print(data.info())
print("\nSummary Statistics:")
print(data.describe())

# Drop the 'Id' column if present
if 'Id' in data.columns:
    data = data.drop(columns=['Id'])

# Check for missing values
print("\nMissing values in the dataset:")
missing_values = data.isnull().sum()
print(missing_values)
if missing_values.any():
    print("Handling missing values...")
    data.fillna(data.mean(), inplace=True)

# Visualize the data
sns.pairplot(data, hue='Species', diag_kind='kde', markers=['o', 's', 'D'])
plt.show()

# Prepare the data for modeling
X = data.drop(columns=['Species'])
y = data['Species']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
from sklearn.model_selection import GridSearchCV
params = {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid=params, cv=3)
model = grid_search.fit(X_train, y_train).best_estimator_

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

# Feature importance
feature_importances = model.feature_importances_
feature_names = X.columns
plt.figure(figsize=(8, 6))
sorted_indices = feature_importances.argsort()
plt.barh([feature_names[i] for i in sorted_indices], feature_importances[sorted_indices], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.show()
