import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'Unemployment in India.csv'
data = pd.read_csv(file_path)

# Display basic information about the dataset
print("Dataset Information:")
data.info()
print("\nFirst 5 rows of the dataset:")
print(data.head())

# Check for missing values
print("\nMissing values in each column:")
print(data.isnull().sum())

# Basic statistics of the dataset
print("\nStatistical Summary:")
print(data.describe())

# Rename columns for easier access (if necessary)
data.columns = [col.strip().lower().replace(' ', '_') for col in data.columns]

# Data preprocessing (Convert 'date' column to datetime and set it as index if exists)
if 'date' in data.columns:
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)

# Exploratory Data Analysis (EDA)
plt.figure(figsize=(12, 6))
plt.title('Distribution of Unemployment Rate')
plt.xlabel('Unemployment Rate')
plt.ylabel('Frequency')
sns.histplot(data['unemployment_rate'], bins=30, kde=True)
plt.show()

# Time-series analysis (Unemployment rate over time)
plt.figure(figsize=(14, 7))
sns.lineplot(data=data, x=data.index, y='unemployment_rate', marker='o', color='red')
plt.title('Unemployment Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.grid()
plt.show()

# Analyze the impact of Covid-19 on unemployment
covid_start = '2020-03-01'
covid_end = '2021-03-01'
covid_data = data.loc[covid_start:covid_end]

plt.figure(figsize=(12, 6))
sns.barplot(x=covid_data.index.strftime('%Y-%m'), y='unemployment_rate', data=covid_data, color='orange')
plt.title('Unemployment Rate During Covid-19')
plt.xlabel('Month')
plt.ylabel('Unemployment Rate (%)')
plt.xticks(rotation=45)
plt.grid()
plt.show()

# Analyze unemployment rate by region (if regional data is available)
if 'region' in data.columns:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='region', y='unemployment_rate', data=data, palette='Set2')
    plt.title('Unemployment Rate by Region')
    plt.xlabel('Region')
    plt.ylabel('Unemployment Rate')
    plt.xticks(rotation=45)
    plt.show()

# Correlation analysis (if numerical columns are available)
plt.figure(figsize=(10, 6))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Save cleaned dataset (if modifications were made)
data.to_csv('cleaned_unemployment_data.csv', index=False)
print("Cleaned dataset saved as 'cleaned_unemployment_data.csv'.")
