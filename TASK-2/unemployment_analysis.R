# Load necessary libraries
library(ggplot2)
library(dplyr)
library(readr)
library(tidyr)
library(ggcorrplot)

# Load the dataset
file_path <- 'Unemployment in India.csv'
data <- read_csv(file_path)

# Data preprocessing
data <- data %>%
  mutate(date = as.Date(date)) %>%
  rename(unemployment_rate = `Unemployment Rate`, region = `Region`)

# Histogram: Distribution of Unemployment Rate
ggplot(data, aes(x = unemployment_rate)) +
  geom_histogram(bins = 30, fill = 'blue', alpha = 0.7) +
  labs(title = 'Distribution of Unemployment Rate', x = 'Unemployment Rate', y = 'Frequency') +
  theme_minimal()

# Line Plot: Unemployment Rate Over Time
ggplot(data, aes(x = date, y = unemployment_rate)) +
  geom_line(color = 'red') +
  geom_point() +
  labs(title = 'Unemployment Rate Over Time', x = 'Date', y = 'Unemployment Rate (%)') +
  theme_minimal()

# Bar Plot: Unemployment Rate During Covid-19
covid_data <- data %>%
  filter(date >= '2020-03-01' & date <= '2021-03-01')

ggplot(covid_data, aes(x = format(date, '%Y-%m'), y = unemployment_rate)) +
  geom_bar(stat = 'identity', fill = 'orange') +
  labs(title = 'Unemployment Rate During Covid-19', x = 'Month', y = 'Unemployment Rate (%)') +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Box Plot: Unemployment Rate by Region
if ("region" %in% colnames(data)) {
  ggplot(data, aes(x = region, y = unemployment_rate)) +
    geom_boxplot(fill = 'lightgreen') +
    labs(title = 'Unemployment Rate by Region', x = 'Region', y = 'Unemployment Rate') +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

# Correlation Matrix
correlation_matrix <- cor(data %>% select_if(is.numeric))
ggcorrplot(correlation_matrix, lab = TRUE, title = 'Correlation Matrix', colors = c('blue', 'white', 'red'))

# Save cleaned dataset (if modifications were made)
write_csv(data, 'cleaned_unemployment_data.csv')
