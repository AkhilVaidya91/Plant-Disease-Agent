# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the dataset
file_path = 'all-crops-state-wise.csv'
df = pd.read_csv(file_path, encoding='ascii')

# Displaying the head of the dataframe to understand its structure
print(df.head())
print(df.columns)

# Renaming the columns
# The first column is 'State/Union Territory' and the second is 'Major Crops'
df.columns = ['State', 'Major Crops']

# Now, we will create a new column that counts the number of crops in 'Major Crops'
df['Crop Count'] = df['Major Crops'].apply(lambda x: len(x.split(',')))

# Getting the top 10 states with the most crop varieties
top_states = df.nlargest(10, 'Crop Count')

# Checking the structure of the DataFrame again to identify any issues
print(df.info())

plt.figure(figsize=(12, 6))
sns.barplot(x='Crop Count', y='State', data=top_states, palette='viridis')
plt.title('Top 10 States by Number of Crop Varieties')
plt.xlabel('Number of Crop Varieties')
plt.ylabel('States')
plt.show()  

# 2. Pie chart showing the distribution of crop varieties across all states
plt.figure(figsize=(10, 10))
df['Major Crops'].str.split(',').explode().value_counts().head(10).plot.pie(autopct='%1.1f%%', startangle=90)
plt.title('Top 10 Crop Varieties Distribution')
plt.ylabel('')
plt.show()

# 3. Count plot of the top 10 most common crops
plt.figure(figsize=(12, 6))
sns.countplot(y=df['Major Crops'].str.split(',').explode().value_counts().head(10).index, palette='magma')
plt.title('Top 10 Most Common Crops')
plt.xlabel('Count')
plt.ylabel('Crops')
plt.show()

# 4. Box plot showing the distribution of crop counts by state
plt.figure(figsize=(12, 6))
sns.boxplot(x='Crop Count', data=df, palette='coolwarm')
plt.title('Distribution of Crop Counts by State')
plt.xlabel('Crop Count')
plt.ylabel('States')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# 5. Count of states by number of crops (categorizing states into bins)
plt.figure(figsize=(8, 12))
df['Crop Count'].value_counts(bins=8).sort_index().plot(kind='bar', color='skyblue')
plt.title('Count of States by Number of Crop Varieties')
plt.xlabel('Number of Crop Varieties')
plt.ylabel('Count of States')
plt.show()  

print('Visualizations completed without the heatmap.')