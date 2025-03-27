import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter

file_path = 'all-crops-state-wise.csv'
df = pd.read_csv(file_path)

# Display the head of the dataframe to understand its structure
print(df.head())

# Extract major crops and create a list of all crops
major_crops = df['Major Crops'].str.split(',', expand=True).stack().str.strip().tolist()

# Count frequency of each crop (case-insensitive) and create a dictionary
crop_counter = Counter()
for crop in major_crops:
    # Standardize crop names by converting to title case
    standardized_crop = crop.title()
    crop_counter[standardized_crop] += 1

# Create a word cloud with the frequency data
wordcloud = WordCloud(
    width=1000, 
    height=500, 
    background_color='white',
    max_words=100,
    normalize_plurals=False
).generate_from_frequencies(crop_counter)

# Display the word cloud
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Major Crops in India', fontsize=22)
plt.axis('off')
plt.tight_layout()
plt.show()