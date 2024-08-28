

# Sentiment Analysis on Twitter Data

## Overview

This script performs sentiment analysis on Twitter data using the `TextBlob` library. It includes steps for data loading, preprocessing, sentiment analysis, and visualization. The script performs the following tasks:

1. **Data Loading**: Reads Twitter data from a CSV file.
2. **Data Preprocessing**: Handles missing values and inspects data.
3. **Sentiment Analysis**: Analyzes sentiment using `TextBlob` and categorizes it.
4. **Visualization**: Creates visualizations for sentiment distribution and trends over time.

## Requirements

To run this script, you need Python 3.x and the following libraries:

- `pandas` for data manipulation.
- `matplotlib` for plotting.
- `seaborn` for data visualization.
- `textblob` for sentiment analysis.
- `numpy` for numerical operations.

You can install these libraries using `pip`:

```bash
pip install pandas matplotlib seaborn textblob numpy
```
## File Path
Ensure your CSV file is located at /home/rguktong/Desktop/twitter_training.csv. Update the file path in the script if your CSV file is located elsewhere.

## Script Details
**1. Import Libraries**
Import the necessary libraries for data manipulation, visualization, and sentiment analysis:

```bash
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import numpy as np
```
**2. Load Data**
Load the dataset from the specified CSV file and display basic information about the DataFrame:
```bash
data = pd.read_csv('/home/rguktong/Desktop/twitter_training.csv')
print(data)
print(data.shape)
print(data.info())
```
**3. Data Preprocessing**
### Check for Missing Values:
   Print the number of missing values in each column:

```bash
print(data.isnull().sum())
```
Handle Missing Values: Drop rows with missing values:
```bash
data = data.dropna()
print(data.columns)
```
**4. Sentiment Analysis**
#### Define Sentiment Function: 
Create a function to compute sentiment polarity using TextBlob:

```bash
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity
```
Apply Sentiment Function: Apply the sentiment function to the tweet text column:

```bash
data['im getting on borderlands and i will murder you all ,'] = data['Positive'].apply(get_sentiment)
```
Categorize Sentiment: Categorize the sentiment into Positive, Negative, or Neutral:
```bash
data['sentiment_category'] = np.where(data['im getting on borderlands and i will murder you all ,'] > 0, 'Positive', 
                                       np.where(data['im getting on borderlands and i will murder you all ,'] < 0, 'Negative', 'Neutral'))
sentiment_counts = data['sentiment_category'].value_counts()
```
**5. Visualization**
#### Sentiment Distribution Pie Chart:
Create a pie chart to visualize the distribution of sentiment categories:

```bash
colors = ['#FF9999', '#66B3FF', '#99FF99']  # Red for Negative, Blue for Neutral, Green for Positive
plt.figure(figsize=(8, 8))
plt.pie(sentiment_counts, labels=sentiment_counts.index, colors=colors, startangle=90, 
        autopct='%1.1f%%', shadow=True, explode=(0.1, 0, 0))  # explode the first slice
plt.title('Sentiment Distribution')
plt.axis('equal')
plt.show()
```
**Daily Average Sentiment Over Time:** Plot the daily average sentiment over time:

```bash
data['2401'] = pd.to_datetime(data['2401'])
daily_sentiment = data.groupby(data['2401'].dt.date)['im getting on borderlands and i will murder you all ,'].mean().reset_index()

plt.figure(figsize=(12, 6))
plt.plot(daily_sentiment['2401'], daily_sentiment['im getting on borderlands and i will murder you all ,'], marker='o', color='purple')
plt.title('Daily Average Sentiment Over Time')
plt.xlabel('Date')
plt.ylabel('Average Sentiment')
plt.xticks(rotation=45)
plt.show()
```
## Troubleshooting
**File Not Found Error:**
Ensure the CSV file exists at the specified path and that the path is correctly referenced in the script.
**Column Name Issues:**
Verify that the column names used in the script match those in your dataset. Adjust the script if column names differ.
**Plot Issues:**
Ensure that the data used for plotting is correctly formatted and contains the expected values.
## License
This script is licensed under the MIT License. See the LICENSE file for details.



### Explanation of the `README.md` Structure

1. **Overview**: Describes the purpose and steps of the script.
2. **Requirements**: Lists necessary Python libraries and provides installation instructions.
3. **File Path**: Notes the importance of updating the file path to the dataset.
4. **Script Details**: Breaks down the script into sections with explanations and code examples:
   - Importing libraries.
   - Loading and inspecting data.
   - Preprocessing data and handling missing values.
   - Performing sentiment analysis and categorization.
   - Creating visualizations for sentiment distribution and trends.
5. **Troubleshooting**: Provides solutions for common issues related to file paths, column names, and plotting.
6. **License**: Specifies the license under which the script is distributed.

This `README.md` file should help users understand and effectively use your sentiment analysis script, including how to troubleshoot potential issues.
