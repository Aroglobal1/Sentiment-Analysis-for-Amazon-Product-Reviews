# Sentiment-Analysis-for-Amazon-Product-Reviews

This type of analysis is based on emotions, determining whether it is positive, negative or neutral. Here, it's been drawn to Amazon product reviews, whereby they are being analysed to evaluate buyers' sentiments(feelings) about the products in question.

### Project Overview

The objective of this project is to delve into the world of natural language processing and sentiment analysis, working with a dataset of product reviews to analyze and classify the sentiment of each review as positive, negative, or neutral. 


### Data Source
The dataset used for this project was gotten from Kaggle. " .csv" includes text reviews and associated ratings, which will be used to perform sentiment analysis.


### Tools

- Excel
- Python libraries Pandas, etc.
- Machine Learning Library: Scikit-learn, Spacy, etc.
- Jupyter Notebook for documentation


### Steps
These are the various steps involved in the sentiment analysis of Amazon product reviews dataset

#### Data Preparation/Exploratory Data Analysis
Converted the txt file to CSV file
Then read the CSV file in jupyter notebook
-drop irrelevant columns 


#### Data Preprocessing
Converted all text string then to lowercase to ensure consistency 
Remove the html tags using regex function
Tokenize the text
Also, remove stop words
Apply the preprocessing

Drop duplicate texts
Then, stemming which involves reducing the texts to their base/root form, using PorterStemmer.


#### Text Vectorization
Using CountVectorizer...


#### Sentiment labeling
Under the sentiment column, it was observed that 21741 reviews fall under "1" representing positive while the remaining 4999 reviews fall under "0" representing negative sentiment.
Then, a bar chart was plotted to display the distribution of the sentiment labels.


#### Model Building

#### Model Evaluation

### Insights and Recommendations

