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

#### Data Preparation
The initial step was to install and import all the required python libraries/modules that are to be used in the course of the analysis, then the next step involved converting the dataset from a text file format to a CSV format. This conversion facilitated easier manipulation and analysis within Jupyter Notebook.

```python
pip install textblob
pip install nltk
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from textblob import TextBlob
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
```

Once the dataset was in CSV format, it was imported into Jupyter Notebook. The next phase involved cleaning the data by removing irrelevant columns that were not necessary for the sentiment analysis. Specifically, columns such as marketplace, customer_id, review_id, product_id, product_parent, product_title, product_category, star_rating, helpful_votes, total_votes, vine, verified_purchase, review_date were dropped, as they did not contribute to the sentiment evaluation process.

```python
# load csv file

df = pd.read_csv(r'C:\Users\USER\Documents\COURSES\Flit Apprenticeship\Projects\3\Amazon Product Review.csv')
```

```python
# drop columns
text_df = df.drop(['marketplace', 'customer_id', 'review_id', 'product_id',
       'product_parent', 'product_title', 'product_category', 'star_rating',
       'helpful_votes', 'total_votes', 'vine', 'verified_purchase',
     'review_date'], axis=1)
text_df.head()
```
![image](https://github.com/Aroglobal1/Sentiment-Analysis-for-Amazon-Product-Reviews/assets/148555924/04e85454-2c3b-4dc9-b46d-1b5506e24d22)

This preparatory work ensured that the dataset was streamlined and ready for detailed exploratory data analysis and subsequent sentiment analysis tasks.


#### Exploratory Data Analysis
#### Data Preprocessing
As part of the Exploratory Data Analysis (EDA), the next step involved comprehensive data preprocessing to prepare the text data for sentiment analysis.

**Text Normalization:** All text strings were converted to lowercase to ensure consistency and uniformity in the analysis.

**HTML Tag Removal:** Any HTML tags present in the review column were removed using regular expressions (regex) to clean the text and focus solely on the content of the reviews.

**Tokenization:** The text data was tokenized, breaking down the strings into individual words or tokens. This step is crucial for subsequent text analysis tasks.

**Stop Words Removal:** Common stop words (e.g., "and," "the," "is") were removed to eliminate noise and focus on the significant words that contribute to the sentiment.
**Duplicate Texts:** Texts that are duplicated were dropped.
**Stemming:** This was done to reduce the texts to their base form, using PorterStemmer.

```python
def preprocess_text(text):
    if pd.isnull(text):  # Check if the text is NaN
        return ''  # Return an empty string if the text is missing
    text = str(text) # Convert the text to string to handle non-string values
    text = text.lower()
    text_clean = re.sub(r"(<[^>]+>(?:\s+|\n|\r)*)", " ", text, flags=re.MULTILINE) # Remove HTML tags
    text_tokenize = word_tokenize(text_clean)
    stop_words = set(stopwords.words('english'))
    filtered_text = [word for word in text_tokenize if word not in stop_words]
    return " ".join(filtered_text)
df['preprocessed_review_body'] = df['review_body'].apply(preprocess_text)
```

```python
# Drop duplicates based on preprocessed review bodies
df.drop_duplicates(subset=['preprocessed_review_body'], inplace=True)

# Stemming
stemmer = PorterStemmer()
def stemming(data):
    text = [stemmer.stem(word) for word in data]
    return text
df['stemmed_review_body'] = df['preprocessed_review_body'].apply(lambda x: stemming(x)
df['preprocessed_review_body'].head()
```
![image](https://github.com/Aroglobal1/Sentiment-Analysis-for-Amazon-Product-Reviews/assets/148555924/4e3c6d40-47fe-482a-9973-cfcdb5642116)


These preprocessing steps were essential to prepare the data for effective sentiment analysis, ensuring that the text was clean, consistent, and ready for further analytical procedures.


#### Text Vectorization
Following the data preprocessing phase, the next step in the sentiment analysis process was text vectorization. This step involves transforming the cleaned text data into numerical representations that machine learning algorithms can process.

The CountVectorizer tool from the scikit-learn library was used for this purpose. It helps in converting the text data into a matrix of token counts. By using CountVectorizer, the textual data was effectively transformed into a structured numerical format, making it suitable for training and applying machine learning models for sentiment data analysis.

```python
# Text Vectorization
vectorizer = CountVectorizer(max_features = 1000)
data = vectorizer.fit_transform(df['preprocessed_review_body'])
```


#### Sentiment Labeling
In the sentiment labeling step, the sentiment column in the dataset was utilized to categorize the reviews as either positive or negative. Specifically, the sentiment values were represented as follows:

- "1" indicated positive sentiment
- "0" indicated negative sentiment

```python
# Sentiment labelling
sentiment_mapping = {
    0: "Negative",
    1: "Positive"
}

def label_sentiment(sentiment_value):
    return sentiment_mapping[sentiment_value]
df['sentiment_label'] = df['sentiment'].apply(label_sentiment)
```

```python
# Plotting a Bar Chart
sentiment_count = df['sentiment_label'].value_counts()

sentiment_count.plot(kind = 'bar', color = ['green', 'red'])
plt.xlabel('Sentiment Label')
plt.ylabel('Count')
plt.title('Distribution of Sentiment Labels')
plt.xticks(rotation=0)
plt.show()
```

Upon analyzing the dataset, it was observed that out of the total reviews:

- 21,741 reviews were labelled as positive (represented by "1")
- 4,999 reviews were labelled as negative (represented by "0")
![image](https://github.com/Aroglobal1/Sentiment-Analysis-for-Amazon-Product-Reviews/assets/148555924/59f117bc-902e-48f5-8ef1-c6deee23e024)

To visualize the distribution of sentiment labels, bar and pie charts were plotted. This chart provided a clear graphical representation of the number of positive and negative reviews, with a significantly higher number of positive reviews compared to the negative ones.
![image](https://github.com/Aroglobal1/Sentiment-Analysis-for-Amazon-Product-Reviews/assets/148555924/71789f89-bb98-452c-be77-1f72e930ebd3)


#### Model Building
In the model building phase, one of the machine learning algorithms, which is Logistic Regression was employed in predicting sentiment based on the preprocessed text data, due to its simplicity and effectiveness in binary classification tasks.
Data Splitting: The dataset was divided into training and testing sets to evaluate the performance of the model, using a split ratio of 80/20.
Training: The Logistic Regression model was trained on the training dataset. Hyperparameters such as the regularization parameter (C) were tuned using grid search to find the best configuration for optimal performance.

#### Model Evaluation
In the model evaluation phase, the performance of the trained model was assessed using the testing dataset by generating a confusion matrix to understand its performance.

```python
# Model Building & Evaluation

X_train, X_test, Y_train, Y_test = train_test_split(data, df['sentiment_label'], test_size = 0.2, random_state = 42) # split data  

model = LogisticRegression()
model.fit(X_train, Y_train) # train the data on a logistic regression model

model_pred = model.predict(X_test)
model_accuracy = model.score(X_test, Y_test)
print("Accuracy: {:.2f}%".format(model_accuracy*100))
```
![image](https://github.com/Aroglobal1/Sentiment-Analysis-for-Amazon-Product-Reviews/assets/148555924/5c003ee9-d1fa-44bf-819d-15390574fc6a)

The confusion matrix includes the following components:
**True Positives (TP)**: 
**True Negatives (TN)**: 
**False Positives (FP)**: 
**False Negatives (FN)**: 

```python
# Confusion matrix
conf_matrix = confusion_matrix(Y_test, model_pred)
print("\n")
print(classification_report(Y_test, model_pred))

sns.heatmap(conf_matrix, annot = True,  fmt = 'd', cmap = 'Blues', 
           xticklabels = ['Negative', 'Positive'],
           yticklabels = ['Negative', 'Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
```
![image](https://github.com/Aroglobal1/Sentiment-Analysis-for-Amazon-Product-Reviews/assets/148555924/4506d9b4-c728-4ff7-ae71-1a37f6633ba6)

The confusion matrix was visualized using a heatmap to provide an intuitive understanding of the classification results, highlighting both the strengths and areas for potential improvement.

**Performance Metrics** (derived from the confusion matrix):

Accuracy: (TP + TN) / (TP + TN + FP + FN) = _____
Precision: TP / (TP + FP) = _____
Recall: TP / (TP + FN) = _____
F1-Score: 2 \times (Precision \times Recall) / (Precision + Recall) = ____

### Recommendations

