# Dependency
import numpy as np
import re
import pickle
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

#nltk(stopwords):- Stop words in NLTK are those little words that don't carry much meaning on their own,
#ike "the," "a," "an," or "in." When we preprocess text data, we often want to clean it up and make
#it easier for the computer to understand. One common step is to filter out these stop words because
#they don't add much value to the analysis. So, essentially, NLTK uses stop words to refer to these
#meaningless words that we want to ignore or remove during text processing.

#importing stowords.txt file
with open('stopwords.txt', 'r') as file:
    stopwords_list = [line.strip() for line in file]

stopwords_set = set(stopwords_list)


# Creating steaming function
pso = PorterStemmer()


def stemming(content):
    # Create a stemmer object
    stemmer = PorterStemmer()
    # Lowercase the content
    content = content.lower()
    # Remove punctuation and special characters
    content = re.sub(r'[^a-zA-Z\s]', '', content)
    # Split the content into words
    words = content.split()
    # Remove stopwords
    words = [word for word in words if word not in stopwords_list]
    # Stem the words
    stemmed_words = [stemmer.stem(word) for word in words]
    # Join the stemmed words back into a string
    stemmed_content = ' '.join(stemmed_words)
    return stemmed_content


# importing model
with open('Sentiment.pkl', 'rb') as file:
    model = pickle.load(file)

# importing TfidfVectorizer vocabulary file
with open('vocab.pkl','rb') as files:
    vocabulary = pickle.load(files)

vec = TfidfVectorizer(vocabulary=vocabulary)

# streamlit
st.title('Positive and Negative Sentiment Sensing Model')
st.write("About Model:-")
st.caption("It is a sentiment sensing model, analyzes tweets to classify them as either positive or negative. It "
           "uses natural language processing techniques to extract features and train a machine learning algorithm, "
           "such as a support vector machine or a neural network. The model learns patterns in the text data to make "
           "predictions about the sentiment of future tweets. This can be useful for understanding public opinion, "
           "brand sentiment, or predicting market trends based on social media activity")
st.subheader('Enter your Emotion')
a = st.text_area(" ")
result = stemming(a)
result = np.array([result])
result = vec.fit_transform(result)
result = model.predict(result)
button_clicked = st.button("Sense Sentiment")
if button_clicked:
    if result == 1:
        st.success('Positive Sentiment')
    else:
        st.error('Negative Sentiment')
