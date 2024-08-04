
# Sentiment Sensing Model Documentation
![Sentiment Sensing Model Documentation](sentiment-analysis.jpg)
--------------------------------------------------------------------------------

## Introduction 

**The Sentiment Sensing Model** is designed to analyze text data and determine the sentiment expressed within it. The model can classify text as positive, negative.

## Setup and Installation

### Prerequisites

Clone Repository

```bash
Git clone https://github.com/147Ayush/Sentiment_Sensing_Model.git
```

Ensure you have Python and the necessary libraries installed. You can install the required libraries using pip:
```bash
pip install -r requirements.txt
```

## Model Architecture

### Components

- **Text Preprocessing**: Tokenization, stop-word removal, stemming/lemmatization.
- **Feature Extraction**: Techniques like TF-IDF, word embeddings.
- **Model Type**:  Logistic Regression machine learning model  
- **Streamlit Cloud**: For Deployment of App

### Preprocessing

- **Tokenization**: Splitting text into individual tokens (words).
- **Lowercasing**: Converting all text to lowercase to maintain uniformity.
- **Stop-word Removal**: Removing common words that do not contribute to sentiment (e.g., 'the', 'is').
- **Stemming/Lemmatization**: Reducing words to their base or root form.

### Feature Extraction

- **TF-IDF**: Term Frequency-Inverse Document Frequency to weigh the importance of words.
- **Word Embeddings**: Representing words in continuous vector space.

## Training

### Dataset

- **Source**: Kaggle (Twitter sentiment analysis dataset).
- **Classes**: Sentiment categories (positive(1), negative(0)).

### Training Process

- **Data Splitting**: Splitting the dataset into training, validation, and test sets.
- **Model Training**: Training the model using the training dataset.
- **Validation**: Tuning hyperparameters using the validation dataset.
- **Evaluation**: Assessing model performance using the test dataset.

## Usage

### Input

- **Text Data**: Raw text input for sentiment analysis.

### Output

- **Sentiment Score**: Probability scores for each sentiment class.
- **Sentiment Label**: Predicted sentiment (positive, negative).

## Deployment Link
https://147ayush-positive-and-negative-sentiment-sensing-mo-main-vs7rke.streamlit.app/


