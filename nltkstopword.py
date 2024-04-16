#Saving nltk stopword in a stopword.txt file
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

english_stopwords = set(stopwords.words('english'))

with open('stopwords.txt', 'w') as file:
    for word in english_stopwords:
        file.write(word + '\n')

