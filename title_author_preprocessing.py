"""
data_exploration.py

Preprocessing required to prepate author names and titles for recommender system

Written by Annisa Chyntia Yusup for COMP20008 Assignment 2
"""
import pandas as pd
import string
import nltk
import re
import ftfy
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
from langdetect import detect_langs
from langdetect.detector import LangDetectException

OTHER_STOPWORDS = ("book", "novel", "paperback", "story", "edition")

############################################################

""" Clean text with question mark """
def clean_quest(text):
    text = re.sub(r'[?]', '', text)
    return text

""" Converts non ASCII characters """
def replace_nonascii(text):
    return ftfy.fix_text(text)

""" Removes another punctuations, 's, and excess whitespace """
def clean_all(text):
    text = re.sub(r'[)(:.!"@$&,]', ' ', text)
    text = re.sub(r'[-\/]', ' ', text)
    text = re.sub(r"[']s", "", text)
    return re.sub(r'\s+', ' ', text)

""" Removes english stopwords """
def remove_stopwords(text):
    nltk_stopwords = stopwords.words('english')
    pattern = r'\b(?:{})\b'.format('|'.join(map(re.escape, nltk_stopwords)))
    text= re.sub(pattern, '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

""" Removes other stopwords """
def remove_other_stopwords(text):
    pattern = r'\b(?:{})\b'.format('|'.join(map(re.escape, OTHER_STOPWORDS)))
    text= re.sub(pattern, '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

""" Lemmatizes words in sentence """
def lemmatize_sentence(sentence):
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(sentence)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    lemmatized_sentence = ' '.join(lemmatized_words)
    return lemmatized_sentence

"""Detect the language with probability >= 80%"""
def detect_language(text):
    try:
        if text.strip() and len(text) >= 3:
            lang_probabilities = detect_langs(text)
            selected_languages = [item.lang for item in lang_probabilities if item.prob >= 0.8]
            return selected_languages[0] if selected_languages else "unknown"
        else:
            return "unknown"
    except LangDetectException:
        return "unknown"

############################################################

#Import File
books= pd.read_csv('BX-Books.csv')

#Replace non ascii characters
for feature in ('Book-Title', 'Book-Author', 'Book-Publisher'):
    books[feature] = books[feature].apply(clean_quest)
    books[feature] = books[feature].apply(replace_nonascii)
    
#Cleans book Title
books['Book-Title-Clean'] = books['Book-Title'].str.lower()
books['Book-Title-Clean'] = books['Book-Title-Clean'].apply(clean_all)

#Detects language of title
books['Book-Language'] = books['Book-Title-Clean'].apply(detect_language)

#Removes meaningless words and lemmatizes sentence
books['Book-Title-Clean'] = books['Book-Title-Clean'].apply(remove_stopwords)
books['Book-Title-Clean'] = books['Book-Title-Clean'].apply(lemmatize_sentence)
books['Book-Title-Clean'] = books['Book-Title-Clean'].apply(remove_other_stopwords)

#Cleans Author
books['Book-Author'] = books['Book-Author'].str.lower()
books['Book-Author'] = books['Book-Author'].apply(clean_all)

books.to_csv('Pre-Processed-Title-Author.csv', index=False)
