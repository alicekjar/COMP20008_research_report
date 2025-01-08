"""
preprocessing_cont.py

Further preprocesses dataset

Written by Alice Kjar for COMP20008 Assignment 2
"""

import pandas as pd
import numpy as np
import re
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer


# Stopwords identified from most frequent words in list of all publishers
STOPWORDS = ("books",
            "publishers",
            "publisher",
            "publishing",
            "pub",
            "company",
            "group",
            "trade",
            "ltd",
            " inc",
            "and",
            )

# Can be changed depending on how many bins we want
YEAR_BINS = 2
AUTH_BINS = 3
PUBL_BINS = 3
RATE_BINS = 2

############################################################

""" Removes full stops after intials, punctuation and excess whitespace """
def clean_text(s):
    s = re.sub(r'[\.~]', ' ', s)
    s = re.sub(r'[,&]', '', s)
    return re.sub(r'\s+', ' ', s)

""" Removes meaningless words from text """
def remove_stopwords(s):
    s = re.sub("|".join(STOPWORDS), '', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

""" Changes numeric binning to labelled categories """
def name(n):
    if n == 0:
        return 'Low'
    return 'High'

############################################################

books = pd.read_csv("Pre-Processed-Title-Author.csv")

# Text processing (casefolding, noise & stopword removal) on publisher
books['Book-Publisher'] = books['Book-Publisher'].str.lower()
books['Book-Publisher'] = books['Book-Publisher'].apply(clean_text)
books['Book-Publisher'] = books['Book-Publisher'].apply(remove_stopwords)

# Identify implausible year values
books.loc[books['Year-Of-Publication'] < 1920, 'Year-Of-Publication'] = np.nan
books.loc[books['Year-Of-Publication'] > 2024, 'Year-Of-Publication'] = np.nan

# Impute using median (preferencial over mean due to skew of years)
imp_med = SimpleImputer(missing_values=np.nan, strategy='median')
books['Year-Of-Publication'] = imp_med.fit_transform(books['Year-Of-Publication'].values.reshape(-1, 1)).flatten()

ratings = pd.read_csv("BX-Ratings.csv")

# Find average rating and number of ratings for each book
ave_ratings = ratings.groupby(ratings['ISBN'])['Book-Rating'].mean()
num_ratings = ratings.groupby(ratings['ISBN'])['Book-Rating'].count()
books = pd.merge(books, ave_ratings, on='ISBN', how='inner')
books = pd.merge(books, num_ratings, on='ISBN', how='inner')
books = books.rename(columns = {'Book-Rating_x':'Average-Rating', 'Book-Rating_y': 'Review-Count'})

# Find the number of books written by each author
author_freq = books.groupby(books['Book-Author']).size()
books['Author-Count'] = books['Book-Author'].map(author_freq)

# Find the number of books published by each publisher
publisher_freq = books.groupby(books['Book-Publisher']).size()
books['Publisher-Count'] = books['Book-Publisher'].map(publisher_freq)

binning_r = KBinsDiscretizer(n_bins=RATE_BINS, encode='ordinal', strategy='quantile')
books['Binned-Rating'] = binning_r.fit_transform(books[['Average-Rating']]).astype(int)
books['Binned-Rating']=books['Binned-Rating'].apply(name)

books.to_csv('Preprocessed_Books.csv', index=False)

