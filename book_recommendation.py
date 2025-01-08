"""
book_recommendation.py

Written by Annisa Chyntia Yusup for COMP20008 Assignment 2
Recommendation system adapted from https://thecleverprogrammer.com/2023/10/30/book-recommendation-system-with-python/
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

############################################################

""" 
    Gets ten books recommendation based on cosine similarity 
"""
def get_recommendation(title, cosine_sim):
    index = books[books['Book-Title'] == title].index[0]

    sim_scores = list(enumerate(cosine_sim[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [(i, score) for i, score in sim_scores if (0.5 < score < 0.9)]
    sim_scores = sim_scores[0:10]

    book_indices = [i[0] for i in sim_scores]

    # Create a DataFrame containing book information and similarity scores
    similar_books = books.iloc[book_indices].copy()
    similar_books['Similarity Score'] = [score for i, score in sim_scores] 
    book_indices = [i[0] for i in sim_scores]
    similar_books = similar_books[['ISBN', 'Book-Title', 'Book-Author', 'Book-Language', 'Similarity Score']]
    return similar_books.to_csv('List-Recommendations.csv', index=False)

############################################################

books = pd.read_csv('Pre-Processed-Title-Author.csv')

# Combines title and Author
books['book_combine'] = books['Book-Title-Clean'] + ' ' + books['Book-Author']

## Drop the duplicates based on combined title+author
books = books.drop_duplicates(subset=['book_combine'])

## Re-index data
books.reset_index(drop=True, inplace=True)


#Get the vector representation
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(books['Book-Title'])

# Calculates the cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Gets other books recommendation for specific title
get_recommendation('Harry Potter and the Sorcerer\'s Stone (Harry Potter (Paperback))', cosine_sim)

