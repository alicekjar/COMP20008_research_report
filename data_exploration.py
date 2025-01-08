"""
data_exploration.py

Written by Yin-Xi Chloe Lin, Alice Kjar and Annisa Chyntia Yusup for COMP20008 Assignment 2
Graphing of subplots adapted from https://matplotlib.org/stable/gallery/statistics/histogram_multihist.html
"""

from matplotlib import pyplot as plt
import pandas as pd
from wordcloud import WordCloud

############################################################

""" prints descriptive statistics """
def get_stats():
    print(pub_count.describe())
    print(aut_count.describe())
    print(books['Year-Of-Publication'].describe())
    print(books['Average-Rating'].describe())

""" plots histogram on axes """
def plot_hist(axis, data, x_label, title, log, colour):
    axis.hist(data, bins=20, color=colour)
    if (log):
        axis.set_yscale("log")
    axis.set_xlabel(x_label)
    axis.set_ylabel('Frequency')
    axis.set_title(title)

############################################################

ratings = pd.read_csv('BX-Ratings.csv')
books = pd.read_csv('Preprocessed_Books.csv')

pub_count = books['Book-Publisher'].value_counts()
aut_count = books['Book-Author'].value_counts()

get_stats()


# plot distributions of three features
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(14, 7))

plot_hist(ax1, books["Year-Of-Publication"], 'Year Of Publication',
           'Distribution of Publication Years', False, '#460B5E')
plot_hist(ax2, aut_count, 'Number of books written by authors',
           'Distribution of Books Written by Author', True, '#37B878')
plot_hist(ax3, pub_count, 'Number of books published by publishers',
           'Distribution of Books Published by Publisher', True, '#EAE51A')

fig.suptitle("Distributions of Book Features across Dataset", size="x-large")

plt.savefig("feature_distributions.png")
plt.clf()


# plot the distribution of average ratings
plt.figure(figsize=(6.4, 4.8))
plt.hist(books['Average-Rating'], bins=20, color='#434086')
plt.xlabel("Average Rating")
plt.ylabel("Frequency")
plt.title("Distribution of Average Ratings over Dataset")
plt.savefig("rating_distributions.png")
plt.clf()

# make word cloud
books['Book-Title-Clean'] = books['Book-Title-Clean'].astype(str)
text = " ".join(title for title in books['Book-Title-Clean'])
word_cloud1 = WordCloud(collocations = False, background_color = 'white',
                        width = 2500, height = 1500 , random_state=1).generate(text)
plt.imshow(word_cloud1, interpolation='bilinear')
plt.axis("off")
plt.show()
plt.savefig("WordCloud.png")







