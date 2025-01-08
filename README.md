# COMP20008 Research Report
*'An Exploration into Content-Based Predictive Models and Recommendation Systems for Books'* is a research investigation completed by Alice Kjar, Annisa Chyntia Yusup and Yin-Xi Chloe for COMP20008 (Elements of Data Processing). This project utilised a number of preprocessing and machine learning techniques, written in Python to extract information from two datasets. This information was collated and expanded upon on in a formal academic report which was presented in class.

## Credits

This assignment utilised techniques from a number of sources. Code adapted from specific websites has been attributed as such. A number of techniques were also built from content taught in COMP20008 and sample code provided by teaching staff.

## Skills

This project allowed me to develop a number of new/emerging skills, including
- **Python** programming
- Data **preprocessing** (incl. text processing, imputation and discretisation)
- **Machine learning** techniques (incl. regression/correlation, k-nearest neighbours and TF-IDF vectorisation)
- Academic **report writing**
- Oral presentation and **scientific communication**
- Team-based **soft skills** (incl. collaboration, communication and delegation)



## Datasets

BX-Books.csv (Books dataset)
- **ISBN**: International Standard Book Number, a unique identifier for books.
- **Book-Title**: Title of the book.
- **Book-Author**: Author(s) of the book.
- **Year-Of-Publication**: Year when the book was published.
- **Book-Publisher**: Publisher of the book.
- Total Rows: 18,185

BX-Ratings.csv (Ratings dataset)
- **User-ID**: Unique identifier for users.
- **ISBN**: International Standard Book Number, a unique identifier for books.
- **Book-Rating**: Rating given by users to books.
- Total Rows: 204,146

Python Source Files
--------------------------------------
1. title_author_preprocessing.py
    - Preprocesses the text-formatted title and author in the Books dataset
    -  Generates book language based on title

2. preprocessing_cont.py
    - Preprocesses publisher and publication year
    - Merges the dataset with Ratings dataset to obtain average rating of books prior to discretising them
    - Generates authors' and publishers' count of books

3. data_exploration.py
    - Retrieves summary statistics
    - Plots the distribution of numerical variables using histograms
    - Generates a word cloud to visualize the disribution of words in title

4. knn.py
    - Computes correlation between book features and ratings
    - Predicts book rating by implementing k-Nearest Neighbours machine learning algorithm
    - Evaluates the suitability of the model on a test dataset
5. book_recommendation.py
    - Vectorises the content of books based on title and author using TF-IDF
    - Computes the similarity score by utilising Cosine Similarity metrics to build a recommendation system


## Running the Program

Some external packages must be installed prior to running the programs:
    
    pip install ftfy langdetect wordcloud

--------------------------------------

Prior to the analysis, perform preprocessing on book features by running:
   
    python title_author_preprocessing.py
    python preprocessing_cont.py

This places Preprocessed_Books.csv in the directory which contains the initial dataset with several additional columns. This dataset is used for all analysis in the study.

This program may require a long runtime to complete this part due to the large size of Books dataset and the language detector utilised in the codes. A fully preprocessed version of this dataset has been provided in the repository, for simplicity.

--------------------------------------
In order to do basic data exploration on our book features, run:

    python data_exploration.py

Some png files of data visualisation and printed detailed statistics will be obtained as outputs.

--------------------------------------
There are two main machine learning techniques used in this study. The first analysis is predicting the average rating of the books based on number of books written by author, number of books published by publisher and year of publication.
To perform this, run:

    python knn.py

As the result, you will get information about book features, correlation and the accuracy of prediction from performing k-NN Technique.

The second analysis is generating book recommendations based on title and author as a reflection of the content of books. To perform this, run:

    book_recommendation.py

The recommendation function will extract a list of book recommendations in List-Recommendations.csv based on one specific book.