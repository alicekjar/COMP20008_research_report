"""
knn.py

Uses k-Nearest-Neighbours on dataset to predict book rating

Written by Alice Kjar for COMP20008 Assignment 2
Adapted from code provided by COMP20008 teaching team for workshops
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split

############################################################

""" 
    Creates a regression model for a given feature and its effect on the average
    rating of a book. 
"""
def find_corr(feature, x, y):
    
    # find and plot regression model
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m*x + b, color='red')
    m = round(m, 2)
    b = round(b, 2)
    plt.legend([f'y_pred = {m}x + {b}'])
    
    # plot individual values
    plt.scatter(x, y, alpha=0.2, s=20)
    
    feature = feature.replace("-", " ")
    plt.suptitle(f"Effect of {feature} on Average Rating")
    plt.xlabel(feature)
    plt.ylabel("Average Rating")
    
    # find correlation
    corr = round(np.corrcoef(x, y)[0][1], 4)
    plt.title(f"Pearson Correlation = {corr}", size = 'medium')

    plt.savefig(feature + ".png")
    plt.clf()

############################################################
              

# open dataset
books = pd.read_csv("Preprocessed_Books.csv")

print(books.groupby('Binned-Rating').agg({'Average-Rating': ['min', 'max']}))

X = books[['Publisher-Count',"Author-Count", "Year-Of-Publication"]]
y = books[['Average-Rating','Binned-Rating']]

# split dataset into test, train and validate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 25)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.25, random_state = 1)

# create correlation graphs for each of the 3 features
for feature in ('Publisher-Count',"Author-Count", "Year-Of-Publication"):
    find_corr(feature, X_train[feature], y_train['Average-Rating'])

y_train = y_train['Binned-Rating']
y_valid = y_valid['Binned-Rating']
y_test = y_test['Binned-Rating']
              
max_k = 0
max_acc = 0
# find accuracy for each k = 1,3,5 NN using train and validate
for k in (1, 3, 5, 7, 9):
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit to the train dataset
    knn.fit(X_train, y_train)

    accuracy = knn.score(X_valid, y_valid)
    if accuracy > max_acc:
        max_k = k
        max_acc = accuracy

print(f"k-value with highest accuracy: {max_k}")
              
knn = KNeighborsClassifier(n_neighbors = max_k)

# merge train and validate
X_train = pd.concat([X_train, X_valid])
y_train = pd.concat([y_train, y_valid])


# Fit to the train dataset
knn.fit(X_train, y_train)

# Note that we're calculating the accuracy score for the test data
y_pred = knn.predict(X_test)

print('Accuracy:', round(accuracy_score(y_test, y_pred),4))      
print('Recall:', round(recall_score(y_test, y_pred, average='weighted'),4))
print('Precision:', round(precision_score(y_test, y_pred, average='weighted'),4))
print('F1:', round(f1_score(y_test, y_pred,average='weighted'),4))


# cm = confusion matrix (variable name)
cm = confusion_matrix(y_test, # test data
                      y_pred, # predictions
                      labels=['Low', 'High'] # class labels from the knn model
                     )

disp = ConfusionMatrixDisplay(confusion_matrix=cm, # pass through the created confusion matrix
                              display_labels=['Low', 'High'] # class labels from the knn model 
                             )
disp.plot()

plt.title("Confusion Matrix for k-NN Analysis of Book Rating")
plt.savefig("Confusion_Matrix.png")
