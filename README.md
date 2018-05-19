# Python-ID3-Decision-Tree
Python Implementation of ID3 decision tree for creating a tree and classifying stuff with it

# The Data
I used a couple of publicly availible datasets on UCI's Machine Learning Repository. I split each of the data sets I used into random training and test sets with a changable percentage of train to test. 

# Creating/Training The Tree
The tree is created by running a training set through the ID3 algorithm that I implemented.

# Using the Tree
The tree is used by traversing the tree using the test set instances

# Accuracy
After each traversal of a test data instance in the tree, its predicted classification is compared to its actual classification to see weather or not the algorithm correctly predicted its class. For an entire test set, I sum the number correct instances and divide by the number of total test set instances to get its accuracy. The accuracy can change a lot more with smaller data sets (like tic.data and car.data). With larger datasets (like mush.data) the accuracy is much much better and variant.

# Running the Accuracy Test Many Times to decrease variance
To decrease variance for smaller data sets, I run the test many (like 1000) times and get the average accuracy of that. For tic.data and car.data datasets, I found the accuracy to be a less variant, hovering around ~83% and ~88% respectively. For mush.data, a large dataset, I ran the accuracy test 20 times and would nearly get 100% each time.
