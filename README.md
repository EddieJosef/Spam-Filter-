**Part 1: Bayes Classifier - Pre-Processing**

This repository contains code for a spam filter model that utilizes the open-source spam library SpamSssasin. This README file provides an overview of the functionality of the first part of the project, which focuses on pre-processing steps before training the classifier.

The code in the `Bayes Classifier - Pre-Processing.ipynb` notebook performs the following tasks:

1. Imports necessary libraries for data processing and visualization, such as `pandas`, `matplotlib`, `nltk`, `BeautifulSoup`, `wordcloud`, `PIL`, and `numpy`.
2. Defines constants and file paths used throughout the pre-processing steps.
3. Reads email files and extracts email bodies.
4. Generates DataFrame from spam and non-spam email directories.
5. Performs data cleaning by checking for missing values and removing empty emails.
6. Drops system file entries from the DataFrame.
7. Adds document IDs to track emails in the dataset.
8. Saves the pre-processed data to a JSON file.
9. Visualizes the number of spam and non-spam messages using pie charts.
10. Performs natural language processing tasks, including text pre-processing, tokenization, removing stop words, stemming, and removing punctuation.
11. Removes HTML tags from email messages using the BeautifulSoup library.
12. Defines functions for cleaning and tokenizing messages.
13. Applies the cleaning and tokenization functions to all messages in the dataset.
14. Creates subsets of spam and non-spam messages based on category.
15. Generates word clouds for visualization.

Please refer to the notebook for detailed code implementation and comments explaining each step. The remaining parts of the README will cover other aspects of the spam filter model.

## Part 2: Bays classifier - Training

This part of the code focuses on training the Naive Bayes classifier using the provided dataset. The following steps are performed:

1. Reading and loading the training data from text files: The code imports the necessary libraries, such as pandas and numpy, and defines the file paths for the training data and test data. These paths are stored in the constants `TRAINING_DATA_FILE` and `TEST_DATA_FILE`. The training data is loaded into a NumPy array using `np.loadtxt()` function.

2. Creating an empty DataFrame: An empty DataFrame is created with columns representing document ID, category, and token occurrences. The column names are defined as `column_names`, which includes 'DOC_ID', 'CATEGORY', and a range of tokens from 0 to `VOCAB_SIZE`. The DataFrame is initialized with zeros using `fillna()` method.

3. Building the full matrix from the sparse matrix: The function `make_full_matrix()` is defined, which takes the sparse matrix, number of words in the vocabulary (`VOCAB_SIZE`), and indices for document ID, word ID, category, and frequency. It creates a full matrix DataFrame by mapping the values from the sparse matrix to the appropriate positions in the full matrix. The resulting DataFrame, `full_train_data`, is returned.

4. Calculating the probability of spam: The code calculates the probability of a message being spam (`prob_spam`) by dividing the number of spam messages (`full_train_data.CATEGORY.sum()`) by the total number of messages (`full_train_data.CATEGORY.size`).

5. Total number of words/tokens: The code creates a subset of the email lengths for spam messages (`spam_lengths`) and calculates the total number of words in spam emails (`spam_wc`). Similarly, it creates a subset for non-spam messages (`ham_lengths`) and calculates the total number of words in non-spam emails (`nonspam_wc`). The sum of `spam_wc` and `nonspam_wc` should be equal to the total number of words/tokens (`total_wc`).

6. Summing the tokens occurring in spam: The code creates a subset of the full training features DataFrame for spam messages (`train_spam_tokens`). It then sums the token occurrences for each token across all spam messages, including a smoothing value of 1 (`summed_spam_tokens`).

By executing this part of the code, you will load and preprocess the training data, calculate the probability of spam, and calculate the token occurrences for spam messages. These steps are essential for training the Naive Bayes classifier in the subsequent parts.
**Part 3: Bayes Classifier - Testing**

## Code Functionality

This code performs spam classification using a trained model. It uses the following libraries: `pandas`, `numpy`, `matplotlib.pyplot`, and `seaborn`. 

The code begins by importing the required libraries and setting up the necessary file paths.

Next, it loads the test data from files into numpy arrays: `X_test` contains the test features, and `y_test` contains the corresponding target labels.

Then, the code loads the probability values from files: `prob_token_spam`, `prob_token_ham`, and `prob_all_tokens`.

Using these probabilities and the test features, the code calculates the logarithm of the joint probability of each email being spam or non-spam. These values are stored in `joint_log_spam` and `joint_log_ham`.

Based on these joint log probabilities, the code makes predictions by comparing `joint_log_spam` and `joint_log_ham` and assigning a label of 1 (spam) if `joint_log_spam` is greater, and 0 (non-spam) otherwise. The predictions are stored in the `predictions` array.

The code then plots a scatter plot of `joint_log_spam` against `joint_log_ham` using matplotlib, with a diagonal line indicating the decision boundary. The plot is displayed twice with different axis limits.

Using seaborn, the code creates a summary dataframe that includes `joint_log_spam`, `joint_log_ham`, and the true labels `y_test`. It then plots a scatter plot with `joint_log_spam` on the x-axis, `joint_log_ham` on the y-axis, and different markers for spam and non-spam emails. The decision boundary is also plotted.

Finally, the code calculates various evaluation metrics such as true positives, false positives, false negatives, recall score, precision, and F1 score. These metrics are printed to the console.

Please note that the code snippet provided may be incomplete, and the complete functionality of the code may extend beyond what is shown here.

**Part 4: Model Deployment**

Part 4: Functionality of the Code

The code provided implements a spam email filter using scikit-learn's machine learning library. Here is a brief overview of its functionality:

1. Importing Required Libraries: The necessary libraries such as pandas, numpy, and scikit-learn modules are imported.

2. Loading the Data: The code reads the email data from a JSON file, 'email-text-data.json', using the pandas library.

3. Sorting the Data: The data is sorted to ensure consistent indexing and ordering.

4. Text Vectorization: The CountVectorizer class from scikit-learn is used to convert the text data into numerical feature vectors. The stop words (common words like "and," "the," etc.) are removed during the vectorization process.

5. Feature Extraction: The fit_transform() method of the vectorizer is applied to the email messages, generating a sparse matrix representation of the features.

6. Vocabulary Extraction: The vocabulary (unique words) learned by the vectorizer is stored in the 'vectorizer.vocabulary_' attribute.

7. Machine Learning Model: A Naive Bayes classifier (MultinomialNB) is used to train the spam filter. The data is split into training and testing sets using the train_test_split() function.

8. Model Evaluation: Various metrics like precision, recall, F1-score, etc., are computed using the precision_recall_fscore_support(), recall_score(), precision_score(), and f1_score() functions to evaluate the performance of the spam filter.

This code provides a basic implementation of a spam email filter using the Naive Bayes algorithm and vectorization techniques. It can serve as a starting point for further improvements and customization.
