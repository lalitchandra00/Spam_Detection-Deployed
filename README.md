# ML-Spam_Detection
This project implements a spam email classifier using Python, scikit-learn, and Logistic Regression. It processes a dataset of SMS/Email messages, cleans and prepares the text data, and then trains a machine learning model to classify messages as spam or ham (non-spam).
<br>
## Project Overview
This is a Machine Learning project that detects whether a given message is spam or ham (non-spam) using Logistic Regression in a Jupyter Notebook. The model is trained on a dataset of SMS messages and classifies incoming text based on learned patterns.
<br>
## Dataset
The dataset contains thousands of SMS messages labeled as either spam or ham. Each message is processed to remove duplicates and handle missing values. Labels are then encoded (ham → 1, spam → 0) before feeding into the model.
<br>
## Model & Accuracy
The model is implemented using Logistic Regression, a supervised learning algorithm for binary classification. Text data is transformed into numerical form using TF-IDF Vectorization, and the model achieves high accuracy in classifying messages correctly.
<br>
## Technologies Used
Python

Jupyter Notebook

Pandas (for data handling)

Scikit-learn (for model building, vectorization, and evaluation)
