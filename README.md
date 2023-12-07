# Python Machine Learning Project: Spam Classifier

## Overview
This Python script is designed to classify emails as spam or not spam using machine learning techniques. The script uses the Spambase dataset, applying several classifiers and cross-validation to evaluate performance.

## Setup
1. **Python Version**: Ensure you have Python installed on your machine. This script is written in Python 3.
2. **Dependencies**: Install required libraries using pip:
   ```bash
   pip install pandas scikit-learn numpy
   ```
3. **Dataset**: The script expects two files from the Spambase dataset: `spambase.names` and `spambase.data`. Please download these files and adjust the file paths in the script accordingly.

## Features
- **Data Preprocessing**: Reads feature names from `spambase.names` and loads `spambase.data` with these feature names.
- **Model Training**: Utilizes three classifiers - Random Forest, Naive Bayes, and Gradient Boosting.
- **Cross-Validation**: Implements Stratified K-Fold cross-validation to evaluate model performance.
- **Performance Metrics**: Computes accuracy, F1-score, and training time for each classifier.
- **Statistical Tests**: Applies the Friedman test and Nemenyi post-hoc test to statistically compare classifier performances.
