#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
# Path to the .names file
spambase_names_path = 'Maskin Learning/A2/spambase.names' # replace to your filepath

# Function to read and extract feature names from the .names file
def extract_feature_names(file_path):
    with open(file_path, 'r') as file:
        # Skipping the first 33 lines and start reading from line 34 since the names are from those rows.
        lines = file.readlines()[33:]
    # Extract feature names from the file content
    feature_names = []
    for line in lines:
        if ':' in line:
            name = line.split(':')[0]
            feature_names.append(name)
    return feature_names

# Adding names
feature_names = extract_feature_names(spambase_names_path)
feature_names.append('is_spam')  # Adding the class label since it is not defined as the last column name

# Load the dataset with names
spambase_data_path = 'Maskin Learning/A2/spambase.data' # replace to your filepath
data = pd.read_csv(spambase_data_path, names=feature_names)


# Separate X and y
X = data.drop('is_spam', axis=1)
y = data['is_spam']


# In[2]:


from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


# In[3]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

# Initianlizing classifiers
classifiers = {
    "Random Forest": RandomForestClassifier(n_estimators=100, max_features='sqrt'),
    "Naive Bayes": GaussianNB(),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
}


# In[4]:


from sklearn.model_selection import StratifiedKFold

# Stratified K-Fold with 10 splits
skf = StratifiedKFold(n_splits=10)


# In[5]:


from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, f1_score
from time import time

# Metrics to compute
scoring = ['accuracy', 'f1']

# Dictionary to store all results
results = {'Accuracy': {name: [] for name in classifiers.keys()},
           'F-Measure': {name: [] for name in classifiers.keys()},
           'Training Time': {name: [] for name in classifiers.keys()}}

# Perform cross-validation for each classifier
for name, clf in classifiers.items():
    cv_results = cross_validate(clf, X_train, y_train, cv=skf, scoring=scoring, return_train_score=False, n_jobs=-1)

    results['Accuracy'][name] = cv_results['test_accuracy']
    results['F-Measure'][name] = cv_results['test_f1']
    results['Training Time'][name] = cv_results['fit_time']

# Convert results into a DataFrame
for metric in results:
    results[metric] = pd.DataFrame(results[metric])


# In[6]:


# Calculate mean and standard deviation for each classifier in each metric
final_results = {}
for metric, df in results.items():
    means = df.mean(axis=0).rename('Mean')
    stds = df.std(axis=0).rename('Std')
    final_results[metric] = pd.concat([means, stds], axis=1)


# In[7]:


# Print the results in table 12.4 format
for metric in results:
    print(f"{metric} with Mean and Std:")
    print('-'*55)
    
    # Print the classifier names
    classifier_names = results[metric].columns
    print('   ', '  '.join(classifier_names))
    print('-'*55)
    
    # Print each row of data (for each fold)
    for index in range(len(results[metric])):
        row_data = [results[metric][classifier].iloc[index] for classifier in classifier_names]
        print(f"{index:<3}", '  '.join(f"{val:.6f}" for val in row_data))
    
    print('-'*55)

    # Calculate and print mean and std for each classifier
    means = results[metric].mean()
    stds = results[metric].std()

    print("mean", '  '.join(f"{means[name]:.6f}" for name in classifier_names))
    print("std", '  '.join(f"{stds[name]:.6f}" for name in classifier_names))
    print('-'*55)
    print()


# In[8]:


# Update the results DataFrame with ranks for each fold
for metric, df in results.items():
    ascending = True if metric == 'Training Time' else False
    ranks = df.rank(axis=1, ascending=ascending)
    
    ranked_df = df.copy()
    for col in ranked_df.columns:
        ranked_df[col] = ranked_df[col].round(6).astype(str) + " (" + ranks[col].astype(int).astype(str) + ")"
    
    results[metric] = ranked_df


# In[9]:


# save avg ranks for fridman test
average_ranks = {}

# # Print the results in table 12.8 format
for metric, df in results.items():
    print(f"{metric} with Ranks:")
    print('-'*55)
    
    # Print the classifier names
    classifier_names = df.columns
    print('  ', '  '.join(classifier_names))
    print('-'*55)
    
    # Print each row of data with ranks
    for index, row in df.iterrows():
        formatted_row = [f"{value}" for value in row]
        print(index, '  '.join(formatted_row))
    
    print('-'*55)

    # Calculate and store the average rank for each classifier
    avg_ranks = df.applymap(lambda x: int(x.split('(')[1].replace(')', ''))).mean(axis=0)
    average_ranks[metric] = avg_ranks.values

    print("Average Rank", '  '.join(f"{avg_ranks[name]:.1f}" for name in classifier_names))
    print('-'*55)
    print()

# Make the avg ranks to list
avg_ranks_accuracy = list(average_ranks.get('Accuracy', []))
avg_ranks_fmeasure = list(average_ranks.get('F-Measure', []))
avg_ranks_training_time = list(average_ranks.get('Training Time', []))


# In[10]:


import numpy as np

def friedman_statistic(avg_ranks, N, k):
    # Calculation of Friedman statistic
    sum_of_squares = np.sum(avg_ranks ** 2)
    chi2 = (12 * N / (k * (k + 1))) * (sum_of_squares - (k * (k + 1) ** 2 / 4))
    degrees_of_freedom = k - 1

    return chi2, degrees_of_freedom


N = 10  # Number of folds
k = 3   # Number of algorithms

# Retrive the avg rank for each metric
avg_ranks_accuracy = np.array(avg_ranks_accuracy) 
avg_ranks_fmeasure = np.array(avg_ranks_fmeasure)
avg_ranks_training_time = np.array(avg_ranks_training_time)

# Friedman statistic for each metric
chi2_accuracy, df_accuracy = friedman_statistic(avg_ranks_accuracy, N, k)
chi2_fmeasure, df_fmeasure = friedman_statistic(avg_ranks_fmeasure, N, k)
chi2_training_time, df_training_time = friedman_statistic(avg_ranks_training_time, N, k)

# Print the stats
print("Friedman Statistic and Degrees of Freedom for Each Metric:")
print(f"Accuracy: Chi2 = {chi2_accuracy}, df = {df_accuracy}")
print(f"F-Measure: Chi2 = {chi2_fmeasure}, df = {df_fmeasure}")
print(f"Training Time: Chi2 = {chi2_training_time}, df = {df_training_time}")


# In[11]:


def is_significant(chi2_statistic):
    # determine if value is significant
    critical_value = 6.2
    return chi2_statistic > critical_value

def nemenyi_critical_difference(N, k, alpha=0.05):
    # Critical value q_alpha for the studentized range statistic (to be verified)
    q_alpha = 2.343  # Example value
    return q_alpha * np.sqrt((k * (k + 1)) / (6 * N))


# Check significance for each metric and calculate critical differences if significant
significance_accuracy = is_significant(chi2_accuracy)
significance_fmeasure = is_significant(chi2_fmeasure)
significance_training_time = is_significant(chi2_training_time)

# Only retrive the value if significant
cd_accuracy = nemenyi_critical_difference(N, k) if significance_accuracy else None
cd_fmeasure = nemenyi_critical_difference(N, k) if significance_fmeasure else None
cd_training_time = nemenyi_critical_difference(N, k) if significance_training_time else None

# Print results
print("Significance and Critical Differences for Each Metric:")
print(f"Accuracy: Significance = {significance_accuracy}, CD = {cd_accuracy}")
print(f"F-Measure: Significance = {significance_fmeasure}, CD = {cd_fmeasure}")
print(f"Training Time: Significance= {significance_training_time}, CD = {cd_training_time}")


# In[12]:


def significant_difference(rank_lst, cd):
    # Retrive indexes where there is significant difference between avg ranks of classifiers.
    result = []
    for i in range(len(avg_ranks_cd[0])-1):
        for j in range(len(rank_lst)):
            if abs(rank_lst[i] - rank_lst[j]) > cd and {i, j} not in result:
                result.append({i, j})
    return result


def show(lst):
    # Print the sets 
    for pair in lst:
        print(f"{cls[pair.pop()]} and {cls[pair.pop()]} are significantly differerent")


avg_ranks_with_cd = {"accuracy":[avg_ranks_accuracy, cd_accuracy],
                     "fmeasure": [avg_ranks_fmeasure, cd_fmeasure], 
                     "training_time":[avg_ranks_training_time, cd_training_time]}

cls = ["Random Forest",
       "Naive Bayes", 
       "Gradient Boosting"]

for measure, avg_ranks_cd in avg_ranks_with_cd.items():
    # Loop through the messures and check for differences.
    print("Significance for", measure, ":")
    if avg_ranks_cd[1]:
        result = significant_difference(avg_ranks_cd[0], avg_ranks_cd[1])
        show(result)
        print()

