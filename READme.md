# Movie Genre Classification Model

## Overview

This project involves building a machine learning model to classify movies into their respective genres based on the movie's description. The dataset used consists of movie titles, genres, and descriptions, which are divided into training and test datasets.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Modeling Process](#modeling-process)
  - [1. Data Loading](#1-data-loading)
  - [2. Data Visualization](#2-data-visualization)
  - [3. Data Preprocessing](#3-data-preprocessing)
  - [4. Model Training](#4-model-training)
  - [5. Model Evaluation](#5-model-evaluation)
  - [6. Predictions](#6-predictions)
- [How to Run the Project](#how-to-run-the-project)
- [Conclusion](#conclusion)
- [Future Work](#future-work)

## Dataset

The project uses three data files:
1. **train_data.txt**: Contains the training data with columns: `ID`, `TITLE`, `GENRE`, `DESCRIPTION`.
2. **test_data.txt**: Contains the test data without genre labels.
3. **test_data_solution.txt**: Contains the test data with genre labels for evaluation purposes.

## Dependencies

The project requires the following Python libraries:
- `pandas`
- `numpy`
- `seaborn`
- `matplotlib`
- `scikit-learn`

Install the necessary libraries using:
```bash
pip install pandas numpy seaborn matplotlib scikit-learn

```

# Modeling Process

## 1. Data Loading

The data is loaded into pandas DataFrames using the `read_csv` function, with `:::` as the delimiter:

```python
train_data = pd.read_csv("train_data.txt", sep=':::', names=['ID', 'TITLE', 'GENRE', 'DESCRIPTION'])
test_data = pd.read_csv("test_data.txt", sep=':::', names=['ID', 'TITLE', 'GENRE', 'DESCRIPTION'])
test_solution_data = pd.read_csv("test_data_solution.txt", sep=':::', names=['ID', 'TITLE', 'GENRE', 'DESCRIPTION'])
```
## 2. Data Visualization
### Some visualizations are performed to understand the data distribution:

- **Number of Movies per Genre:** A count plot to show the distribution of movies across different genres.
- **Description Length by Genre:** A bar plot to analyze the length of descriptions for each genre.
## 3. Data Preprocessing
- **Handling Missing Values:** Missing values in the DESCRIPTION column are filled with empty strings.
- **Feature Extraction:** The TfidfVectorizer is used to convert text data into TF-IDF features.
- **Label Encoding:** The genres are encoded into numerical labels using LabelEncoder.
## 4. Model Training
### Three models were used for training:

- **Linear Support Vector Classifier (LinearSVC)**
- **Multinomial Naive Bayes (MultinomialNB)**
- **Logistic Regression**
The dataset is split into training and validation sets using train_test_split, and the models are trained on the training set.

## 5. Model Evaluation
- The validation accuracy and classification report are generated for each model.
- The best-performing model can be selected based on these metrics.
## 6. Predictions
- A function predict_movie is implemented to predict the genre of a movie based on its description:

```python
def predict_movie(description):
    t_v1 = t_v.transform([description])
    pred_label = clf.predict(t_v1)
    return label_encoder.inverse_transform(pred_label)[0]
```
# How to Run the Project
- Clone the repository (if applicable) or download the code files.
- Install the dependencies as mentioned in the Dependencies section.
- Run the Python script containing the code provided above.
- Use the predict_movie function to classify new movie descriptions.
# Conclusion
This project demonstrates the use of text data and machine learning models to classify movie genres. The LinearSVC model provided the best validation accuracy and is used for the final predictions.

# Future Work
- Model Improvement: Experiment with different models, hyperparameter tuning, and feature engineering.
- More Data: Add more data sources to increase the variety of genres and descriptions.
- Deployment: Deploy the model as a web application or an API.
