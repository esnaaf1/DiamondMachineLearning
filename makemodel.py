import pandas as pd
import numpy as np

# Read in CSV file.
diamonds_df = pd.read_csv("data/new_clean_data.csv")

# Make 'id' the index column.
diamonds_df.set_index('id', inplace = True)

# Create our features.
X = diamonds_df.drop('type', axis =1)

# Create our target.
y = diamonds_df['type']


# Split our data into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,
    y, random_state=1, stratify=y)

# Create a Logistic Regression Model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver='lbfgs',
   max_iter=200,
   random_state=1)

# Train the data
classifier.fit(X_train.values, y_train)

# Save (pickle) the model
from pickle import dump, load
dump(classifier, open('classifier.pkl', 'wb'))



