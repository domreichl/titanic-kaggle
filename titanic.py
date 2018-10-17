''' titanic.py
Date: October 17, 2018
Author: Dominic Reichl, @domreichl
Title: Titanic (my first kaggle competition)
Goal: Predict which passengers survived the sinking of the RMS Titanic
Method: C-support vector classification
Kaggle Score: 0.80861 -> top 8% in leaderboard
'''

import pandas as pd
from numpy import array
from sklearn.svm import SVC

def preprocess(df):
    # create new feature Solo from SibSp and Parch
    df['Solo'] = 0
    df.loc[(df['SibSp'] == 0) & (df['Parch'] == 0), 'Solo'] = 1
    
    # create new feature Title from Name
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    other = ['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
    df['Title'] = df['Title'].replace(other, 'Other')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    # fill in missing values
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Age'] = df['Age'].fillna(value=-1)
    df.loc[(df['Age'] == -1) &(df['Title'] == 'Master'), 'Age'] = 5.25
    df.loc[(df['Age'] == -1) &(df['Title'] == 'Miss'), 'Age'] = 21.5
    df.loc[(df['Age'] == -1) &(df['Title'] == 'Mr'), 'Age'] = 29.25
    df.loc[(df['Age'] == -1) &(df['Title'] == 'Mrs'), 'Age'] = 35.75
    df.loc[(df['Age'] == -1) &(df['Title'] == 'Other'), 'Age'] = 46.25

    # convert string categories to integers
    df['Title'] = df['Title'].map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    # categorize Fare
    df.loc[ df['Fare'] <= 7.91, 'Fare'] = 0
    df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1
    df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare']   = 2
    df.loc[ df['Fare'] > 31, 'Fare'] = 3

    # categorize Age
    df.loc[ df['Age'] <= 16, 'Age'] = 0
    df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1
    df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2
    df.loc[ df['Age'] > 48, 'Age'] = 3
    
    return df

# load and preprocess data
train = preprocess(pd.read_csv('train.csv'))
test  = preprocess(pd.read_csv('test.csv'))

# create arrays with important features
features = ['Pclass', 'Solo', 'Title', 'Embarked', 'Sex', 'Fare', 'Age']
X = array(train[features])
y = array(train['Survived'])
X_test = array(test[features])

# build and fit support vector machine
model = SVC()
model.fit(X, y)

# get predictions and store them as csv
predictions = model.predict(X_test)
results = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': predictions})
results.to_csv('submission.csv', index=False)
