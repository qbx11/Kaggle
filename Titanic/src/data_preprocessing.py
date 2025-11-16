import pandas as pd
from pathlib import Path

#file paths
script_dir = Path(__file__).parent  # To jest src/
project_dir = script_dir.parent      # To jest Titanic/
train_path = project_dir / 'data' / 'raw' / 'train.csv'
output_path = project_dir / 'data' / 'processed' / 'train_processed.csv'

#read data
train = pd.read_csv(train_path)

#FEATURE ENGINNERING AND PREPROCESSING

#convert 'Sex' column into binary feature
train['Sex_numeric'] = train['Sex'].map({'male': 0,'female': 1})

#extract titles from 'Name' column
train['Title'] = train['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

#map other (less frequent) titles
title_mapping = {
    'Dr': 'Mr',
    'Rev': 'Mr',
    'Mlle': 'Miss',
    'Major': 'Mr',
    'Col': 'Mr',
    'Countess': 'Mrs',
    'Capt': 'Mr',
    'Ms': 'Miss',
    'Sir': "Mr",
    'Lady': 'Mrs',
    'Mme': 'Mr',
    'Don': 'Mr',
    'Jonkheer': 'Mr'
}
train['Title'] = train['Title'].replace(title_mapping)

#fill gaps in 'Age' column using the mean age from each 'Title' column group
train['Age'] = train.groupby('Title')['Age'].transform(lambda x: x.fillna(x.mean()))

#feature for 'Family size'
train['Family size'] = train['SibSp'] + train['Parch']
train['Alone'] = (train['Family size']==0).astype(int)
train['Small family'] = ((train['Family size']>=1) & (train['Family size']<=3)).astype(int)
train['Large family'] = (train['Family size']>3).astype(int)

#fill gaps in 'Embarked' column with the most common port (S)
train["Embarked"] = train["Embarked"].transform(lambda x: x.fillna('S'))

#one-hot encoding for 'Title' column
title_dummies = pd.get_dummies(train['Title'],prefix='Title_')
train = pd.concat([train,title_dummies],axis=1)

#one-hot encoding for 'Embarked' column
embarked_dummies = pd.get_dummies(train['Embarked'],prefix='Embarked_')
train = pd.concat([train,embarked_dummies],axis=1)


#processed data frame to csv
train.to_csv(output_path, index=False)
