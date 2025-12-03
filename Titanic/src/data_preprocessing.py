import pandas as pd
from pathlib import Path

#file paths
current_dir = Path.cwd()
project_dir = current_dir.parent
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
title_dummies = pd.get_dummies(train['Title'], prefix='Title_').astype(int)
train = pd.concat([train,title_dummies], axis=1)

#one-hot encoding for 'Embarked' column
embarked_dummies = pd.get_dummies(train['Embarked'],prefix='Embarked_').astype(int)
train = pd.concat([train,embarked_dummies],axis=1)

#categorise age
def categorize_age(age):
    if age <= 12:
        return 'Child'
    elif age <= 17:
        return 'Teen'
    elif age <= 59:
        return 'Adult'
    else:
        return 'Elderly'
train['AgeGroup'] = train['Age'].apply(categorize_age)

#one-hot encoding for 'Age' column
age_dummies = pd.get_dummies(train['AgeGroup'], prefix='Age').astype(int)
train = pd.concat([train,age_dummies],axis=1)

#new feature: FarePerPerson
train['Fare'] = train['Fare'].fillna(train['Fare'].median())
train['FarePerPerson'] = (train['Fare'] / (train['SibSp'] + train['Parch'] + 1)).round(2)

#new feature: HasCabin
train['HasCabin'] = train['Cabin'].notna().astype(int)

#new feature: Deck number
train['Deck'] = train['Cabin'].str[0]
train['Deck'] = train['Deck'].fillna('Unknown')

deck_dummies = pd.get_dummies(train['Deck'],prefix='Deck').astype(int)
train = pd.concat([train,deck_dummies],axis=1)

#new feature: isFemaleFirstClass (highest chance of survival)
train['isFemaleFirstClass'] = ((train['Sex_numeric']==1) & (train['Pclass']==1)).astype(int)

#new featyre: isMaleThirdClass (lowest chance of survival)
train['isMaleThirdClass'] = ((train['Sex_numeric']==0) & (train['Pclass']==3)).astype(int)


#processed data frame to csv
train.to_csv(output_path, index=False)
