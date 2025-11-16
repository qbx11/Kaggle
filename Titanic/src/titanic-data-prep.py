import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler




train = pd.read_csv("train.csv")

"""data preparation"""

#check for missing data
#print(train['Fare'].isnull().sum())

#text->number conversion
train['Sex_numeric'] = train['Sex'].map({'male': 0,'female': 1})

#titles:
train['Title'] = train['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)



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
#converting rare titles into more common ones
train['Title'] = train['Title'].replace(title_mapping)
#print(train['Title'].value_counts())

#filling gaps in 'Age' column using mean age from 'Title' column
train['Age'] = train.groupby('Title')['Age'].transform(lambda x: x.fillna(x.mean()))
#print(train['Age'].isnull().value_counts())

#family size = SibSp + Parch (ONE-HOT ENCODING)
train['Family size'] = train['SibSp'] + train['Parch']
train['Alone'] = (train['Family size']==0).astype(int)
train['Small family'] = ((train['Family size']>=1) & (train['Family size']<=3)).astype(int)
train['Large family'] = (train['Family size']>3).astype(int)

#filling gaps in 'Embarked' column
train["Embarked"] = train["Embarked"].transform(lambda x: x.fillna('S'))

#one-hot encoding for 'Title' column
title_dummies = pd.get_dummies(train['Title'],prefix='Title_')
train = pd.concat([train,title_dummies],axis=1)

#one-hot encoding for 'Embarked' column
embarked_dummies = pd.get_dummies(train['Embarked'],prefix='Embarked_')
train = pd.concat([train,embarked_dummies],axis=1)


feature_cols = ['Sex_numeric','Age', 'Pclass','Fare','Alone',
                #Family size
                'Small family','Large family',
                #Title
                'Title__Master','Title__Miss', 'Title__Mr', 'Title__Mrs',
                #Embarked
                'Embarked__C', 'Embarked__Q', 'Embarked__S'
                ]

X = train[feature_cols]
Y = train['Survived']


"""Train/Validation Split"""
X_train, X_val, Y_train, Y_val = train_test_split(
    X,Y,
    test_size=0.2,
    random_state=42
)

#SCALER
scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)

model = LogisticRegression(max_iter=100)
model.fit(X_train_scaled,Y_train)

predictions = model.predict(X_val_scaled)
accuracy = accuracy_score(Y_val, predictions)
print(f"Accuracy: {accuracy:.2%}")



feature_weight = sorted(list(zip(feature_cols,model.coef_[0])),key=lambda x: abs(x[1]),reverse=True)
for name,wage in feature_weight:
    print(f'{name:<15}: {wage:+.3f}')
print("Bias:", model.intercept_)

#data frame to csv
train.to_csv('train_po_przetworzeniu.csv', index=False)
