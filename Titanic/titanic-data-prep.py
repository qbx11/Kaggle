import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


train = pd.read_csv("train.csv")

"""data preparation"""

#check for missing data
#print(train['Fare'].isnull().sum())

#text->number conversion
train['Sex_numeric'] = train['Sex'].map({'male': 0,'female': 1})

Y = train['Survived']
X = train[['Sex_numeric', 'Pclass','Fare']]

#double check for missing data
#print(X.isnull().sum())

"""Train/Validation Split"""
X_train, X_val, Y_train, Y_val = train_test_split(
    X,Y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression()
model.fit(X_train,Y_train)

print("Wagi:", model.coef_)
print("Bias:", model.intercept_)

predictions = model.predict(X_val)
accuracy = accuracy_score(Y_val, predictions)
print(f"Accuracy: {accuracy:.2%}")
