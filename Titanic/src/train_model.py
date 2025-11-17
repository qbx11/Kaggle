import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path

#file paths
script_dir = Path(__file__).parent
project_dir = script_dir.parent
train_path = project_dir / 'data' / 'processed' / 'train_processed.csv'

#read data
train = pd.read_csv(train_path)

#define the features
feature_cols = ['Sex_numeric','Age', 'Pclass','Fare',
                #Family size
                'Alone','Small family','Large family',
                #Title
                'Title__Master','Title__Miss', 'Title__Mr', 'Title__Mrs',
                #Embarked
                'Embarked__C', 'Embarked__Q', 'Embarked__S'
                ]

X = train[feature_cols] #features
Y = train['Survived'] #target

#train/calidation split
X_train, X_val, Y_train, Y_val = train_test_split(
    X,Y,
    test_size=0.2,
    random_state=42
)

#data scaling (Standard Scaler)
scaler = StandardScaler()
scaler.fit(X_train)

#update data after scaling
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)

#model training
model = LogisticRegression(max_iter=100)
model.fit(X_train_scaled,Y_train)

#eval
predictions = model.predict(X_val_scaled)
accuracy = accuracy_score(Y_val, predictions)

print(f"Accuracy: {accuracy:.2%}")

#print feature name and wage pairs, print bias
feature_weight = sorted(list(zip(feature_cols,model.coef_[0])),key=lambda x: abs(x[1]),reverse=True)
for name,wage in feature_weight:
    print(f'{name:<15}: {wage:+.3f}')
print("Bias:", model.intercept_)