import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from xgboost import XGBClassifier


#file paths
script_dir = Path(__file__).parent
project_dir = script_dir.parent
train_path = project_dir / 'data' / 'processed' / 'train_processed.csv'
models_dir = project_dir / 'models'

#read data
train = pd.read_csv(train_path)

#define the features
feature_cols = ['Sex_numeric','Age', 'Pclass','Fare','FarePerPerson','HasCabin',
                #Family size
                'Alone','Small family','Large family',
                #Title
                'Title__Master','Title__Miss', 'Title__Mr', 'Title__Mrs',
                #Embarked
                'Embarked__C', 'Embarked__Q', 'Embarked__S',
                #Age Group
                'Age_Adult', 'Age_Child', 'Age_Elderly', 'Age_Teen'
                ]

X = train[feature_cols] #features
Y = train['Survived'] #target

#train/validation split
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

#save scaler
with open(models_dir,'wb') as file:
    pickle.dump(scaler,file)


#model training - Logistic Regression
lr_model = LogisticRegression(max_iter=100)
lr_model.fit(X_train_scaled,Y_train)

#eval
lr_predictions = lr_model.predict(X_val_scaled)
lr_accuracy = accuracy_score(Y_val, lr_predictions)

print(f"Logistic Regression Accuracy: {lr_accuracy:.2%}")

#print feature name and wage pairs, print bias
feature_weight = sorted(list(zip(feature_cols,lr_model.coef_[0])),key=lambda x: abs(x[1]),reverse=True)
for name,wage in feature_weight:
    print(f'{name:<15}: {wage:+.3f}')
print("Bias:", lr_model.intercept_)

#save model
with open(models_dir / 'lr_model.pkl','wb') as file:
    pickle.dump(lr_model,file)




#model training - Random Forrest (no scaling)
rf_model = RandomForestClassifier(
    n_estimators=500,      # num of trees
    max_depth=10,          # max tree depth
    min_samples_split=5,   # min samples to split
    random_state=42
)

rf_model.fit(X_train,Y_train)
rf_predictions = rf_model.predict(X_val)
rf_accuracy = accuracy_score(Y_val,rf_predictions)

print(f"\nRandom Forrest Accuracy: {rf_accuracy:.2%}")
feature_importance = sorted(list(zip(feature_cols, rf_model.feature_importances_)),
                           key=lambda x: x[1], reverse=True)
for name, importance in feature_importance:
    print(f'{name:<15}: {importance*100:.3f}')

#save model
with open(models_dir / 'rf_model.pkl','wb') as file:
    pickle.dump(rf_model,file)

#model training - XGBoost
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
xgb_model.fit(X_train, Y_train)
xgb_predictions = xgb_model.predict(X_val)
xgb_accuracy = accuracy_score(Y_val,xgb_predictions)
print(xgb_accuracy)

#save model
with open(models_dir / 'xgb_model.pkl','wb') as file:
    pickle.dump(xgb_model,file)

