import pandas as pd
import pickle
from pathlib import Path

#file paths
script_dir = Path(__file__).parent
project_dir = script_dir.parent
train_path = project_dir / 'data' / 'processed' / 'train_processed.csv'
test_path = project_dir / 'data' / 'processed' / 'test_processed.csv'
models_dir = project_dir / 'models'
results_dir = project_dir / 'results'

#read test data
test = pd.read_csv(test_path)
passenger_ids = test['PassengerId']

#read feature columns
with open(models_dir / 'features.pkl','rb') as file:
    feature_cols = pickle.load(file)

X_test = test[feature_cols] #features

#read scaler
with open(models_dir / 'scaler.pkl','rb') as file:
    scaler = pickle.load(file)

X_test_scaled = scaler.transform(X_test)


'''LOGISTIC REGRESSION'''
#read LR model
with open(models_dir / 'lr_model.pkl','rb') as file:
    lr_model = pickle.load(file)

lr_predictions = lr_model.predict(X_test_scaled)

submission_lr = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived': lr_predictions
})
submission_lr.to_csv(results_dir / 'submission_lr.csv',index=False)

'''RANDOM FOREST'''
#read RF model
with open(models_dir / 'rf_model.pkl','rb') as file:
    rf_model = pickle.load(file)

rf_predictions = rf_model.predict(X_test)

submission_rf = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived': rf_predictions
})
submission_rf.to_csv(results_dir / 'submission_rf.csv',index=False)

'''XGB BOOST'''
#read XGB model
with open(models_dir / 'xgb_model.pkl','rb') as file:
    xgb_model = pickle.load(file)

xgb_predictions = xgb_model.predict(X_test_scaled)

submission_xgb = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived': xgb_predictions
})
submission_xgb.to_csv(results_dir / 'submission_xgb.csv',index=False)


#read MLP from scratch model
from models.mlp_from_scratch import Network

with open(models_dir / 'mlp_model.pkl','rb') as file:
    mlp_model = pickle.load(file)

mlp_model = Network([len(feature_cols),40,20,1])
mlp_model.load_weights(models_dir / 'mlp_model_40_20.pkl')

mlp_predictions = mlp_model.predict_classes(X_test_scaled)

submission_mlp = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived': mlp_predictions
})
submission_mlp.to_csv(results_dir / 'submission_mlp.csv', index=False)



#ensemble
ensemble_predictions = (lr_predictions + rf_predictions + xgb_predictions + mlp_predictions >=2).astype(int)

submission_ensemble = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived': ensemble_predictions
})
submission_ensemble.to_csv(results_dir / 'submission_ensemble_40_20.csv',index=False)

