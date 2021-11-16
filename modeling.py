#Load libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import lightgbm as lgb
from scipy.misc import derivative

## MODELING
# Create features & target and split dataset
X = df_train.drop(['status','category_code'], axis=1).copy()
y = df_train['status'].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Conver to lgb datasets
lgb_train = lgb.Dataset(X_train, y_train)
lgb_test = lgb.Dataset(X_test, y_test)

# Custom Loss Function (Multiclass F1-Score)
def multiclass_f1(preds, df_train):
    """
    Custom multiclass f1-score
    :param preds: Predictions for each observation
    :param df_train: Training dataset
    """
    labels = df_train.get_label()
    preds = preds.reshape(4, -1).T
    preds = preds.argmax(axis = 1)
    f_score = f1_score(labels , preds,  average = 'weighted')
    return 'f1_score', f_score, True

# Define Params
params = {'boosting_type': 'gbdt',
          'objective': 'multiclass',
          'metric': 'multi_logloss',
          'num_class':4,
          'min_data_in_leaf':300,
          'feature_fraction':0.8,
          'bagging_fraction':0.8,
          'bagging_freq':5,
          'max_depth':8,
          'num_leaves':70,
          'learning_rate':0.04}

# Train model
gbm = lgb.train(params, 
                lgb_train,
                feval=multiclass_f1,
                num_boost_round=500,
                valid_sets=[lgb_train, lgb_test],
                early_stopping_rounds=10)


# SHAP
#[0, 1, 2, 3] = ['acquired', 'closed', 'ipo', 'operating']
# Relationships = "Representation of the people involved in the team for that startup"
explainer = shap.TreeExplainer(gbm)
shap_values = explainer.shap_values(X)