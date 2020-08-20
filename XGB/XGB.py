import xgboost as xgb
import pandas as pd
import numpy as np

diabetes=pd.read_csv("C:/Users/snair/Documents/Data Science Assignment/Data Sets/Diabetes_RF.csv")
diabetes.head()
diabetes.columns

colnames = list(diabetes.columns)
predictors = colnames[:8]
target = colnames[8]

from sklearn.model_selection import train_test_split
train,test=train_test_split(diabetes,test_size=0.2,random_state=5)

model=xgb.XGBClassifier(n_estimators=2000,learning_rate=0.01)
model.fit(train[predictors],train[target])

pred=model.predict(train[predictors])
pd.crosstab(train[target],pred)
train_accuracy=np.mean(train[target]==pred) # 100%

pred=model.predict(test[predictors])
pd.crosstab(test[target],pred)
test_accuracy=np.mean(test[target]==pred) # 76%

from xgboost import plot_importance
plot_importance(model)

#Hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV

#RandomizedSearchCV

params={
        'learning_rate':[0.01,0.05,0.1,0.15,0.2,0.25,0.3],
        'max_depth' :  [3,4,5,6,8,10,12,15],
        'min_child_weight':[1,3,5,7],
        'gamma'  : [0.0,0.1,0.2,0.3,0.4],
        'colsample_bytree':[0.3,0.4,0.5,0.7],
        'subsample' : [0.5,0.6,0.7,0.8,0.9,1.0],
        'n_estimators':[100,200,300,500,700,1000]
        }

classifier=xgb.XGBClassifier()
random_search=RandomizedSearchCV(classifier,param_distributions=params,n_iter=10,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)
random_search.fit(train[predictors],train[target])

random_search.best_estimator_
random_search.best_params_

classifier=xgb.XGBClassifier(base_score=0.5, booster=None, colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.5, gamma=0.3, gpu_id=-1,
              importance_type='gain', interaction_constraints=None,
              learning_rate=0.01, max_delta_step=0, max_depth=3,
              min_child_weight=3,  monotone_constraints=None,
              n_estimators=100, n_jobs=-1, num_parallel_tree=1, random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=0.9,
              tree_method=None, validate_parameters=False, verbosity=None)

classifier.fit(train[predictors],train[target])
pred=classifier.predict(test[predictors])
pd.crosstab(test[target],pred)
np.mean(test[target]==pred) # 81.16%

#GridSearchCV
params=[
        {'learning_rate':[0.01,0.05,0.1,0.15,0.2,0.25,0.3]},
        {'max_depth' :  [3,4,5,6,8,10,12,15]},
        {'min_child_weight':[1,3,5,7]},
        {'gamma'  : [0.0,0.1,0.2,0.3,0.4]},
        {'colsample_bytree':[0.3,0.4,0.5,0.7]},
        {'subsample' : [0.5,0.6,0.7,0.8,0.9,1.0]},
        {'n_estimators':[100,200,300,500,700,1000]}
        ]
classifier=xgb.XGBClassifier()
grid_search=GridSearchCV(estimator=classifier,param_grid=params,scoring='accuracy',n_jobs=-1,cv=10)
grid_search.fit(train[predictors],train[target])

grid_search.best_score_
grid_search.best_params_

classifier=xgb.XGBClassifier(learning_rate=0.01)
classifier.fit(train[predictors],train[target])
pred=classifier.predict(test[predictors])
pd.crosstab(test[target],pred)
np.mean(test[target]==pred) # 77.92%

##########################################################################

housing=pd.read_csv("C:/Users/snair/Documents/Data Science Assignment/Data Sets/housing_XGBRegressor.csv")
housing.head()
housing.columns

colnames = list(housing.columns)
predictors = colnames[:13]
target = colnames[13]

train,test=train_test_split(housing,test_size=0.2)

model=xgb.XGBRegressor(n_estimators=2000,learning_rate=0.01)
model.fit(train[predictors],train[target])

pred=model.predict(train[predictors])
from sklearn.metrics import explained_variance_score
explained_variance_score(train[target],pred) # accuracy=99.98%
np.sqrt(np.mean((train[target]-pred)**2))    # RMSE error=0.105

pred=model.predict(test[predictors])
explained_variance_score(test[target],pred) # accuracy=94.23%
np.sqrt(np.mean((test[target]-pred)**2))    # RMSE error=2.03

plot_importance(model)













