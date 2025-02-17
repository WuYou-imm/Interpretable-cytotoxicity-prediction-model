import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import mean_squared_error,r2_score
import sklearn.metrics
import warnings
import xgboost as xgb
import scipy as sc
import math
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from math import sqrt
from sklearn import svm, datasets
from sklearn.metrics import r2_score
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from lightgbm import LGBMClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import StratifiedKFold
from bayes_opt import BayesianOptimization
import os
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import matplotlib.font_manager as font_manager
###Define function
def get_data(cv):
    Train_metadata= pd.read_csv('Y_train20230610.csv')
    Test_metadata= pd.read_csv('Y_test20230610.csv')
    Train_signature= pd.read_csv('X_train_RRA_CV50_20230630.csv',index_col=0)
    Test_signature= pd.read_csv('X_test_RRA_CV50_20230630.csv',index_col=0)
    Y_train_val=Train_metadata[cv]
    Y_test=Test_metadata[cv]
    X_train_val=Train_signature
    X_test=Test_signature
    X_all=pd.concat([Train_signature,Test_signature], axis=0)
    Y_all=pd.concat([Y_train_val,Y_test], axis=0)
    return X_train_val,Y_train_val,X_test,Y_test,X_all,Y_all
def get_stacking():
 # define the base models
    level0 = list()
    #level0.append(('lr', LogisticRegression()))
    level0.append(('svm', model1))
    level0.append(('RF', model2))
    level0.append(('XGB',model3))
    level0.append(('LGB', model4))
 # define meta learner model
    level1 = LogisticRegression()
 # define the stacking ensemble
    model = StackingClassifier(estimators=level0, final_estimator=level1)
    return model
def model_score2(name, X_train_val, Y_train_val, X_test, Y_test, model1, model2, model3, model4, model5, model6):
    df = pd.DataFrame()
    for model, label in zip([model1, model2, model3, model4, model5, model6],
                            ['SVM', 'RF', 'XGB', 'LGB', 'Voting', 'Stacking']):
        model.fit(X_train_val, Y_train_val)
        predict_test = model.predict(X_test)
        prob_test = model.predict_proba(X_test)
        predict_test_value = prob_test[:, 1]
        report = sklearn.metrics.classification_report(Y_test, predict_test, output_dict=True)
        df_1 = pd.DataFrame(report).transpose()
        df = pd.concat([df, df_1], axis=0)
        print(label, "ROC_AUC:", sklearn.metrics.roc_auc_score(Y_test, predict_test_value),
              sklearn.metrics.classification_report(Y_test, predict_test))

    plt.figure(dpi=600, figsize=(3.15, 2.75))
    sns.set_palette('GnBu', 5)
    sns.set(style='ticks')
    # Set font to Times New Roman
    font_prop = font_manager.FontProperties(family='Times New Roman', size=9, weight='bold')
    plt.xlabel('FPR', fontproperties=font_prop)
    plt.ylabel('TPR', fontproperties=font_prop)
    # Calculate ROC for each model
    models = [model1, model2, model3, model4, model5, model6]
    labels = ['SVM', 'RF', 'XGB', 'LGB', 'Voting', 'Stacking']
    for model, label in zip(models, labels):
        model.fit(X_train_val, Y_train_val)
        predict_test = model.predict(X_test)
        prob_test = model.predict_proba(X_test)
        predict_test_value = prob_test[:, 1]
        fpr, tpr, _ = metrics.roc_curve(Y_test, predict_test_value)
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1.5, label=f'{label}:AUC = {roc_auc:.4f}')
    plt.legend(loc=0, prop=font_prop)
    plt.ylim(0, 1.1)
    plt.xticks(size=9, weight='bold', fontname='Times New Roman')
    plt.yticks(size=9, weight='bold', fontname='Times New Roman')
    plt.tight_layout()
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(name, dpi=600, format='tiff')

    # Prepare data for return
    x = pd.concat([pd.DataFrame(fpr), pd.DataFrame(tpr)], axis=1)
    x.columns = ['FPR', 'TPR']
    return df, x
def svc_cv(C, gamma):
    val = cross_val_score(
        SVC(
            kernel='rbf',
            probability=True,
            C=C,
            gamma=gamma,
            random_state=123
        ),
        X_train_val, Y_train_val, scoring='roc_auc', cv=10).mean()
    return val
def lgb_cv(n_estimators,subsample, max_depth,colsample_bytree, reg_alpha,reg_lambda,num_leaves,learning_rate):
    val = cross_val_score(
            lgb.LGBMClassifier(boosting_type='gbdt', objective='binary',n_jobs=-1,
                                   colsample_bytree=float(colsample_bytree),
                                   #min_child_samples=int(min_child_samples),
                                   n_estimators=int(n_estimators),
                                   num_leaves=int(num_leaves),
                                   reg_alpha=float(reg_alpha),
                                   reg_lambda=float(reg_lambda),
                                   max_depth=int(max_depth),
                                   subsample=float(subsample),
                                   #min_gain_to_split = float(min_gain_to_split),
                                   learning_rate=float(learning_rate),
                                   random_state=123,is_unbalance=True
        ),
    X_train_val,Y_train_val,scoring='roc_auc', cv=10).mean()
    return val
def rf_cv(n_estimators, min_samples_split, max_features, max_depth):
    val = cross_val_score(
        RandomForestClassifier(n_estimators=int(n_estimators),
                               min_samples_split=int(min_samples_split),
                               max_features=min(max_features, 0.999),
                               max_depth=int(max_depth),

                               n_jobs=-1,
                               class_weight='balanced',
                               random_state=123
                               ),
        X_train_val, Y_train_val, scoring='roc_auc', cv=10).mean()
    return val
def xgb_cv(n_estimators, subsample, max_depth, colsample_bytree, reg_alpha, reg_lambda, learning_rate):
    val = cross_val_score(
        xgb.XGBClassifier(colsample_bytree=float(colsample_bytree),
                          # min_child_samples=int(min_child_samples),
                          n_estimators=int(n_estimators),

                          reg_alpha=float(reg_alpha),
                          reg_lambda=float(reg_lambda),
                          max_depth=int(max_depth),
                          subsample=float(subsample),
                          # min_gain_to_split = float(min_gain_to_split),
                          learning_rate=float(learning_rate),
                          random_state=123
                          ),
        X_train_val, Y_train_val, scoring='roc_auc', cv=10).mean()
    return val
model1=SVC(probability=True,kernel='rbf')
model2=RandomForestClassifier(random_state=123,class_weight='balanced')
model3=xgb.XGBClassifier(random_state=123,is_unbalance=True)
model4=LGBMClassifier(random_state=123,class_weight='balanced')
model5=VotingClassifier(estimators=[('SVC',model2),('RF',model2),('XGB',model3),('LGB',model4)],voting='soft')
model6=get_stacking()
svc_bo = BayesianOptimization(svc_cv,
        { 'C': (0.1,2),'gamma': (0.001,0.1)}
    )
svc_bo.maximize(n_iter=25)
print(svc_bo.max)

lgb_bo = BayesianOptimization(
    lgb_cv,
    {
        'colsample_bytree': (0.8, 1),

        'num_leaves': (16, 256),
        'subsample': (0.4, 1),
        'max_depth': (3, 5),
        'n_estimators': (100, 1000),
        'reg_alpha': (0, 0.015),
        'reg_lambda': (0, 0.015),

        'learning_rate': (0.0001, 0.1)
    },
)
lgb_bo.maximize(n_iter=25)
print(lgb_bo.max)

rf_bo = BayesianOptimization(
    rf_cv,
    {'n_estimators': (100, 500),  
     'min_samples_split': (2, 8),
     'max_features': (0.2, 1),

     'max_depth': (4, 12)},

)
rf_bo.maximize(n_iter=25)
print(rf_bo.max)

xgb_bo = BayesianOptimization(
    xgb_cv,
    {
        'colsample_bytree': (0.8, 1),

        'subsample': (0.4, 1),
        'max_depth': (3, 5),
        'n_estimators': (100, 1000),
        'reg_alpha': (0, 0.015),
        'reg_lambda': (0, 0.015),

        'learning_rate': (0.0001, 0.1)
    },
)
xgb_bo.maximize(n_iter=25)
print(xgb_bo.max)

model1=SVC(probability=True,kernel='rbf',C=svc_bo.max['params']['C'], gamma=svc_bo.max['params']['gamma'])
model2=RandomForestClassifier(random_state=123,class_weight='balanced',max_depth=int(rf_bo.max['params']['max_depth']), max_features=rf_bo.max['params']['max_features'], min_samples_split= int(rf_bo.max['params']['min_samples_split']), n_estimators= int(rf_bo.max['params']['n_estimators']))
model3=xgb.XGBClassifier(random_state=123,colsample_bytree= xgb_bo.max['params']['colsample_bytree'], learning_rate= xgb_bo.max['params']['learning_rate'],
                      max_depth= int(xgb_bo.max['params']['max_depth']), n_estimators=int(xgb_bo.max['params']['n_estimators']), reg_alpha=xgb_bo.max['params']['reg_alpha'], reg_lambda=xgb_bo.max['params']['reg_lambda'], subsample= xgb_bo.max['params']['subsample'])
model4=LGBMClassifier(random_state=123,class_weight='balanced',colsample_bytree= lgb_bo.max['params']['colsample_bytree'], learning_rate= lgb_bo.max['params']['learning_rate'],
                      max_depth= int(lgb_bo.max['params']['max_depth']), n_estimators=int(lgb_bo.max['params']['n_estimators']), num_leaves= int(lgb_bo.max['params']['num_leaves']), reg_alpha=lgb_bo.max['params']['reg_alpha'], reg_lambda=lgb_bo.max['params']['reg_lambda'], subsample= lgb_bo.max['params']['subsample'])
model5=VotingClassifier(estimators=[('SVC',model1),('RF',model2),('XGB',model3),('LGB',model4)],voting='soft')
model6=get_stacking()

report_cv50_after_tune_2,x2=model_score2("  ",X_train_val,Y_train_val,X_test,Y_test,model1, model2, model3, model4, model5, model6)