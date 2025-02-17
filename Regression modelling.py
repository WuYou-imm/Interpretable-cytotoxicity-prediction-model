import pandas as pd
import numpy as np
import sklearn
import sklearn.metrics
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
from sklearn import svm
from sklearn.model_selection import KFold
from math import sqrt
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import io
from PIL import Image
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import matplotlib.font_manager as fm
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


def svc_cv(C, gamma):
    val = cross_val_score(
        svm.SVR(
            kernel='rbf',
            C=C,
            gamma=gamma,
        ),
        X_train_val, Y_train_val, scoring='r2', cv=5).mean()
    return val


def rf_cv(n_estimators, min_samples_split, max_features, max_depth):
    val = cross_val_score(
        RandomForestRegressor(n_estimators=int(n_estimators),
                              min_samples_split=int(min_samples_split),
                              max_features=min(max_features, 0.999),
                              max_depth=int(max_depth),

                              n_jobs=-1,

                              random_state=123
                              ),
        X_train_val, Y_train_val, scoring='r2', cv=5).mean()
    return val
def lgb_cv(n_estimators,subsample, max_depth,colsample_bytree, reg_alpha,reg_lambda,num_leaves,learning_rate):
    val = cross_val_score(
            lgb.LGBMRegressor(objective='regression',n_jobs=-1,
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
    X_train_val,Y_train_val,scoring='r2', cv=5).mean()
    return val


def xgb_cv(n_estimators, subsample, max_depth, colsample_bytree, reg_alpha, reg_lambda, learning_rate):
    val = cross_val_score(
        xgb.XGBRegressor(colsample_bytree=float(colsample_bytree),
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
        X_train_val, Y_train_val, scoring='r2', cv=5).mean()
    return val
X_train_val,Y_train_val,X_test,Y_test,X_all,Y_all=get_data("Cell viability")
rf_bo = BayesianOptimization(
    rf_cv,
    {'n_estimators': (100, 500),
     'min_samples_split': (2, 8),
     'max_features': (0.2, 1),
     'max_depth': (4, 12)},

)
rf_bo.maximize(n_iter=25)
print(rf_bo.max)

svc_bo = BayesianOptimization(svc_cv,
        { 'C': (0.1,2),'gamma': (0.001,0.1)}
    )

svc_bo.maximize(n_iter=25)

# 可以输出最优的值以及最优参数等等
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

# 训练
lgb_bo.maximize(n_iter=25)

# 可以输出最优的值以及最优参数等等
print(lgb_bo.max)

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

# 训练
xgb_bo.maximize(n_iter=25)

# 可以输出最优的值以及最优参数等等
print(xgb_bo.max)

model1=svm.SVR(kernel='rbf',C=svc_bo.max['params']['C'], gamma=svc_bo.max['params']['gamma'])
model2=RandomForestRegressor(random_state=123,max_depth=int(rf_bo.max['params']['max_depth']), max_features=rf_bo.max['params']['max_features'], min_samples_split= int(rf_bo.max['params']['min_samples_split']), n_estimators= int(rf_bo.max['params']['n_estimators']))
model3=xgb.XGBRegressor(random_state=123,colsample_bytree= xgb_bo.max['params']['colsample_bytree'], learning_rate= xgb_bo.max['params']['learning_rate'],
                      max_depth= int(xgb_bo.max['params']['max_depth']), n_estimators=int(xgb_bo.max['params']['n_estimators']), reg_alpha=xgb_bo.max['params']['reg_alpha'], reg_lambda=xgb_bo.max['params']['reg_lambda'], subsample= xgb_bo.max['params']['subsample'])
model4=lgb.LGBMRegressor(objective='regression',random_state=123,colsample_bytree= lgb_bo.max['params']['colsample_bytree'], learning_rate= lgb_bo.max['params']['learning_rate'],
                      max_depth= int(lgb_bo.max['params']['max_depth']), n_estimators=int(lgb_bo.max['params']['n_estimators']), num_leaves= int(lgb_bo.max['params']['num_leaves']), reg_alpha=lgb_bo.max['params']['reg_alpha'], reg_lambda=lgb_bo.max['params']['reg_lambda'], subsample= lgb_bo.max['params']['subsample'])
from sklearn.ensemble import StackingRegressor
def get_stacking():
 # define the base models
    level0 = list()
    #level0.append(('lr', LogisticRegression()))
    level0.append(('svm', model1))
    level0.append(('RF', model2))
    level0.append(('XGB',model3))
    level0.append(('LGB', model4))
 # define meta learner model
    level1 = LinearRegression()
 # define the stacking ensemble
    model = StackingRegressor(estimators=level0, final_estimator=level1)
    return model

from sklearn.ensemble import VotingRegressor
model5=VotingRegressor(estimators=[('SVR',model1),('RF',model2),('XGB',model3),('LGB',model4)])
model6=get_stacking()

for model,label in zip([model1,model2,model3,model4,model5,model6],['SVR','RF','XGB','LGB','Voting','Stacking']):
    #scores=cross_val_score(model,X,Y,cv=5,scoring='r2')#交叉验证
    #print('{} r2:{}'.format(label,scores.mean()))

    model.fit(X_train_val,Y_train_val)

    print(label,'Test R2:',r2_score(Y_test,model.predict(X_test)))
    print(label,'Test Pearson:',sc.stats.pearsonr(Y_test,model.predict(X_test)))
    print("___________________________________________________________")

pearson=['0.7025','0.7422','0.7275','0.7337','0.7415','0.7405']
r2=['0.4903','0.5446','0.5238','0.5360','0.5467','0.5468']
model=['SVR','RF','XGB','LGB','Voting model','Stacking model']
Date='_20240401'
Name='Regression_scatter_plot_using_cellViability_50_features'
Models=[model1,model2,model3,model4,model5,model6]
for i in range(0,6):
    sns.set_palette('YlGnBu',3)
    sns.set_style('white')
    font_prop = fm.FontProperties(family='Arial', size=16, weight='bold')
    Model=Models[i]
    g=sns.jointplot(x=Model.predict(X_test)*100, y=Y_test*100 ,color = '#4C91C2',s =50, edgecolor="black",linewidth=1,
                  kind = 'scatter',xlim=(-10,120),ylim=(-10,120),height=5,marginal_kws=dict(bins=40, rug=True))
    plt.plot((0, 100), (0, 100),ls='--',c='k')
    g.ax_marg_x.set_axis_off()
    g.ax_marg_y.set_axis_off()
    font_prop = fm.FontProperties(family='Arial', size=14, weight='bold')
    g.ax_joint.set_xlabel('Predicted cell viability using '+model[i]+' (%)', fontproperties=font_prop)
    g.ax_joint.set_ylabel('Observed cell viability(%)',fontproperties=font_prop)
    rsquare = lambda a, b: pcor(a, b)[0]

    g.ax_joint.set_xticklabels(g.ax_joint.get_xticklabels(), fontproperties=font_prop)
    g.ax_joint.set_yticklabels(g.ax_joint.get_yticklabels(), fontproperties=font_prop)
    font_prop = fm.FontProperties(family='Arial', size=16, weight='bold')
    plt.text(-5,115,'Pearson r = '+pearson[i], fontproperties=font_prop)
    plt.text(-5,105,'R$^2$ = '+ r2[i], fontproperties=font_prop)
    plt.tight_layout()
    plt.savefig(Name+model[i]+Date+".svg", format='svg',dpi=600)
    plt.savefig(Name+model[i]+Date+".tiff", format='tiff',dpi=600)
    plt.savefig(Name+model[i]+Date+".png", format='png',dpi=600)

import pickle
pickle.dump(model1,open("CV0.5_SVR_regression_model_20240401.dat","wb"))
pickle.dump(model2,open("CV0.5_RF_regression_model_20240401.dat","wb"))
pickle.dump(model3,open("CV0.5_XGB_regression_model_20240401.dat","wb"))
pickle.dump(model4,open("CV0.5_LGB_regression_model_20240401.dat","wb"))
pickle.dump(model5,open("CV0.5_Voting_regression_model_20240401.dat","wb"))
pickle.dump(model6,open("CV0.5_Stacking_regression_model_20240401.dat","wb"))