import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import shap
import os
from matplotlib.lines import Line2D
import io
from PIL import Image
from matplotlib.font_manager import FontProperties
import matplotlib.font_manager as fm
import os
path=os.getcwd()
listDir=os.listdir(path)
for x in listDir:
    print(x)


def get_data(cv):
    Train_metadata = pd.read_csv('Y_train20230610.csv', index_col=0)
    Test_metadata = pd.read_csv('Y_test20230610.csv', index_col=0)

    Train_signature = pd.read_csv('X_train_RRA_CV50_20230630.csv', index_col=0)
    Test_signature = pd.read_csv('X_test_RRA_CV50_20230630.csv', index_col=0)

    Y_train_val = Train_metadata[cv]
    Y_test = Test_metadata[cv]

    X_train_val = Train_signature
    X_test = Test_signature

    X_all = pd.concat([Train_signature, Test_signature], axis=0)
    Y_all = pd.concat([Y_train_val, Y_test], axis=0)
    return X_train_val, Y_train_val, X_test, Y_test, X_all, Y_all

X_train_val,Y_train_val,X_test,Y_test,X_all,Y_all=get_data(" ")

shap_values_50_value= shap_values_50_train.values
# after standard scalar
shap.initjs()
sns.set_theme(style='ticks')
shap.summary_plot(shap_values_50_value, X_train_val, feature_names=X_train_val.columns, max_display=10,show=False, plot_size=(9, 5), cmap=plt.get_cmap("RdBu_r"))
# Get the current figure and axes objects.
fig, ax = plt.gcf(), plt.gca()
# Modifying main plot parameters
ax.tick_params(labelsize=20)
ax.set_xlabel('SHAP value', weight = "bold", fontsize=20)
ax.set_title('The voting model for 50% cell viability prediction', fontsize=22, weight = "bold", pad = 20)
# Get colorbar
cb_ax = fig.axes[1]
# Modifying color bar parameters
plt.gcf().axes[-1].set_aspect(100)
plt.gcf().axes[-1].set_box_aspect(100)
cb_ax.tick_params(labelsize=15)
cb_ax.set_ylabel("MODZ", weight = "bold",fontsize=20)
for tick in ax.get_yticklabels():
    tick.set_fontstyle('italic')
plt.tight_layout()
plt.savefig("Beeplot.tiff", format='tiff',dpi=600)


shap.initjs()
sns.set_theme(style='white')
shap.summary_plot(shap_values_50_value, X_train_val, feature_names=X_train_val.columns, max_display=10,show=False, color="#CD4B47",plot_size=(9, 5),plot_type="bar")
fig, ax = plt.gcf(), plt.gca()
ax.tick_params(labelsize=20)
ax.set_xlabel('Mean(|SHAP value|)', weight = "bold", fontsize=20)
ax.set_title('The voting model for 50% cell viability prediction', fontsize=22, weight = "bold", pad = 20)
plt.tight_layout()
for tick in ax.get_yticklabels():
    tick.set_fontstyle('italic')
plt.tight_layout()
plt.savefig("Barplot.tiff", format='tiff',dpi=600)


Top10_genes_up=[ ]
for i in Top10_genes_up:
    df=pd.concat([X_train_val[i],shap_values_50_train[i]],axis=1)
    sns.jointplot(x=df.iloc[:,0], y=df.iloc[:,1],
                  data=df,
                  color = '#CD4B47',
                  s =70, edgecolor="black",linewidth=1,
                  kind = 'scatter',
                  xlim=(-12,12),
                  ylim=(-0.3,0.15),
                  space = 0.05,
                  height=6,
                  ratio = 3,
                  marginal_kws=dict(bins=20, rug=True,color='#CD4B47')
                  )
    fig, ax = plt.gcf(), plt.gca()
    ax.tick_params(labelsize=20)
    xlabel_name = "MODZ of "+i
    ylabel_name = "SHAP value of "+i
    ax.set_xlabel(xlabel_name, weight = "bold", fontsize=20)
    ax.set_ylabel(ylabel_name, weight = "bold", fontsize=20)
    ax.axhline(y=0, linestyle='--', color='grey',linewidth=3)
    plt.tight_layout()
    file_name='Dotplot_of '+i+'_20240319'

    plt.savefig(file_name+".tiff", format='tiff',dpi=600)

Top10_genes_up=[ ]
for i in Top10_genes_up:
    df=pd.concat([X_train_val[i],shap_values_50_train[i]],axis=1)
    sns.jointplot(x=df.iloc[:,0], y=df.iloc[:,1],
                  data=df,
                  color = '#CBD6E8',
                  s =70, edgecolor="black",linewidth=1,
                  kind = 'scatter',
                  xlim=(-12,12),
                  ylim=(-0.3,0.15),
                  space = 0.05,
                  height=6,
                  ratio = 3,
                  marginal_kws=dict(bins=20, rug=True,color='#002060')
                  )
    fig, ax = plt.gcf(), plt.gca()
    ax.tick_params(labelsize=20)
    xlabel_name = "MODZ of "+i
    ylabel_name = "SHAP value of "+i
    ax.set_xlabel(xlabel_name, weight = "bold", fontsize=20)
    ax.set_ylabel(ylabel_name, weight = "bold", fontsize=20)
    ax.axhline(y=0, linestyle='--', color='grey',linewidth=3)
    plt.tight_layout()
    file_name='Dotplot_of '+i+'_20240319'

    plt.savefig(file_name+".tiff", format='tiff',dpi=600)



