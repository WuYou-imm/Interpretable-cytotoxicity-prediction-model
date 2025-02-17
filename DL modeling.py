import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, roc_curve
import torch
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from pytorch_widedeep import Trainer
from pytorch_widedeep.preprocessing import WidePreprocessor, TabPreprocessor
from pytorch_widedeep.models import Wide, TabMlp, WideDeep
from pytorch_widedeep.metrics import Accuracy
from pytorch_widedeep import Trainer
from pytorch_widedeep.callbacks import EarlyStopping
from pytorch_widedeep.metrics import Accuracy
from pytorch_widedeep.models import TabResnet

def get_data(cv):
    Train_metadata = pd.read_csv('Y_train20230610.csv')
    Test_metadata = pd.read_csv('Y_test20230610.csv')

    Train_signature = pd.read_csv('X_train20230610.csv', index_col=0)
    Test_signature = pd.read_csv('X_test20230610.csv', index_col=0)

    Y_train_val = Train_metadata[cv]
    Y_test = Test_metadata[cv]

    X_train_val = Train_signature
    X_test = Test_signature

    X_all = pd.concat([Train_signature, Test_signature], axis=0)
    Y_all = pd.concat([Y_train_val, Y_test], axis=0)
    return X_train_val, Y_train_val, X_test, Y_test, X_all, Y_all

X_train, y_train, X_test, y_test, X_all, Y_all = get_data("cv0.5")
# define wide, crossed, embedding and continuous columns, and target
cont_cols =X_train.columns
target = y_train.values

tab_preprocessor = TabPreprocessor(continuous_cols=cont_cols)
X_tab = tab_preprocessor.fit_transform(X_train)
deeptabular = TabResnet(
    blocks_dims=[16, 8],
    blocks_dropout=0.05,
    column_idx=tab_preprocessor.column_idx,
    #embed_input=tab_preprocessor.embeddings_input,
    continuous_cols=cont_cols,
)
model = WideDeep(deeptabular=deeptabular)

# Train
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 创建 EarlyStopping 回调，早停条件为 5 个 epoch 内没有改善
early_stopping = EarlyStopping(monitor='train_loss',patience=5, min_delta=0)

# 创建 Trainer 对象，指定目标为 "binary"，添加 Accuracy 指标
trainer = Trainer(model,
                   objective="binary",
                   optimizer=optimizer,
                   metrics=[Accuracy()],
                   callbacks=[early_stopping])  # 将 EarlyStopping 回调添加到训练器

# 训练模型，指定训练集和目标，设置 epoch 数量为 50，批量大小为 256
# 验证集比例设置为 10%，通过 validation_split 参数
trainer.fit(X_tab=X_tab,
            target=target,
            n_epochs=50,
            batch_size=256,
            validation_split=0.1)
# predict

X_tab_te = tab_preprocessor.transform(X_test)
preds = trainer.predict(X_tab=X_tab_te)
# 计算评估指标

y_test_np = y_test.values.astype(np.float32).reshape(-1)
tabresnet_accuracy = accuracy_score(y_test_np, preds)
tabresnet_accuracy = accuracy_score(y_test_np, preds)
tabresnet_roc_auc = roc_auc_score(y_test_np, preds)
tabresnet_precision = precision_score(y_test_np, preds)
tabresnet_recall = recall_score(y_test_np, preds)
tabresnet_f1 = f1_score(y_test_np, preds)
report = classification_report(y_test_np,preds)

# 打印分类报告
print(report)

# 计算混淆矩阵
conf_matrix = confusion_matrix(y_test_np, preds)
TN, FP, FN, TP = conf_matrix.ravel()
sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

# 打印评估指标
print("tabresnet Evaluation Metrics:")
print(f"Accuracy: {tabresnet_accuracy:.4f}")
print(f"AUROC: {tabresnet_roc_auc:.4f}")
print(f"Precision: {tabresnet_precision:.4f}")
print(f"Recall: {tabresnet_recall:.4f}")
print(f"F1 Score: {tabresnet_f1:.4f}")
print(f"Sensitivity: {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, roc_curve, classification_report
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
X_train, y_train, X_test, y_test, X_all, Y_all = get_data("cv0.5")
MLP_model = Sequential([
    Dense(512, activation='relu', input_shape=(12328,)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
MLP_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(patience=5)
MLP_history = MLP_model.fit(X_train, y_train, epochs=50, validation_split=0.1, callbacks=[early_stopping])
y_pred_MLP_prob = MLP_model.predict(X_test)
y_pred_MLP_class = np.where(y_pred_MLP_prob > 0.5, 1, 0)
MLP_accuracy = accuracy_score(y_test, y_pred_MLP_class)
MLP_roc_auc = roc_auc_score(y_test, y_pred_MLP_prob)
MLP_precision = precision_score(y_test, y_pred_MLP_class)
MLP_recall = recall_score(y_test, y_pred_MLP_class)
MLP_f1 = f1_score(y_test, y_pred_MLP_class)
fpr, tpr, _ = roc_curve(y_test, y_pred_MLP_prob)
report = classification_report(y_test, y_pred_MLP_class)
print(report)
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_pred_MLP_class)
TN = conf_matrix[0][0]
FP = conf_matrix[0][1]
FN = conf_matrix[1][0]
TP = conf_matrix[1][1]
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)
print("MLP Evaluation Metrics:")
print(f"Accuracy: {MLP_accuracy}")
print(f"AUROC: {MLP_roc_auc}")
print(f"Precision: {MLP_precision}")
print(f"Recall: {MLP_recall}")
print(f"F1 Score: {MLP_f1}")
print(f"Specificity: {specificity:.4f}")

import sklearn
# 卷积神经网络 (CNN)
X_train, y_train, X_test, y_test, X_all, Y_all = get_data("cv0.5")
cnn_model = models.Sequential([
    layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=(12328, 1)),
    layers.MaxPooling1D(pool_size=2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

cnn_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
# 假设 X_train 和 X_test 原本是 pandas DataFrame 对象
X_train_np = X_train.values.reshape(-1, 12328, 1)
X_test_np = X_test.values.reshape(-1, 12328, 1)

# 现在可以正常使用 cnn_model.fit 和 cnn_model.predict
cnn_history = cnn_model.fit(X_train_np, y_train, epochs=50, validation_split=0.1, callbacks=[EarlyStopping(patience=5)])

# 评估CNN模型
y_pred_cnn_prob = cnn_model.predict(X_test_np)
y_pred_cnn_class = np.where(y_pred_cnn_prob > 0.5, 1, 0)

cnn_accuracy = accuracy_score(y_test, y_pred_cnn_class)
cnn_roc_auc = roc_auc_score(y_test, y_pred_cnn_prob)
cnn_precision = precision_score(y_test, y_pred_cnn_class)
cnn_recall = recall_score(y_test, y_pred_cnn_class)
cnn_f1 = f1_score(y_test, y_pred_cnn_class)
fpr, tpr, _ = roc_curve(y_test, y_pred_cnn_prob)
print(report)
conf_matrix = confusion_matrix(y_test, y_pred_cnn_class)
TN = conf_matrix[0][0]
FP = conf_matrix[0][1]
FN = conf_matrix[1][0]
TP = conf_matrix[1][1]
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)
print("CNN Evaluation Metrics:")
print(f"Accuracy: {cnn_accuracy}")
print(f"AUROC: {cnn_roc_auc}")
print(f"Precision: {cnn_precision}")
print(f"Recall: {cnn_recall}")
print(f"F1 Score: {cnn_f1}")
print(f"Specificity: {specificity:.4f}")

X_train,y_train,X_test,y_test,X_all,Y_all=get_data("cv0.5")
x_train,X_val,y_train,Y_val = train_test_split(X_train, y_train, test_size=0.1,
                                                 stratify=y_train,
                                                 shuffle=True,
                                                 random_state=123)
tabnet_params = dict(
    n_d =1024,
    n_a = 16,
    n_steps = 4,
    gamma = 1.5,
    lambda_sparse = 1e-6,
    optimizer_fn = torch.optim.Adam,
    optimizer_params = dict(lr = 0.001, weight_decay = 1e-5),
    momentum = 0.95,
    mask_type = "entmax",
    seed = 0
)
clf = TabNetClassifier()
clf.fit(
    x_train.values, y_train.values,
    eval_set=[(X_val.values, Y_val.values)],
    eval_name=['val'],
    eval_metric=['accuracy'],
    max_epochs=500,
    patience=5
)
preds = clf.predict(X_test.values)
conf_matrix = confusion_matrix(y_test, preds)
TabNet_accuracy = accuracy_score(y_test, preds)
TabNet_roc_auc = roc_auc_score(y_test, preds)
TabNet_precision = precision_score(y_test, preds)
TabNet_recall = recall_score(y_test, preds)
TabNet_f1 = f1_score(y_test, preds)
fpr, tpr, _ = roc_curve(y_test, preds)
print(report)
TN = conf_matrix[0][0]
FP = conf_matrix[0][1]
FN = conf_matrix[1][0]
TP = conf_matrix[1][1]
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)
print("TabNet Evaluation Metrics:")
print(f"Accuracy: {TabNet_accuracy}")
print(f"AUROC: {TabNet_roc_auc}")
print(f"Precision: {TabNet_precision}")
print(f"Recall: {TabNet_recall}")
print(f"F1 Score: {TabNet_f1}")
print(f"Specificity: {specificity:.4f}")