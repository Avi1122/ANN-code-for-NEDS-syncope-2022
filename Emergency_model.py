#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pyreadstat
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve

import tensorflow as tf
import sys
import os
print(f"Tensorflow Version: {tf.__version__}")
print(f"Pandas Version: {pd.__version__}")
print(f"Numpy Version: {np.__version__}")
print(f"System Version: {sys.version}")

from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l1_l2, l1, l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# from keras.callbacks import EarlyStopping
# from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import backend as K

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import platform
uname = platform.node()
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
machine = int(uname.split(".")[0][-1])
if machine == 8:
    os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import logging
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except Exception as err:
        logging.error(err)


global mirrored_strategy
global BATCH_SIZE_PER_REPLICA, EPOCHS, GLOBAL_BATCH_SIZE, BUFFER_SIZE, CURR_EPOCH



if tf.config.list_physical_devices('GPU'):

    mirrored_strategy = tf.distribute.MirroredStrategy(devices=[ "/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3"],
                                                       cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())


def simple_ann_model(OUTPUT_CHANNELS):
    inputs = Input(shape=(encoded_items.shape[1]))
    x = Dense(64, activation='relu',kernel_regularizer= l1_l2(0.01), name='dense_a')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = Dropout(0.05)(x)
    x = Dense(32, activation='relu',kernel_regularizer= l1_l2(0.01), name='dense_b')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = Dropout(0.05)(x)
    x = Dense(16, activation='relu',kernel_regularizer= l1_l2(0.01), name='dense_c')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = Dropout(0.05)(x)
#     x = Dense(64, activation='relu',kernel_regularizer= l1_l2(0.01), name='dense_d')(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = Dropout(0.05)(x)

# #     x = Dense(32, activation='relu',kernel_regularizer= l1_l2(0.01), name='dense_e')(x)
# #     x = tf.keras.layers.BatchNormalization()(x)
# #     x = Dropout(0.1)(x)
#     x = Dense(8, activation='relu',kernel_regularizer= l1_l2(0.001), name='dense_f')(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = Dropout(0.05)(x)
#     out =Dense(OUTPUT_CHANNELS, activation='softmax')(x)
    out =Dense(OUTPUT_CHANNELS, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=out)
    return model


if __name__ == '__main__':
    THRES = sys.argv[1]
    X = pickle.load(open("X_aug22_small_up_m_"+str(THRES)+"thr.pkl", 'rb'))
    Y = pickle.load(open("Y_aug22_small_up_m_"+str(THRES)+"thr.pkl", 'rb'))

    i_dims = X.shape[1]


    maindf_concat4_small = pickle.load( open("df_upsampled_aug22_m_"+str(THRES)+"thr.pkl", 'rb'))

    maindf_concat4_small.columns = [i.replace(".0","") for i in maindf_concat4_small.columns]

    for i in maindf_concat4_small.columns:
        maindf_concat4_small[i] = maindf_concat4_small[i].astype(float)

    cols = ["Congestive Heart Failure", "Cardiac Arrhythmias", "Valvular Disease","Pulmonary Circulation Disorders", 
             "Peripheral Vascular Disorders", "Hypertension, Uncomplicated",
             "Paralysis","Other Neurological Disorders", "Chronic Pulmonary Disease", "Diabetes, Uncomplicated", "Diabetes, Complicated",
             "Hypothyroidism", "Renal Failure", "Liver Disease", "Peptic Ulcer Disease Excluding Bleeding","AIDS/HIV",
            "Lymphoma", "Metastatic Cancer", "Solid Tumor Without Metastasis", "Rheumatoid Arthritis/Collagen Vascular", 
             "Coagulopathy", "Obesity", "Weight Loss", "Fluid and Electrolyte  Disorders", "Blood Loss Anemia",
            "Deficiency Anemia", "Alcohol Abuse", "Drug Abuse", "Psychoses", "Depression", "Hypertension, Complicated",
           "age_less_than_40","age_40_to_60","age_60_to_75","age_75_above", "female_no", "female_yes", "female_unknown",
           "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November",
           "December","month_nan", "Weeken_no", "Weekend_yes", "HOSP_CONTROL_0", "HOSP_CONTROL_1", "HOSP_CONTROL_2", "HOSP_CONTROL_3",
           "HOSP_CONTROL_4", "HOSP_REGION_1", "HOSP_REGION_2","HOSP_REGION_3", "HOSP_REGION_4", "HOSP_UR_TEACH_0",
           "HOSP_UR_TEACH_1", "HOSP_UR_TEACH_2", "HOSP_URCAT4_1","HOSP_URCAT4_2", "HOSP_URCAT4_3", "HOSP_URCAT4_4",
           "HOSP_URCAT4_7", "HOSP_URCAT4_8", "HOSP_URCAT4_9", "label"]


    maindf_concat4_small.columns  = cols


    maindf_concat4_small.columns = [i.replace("Diabetes,","Diabetes") for i in maindf_concat4_small.columns]
    maindf_concat4_small.columns = [i.replace("Hypertension,","Hypertension") for i in maindf_concat4_small.columns]

    maindf_concat4_small.columns = [i.replace(" ","_") for i in maindf_concat4_small.columns]
    maindf_concat4_small.columns = [i.replace("/","_or_") for i in maindf_concat4_small.columns]

    maindf_concat4_small.drop(maindf_concat4_small.columns[[34, 37, 49, 50,57,61,64,71]], axis=1, inplace=True) #Dropping dummy variables


    base = ''
    for i in list(maindf_concat4_small.columns[:-1]):
        base = base + " + "+ i


    smf_model = smf.logit(formula = "label ~ "+ str(base[3:]), data= maindf_concat4_small).fit_regularized()


    results_summary = smf_model.summary()

    results_as_html = results_summary.tables[1].as_html()


    stats_df = pd.read_html(results_as_html, header=0, index_col=0)[0]

    stats_df.to_excel('stats_df_'+str(THRES)+'.xlsx', engine='xlsxwriter')  


    X_train, X_test,y_train, y_test = train_test_split(maindf_concat4_small.iloc[:,:-1].values, Y, test_size=0.20, random_state=42, shuffle = True, stratify = Y)

    model = LogisticRegression(solver='liblinear', random_state=0)

    model.fit(X_train, y_train[:,1])

    y_pred = model.predict_proba(X_test)


    # roc curve
    fpr = dict()
    tpr = dict()
    cs =["Short_stay (≤ 48 hours)", "Long_stay (> 48 hours)"]
    operating_points = []
    for i in range(len(cs)):
        fpr[i], tpr[i], thresholds = roc_curve(y_test[:, i],y_pred[:, i])
        optimal_idx = np.argmax(tpr[i] - fpr[i])
        optimal_threshold = thresholds[optimal_idx]
        operating_points.append(optimal_threshold)
        #print("Threshold value is:", optimal_threshold)
        auc = roc_auc_score(y_test[:, i],y_pred[:, i])
        print(auc)
        plt.plot(fpr[i], tpr[i], label='{} (auc = {})'.format(cs[i], round(auc,5)))

    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.legend(loc="best")
    plt.title("ROC curve")
    plt.savefig('model_logit'+str(THRES)+'.jpg', dpi=300)
    plt.show()

    labels =[ "Short_stay", "Long_stay"]

    conf_mat_dict={}
    thresholds = operating_points


    for label_col in range(len(labels)):
        print(labels[label_col])
        y_true_label = y_test[:, label_col]
        y_pred_label = y_pred[:, label_col]
        y_pred_label = np.where(y_pred_label>thresholds[label_col], 1., 0.)
        print(classification_report(y_true_label,y_pred_label))
        conf_mat_dict[labels[label_col]] = confusion_matrix(y_pred=y_pred_label, y_true=y_true_label)


    for label, matrix in conf_mat_dict.items():
        print("Confusion matrix for label {}:".format(label))
        print(matrix)


    cm = confusion_matrix(y_test[:,1], model.predict(X_test))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cm)
    ax.grid(False)
    ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
    ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
    ax.set_ylim(1.5, -0.5)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
    plt.show()


    encoded_items = X_train

    with mirrored_strategy.scope():
        model = simple_ann_model(2)

        opt = tf.keras.optimizers.Adam(
            learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
            name='Adam', decay= 1e-4, clipvalue=0.5
        )
    #     opt =tf.keras.optimizers.SGD(0.00001)
        model.compile(optimizer=opt,
                      loss='binary_crossentropy',
                      metrics=[tf.keras.metrics.AUC(), 'accuracy'])

    #loss='mean_squared_error',
    ES = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.0001, verbose=1, patience=10)
    Reduce_LR = ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.9, patience=15, min_lr=1e-20, verbose=1, cooldown=3)

    model_history = model.fit(encoded_items, y_train, 
                              verbose=2, 
                              validation_split=0.20,
                              epochs=750, 
                              batch_size = 8192,
                              shuffle=True,
                             callbacks=[ES, Reduce_LR])

    y_pred = model.predict( X_test)

    means = [np.mean(y_pred[:,0]),np.mean(y_pred[:,1])]

    min_maxes = [(np.min(y_pred[:,0])+np.min(y_pred[:,0]))/2,
             (np.min(y_pred[:,0])+np.min(y_pred[:,1]))/2]


    # roc curve
    fpr = dict()
    tpr = dict()
    cs =["Short_stay (≤ 48 hours)", "Long_stay (> 48 hours)"]
    operating_points = []
    for i in range(len(cs)):
        fpr[i], tpr[i], thresholds = roc_curve(y_test[:, i],y_pred[:, i])
        optimal_idx = np.argmax(tpr[i] - fpr[i])
        optimal_threshold = thresholds[optimal_idx]
        operating_points.append(optimal_threshold)
        #print("Threshold value is:", optimal_threshold)
        auc = roc_auc_score(y_test[:, i],y_pred[:, i])
        print(auc)
        plt.plot(fpr[i], tpr[i], label='{} (auc = {})'.format(cs[i], round(auc,5)))

    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.legend(loc="best")
    plt.title("ROC curve")
    plt.savefig('model_review_'+str(THRES)+'.jpg', dpi=300)
    plt.show()


    labels =[ "Short_stay", "Long_stay"]

    conf_mat_dict={}
    thresholds = operating_points


    for label_col in range(len(labels)):
        print(labels[label_col])
        y_true_label = y_test[:, label_col]
        y_pred_label = y_pred[:, label_col]
        y_pred_label = np.where(y_pred_label>thresholds[label_col], 1., 0.)
        print(classification_report(y_true_label,y_pred_label))
        conf_mat_dict[labels[label_col]] = confusion_matrix(y_pred=y_pred_label, y_true=y_true_label)


    for label, matrix in conf_mat_dict.items():
        print("Confusion matrix for label {}:".format(label))
        print(matrix)

