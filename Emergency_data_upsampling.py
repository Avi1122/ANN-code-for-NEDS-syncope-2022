#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pyreadstat
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder

maindf_concat = pickle.load(open("maindf_concat_aug22.pkl", 'rb'))
maindf_concat2 = pickle.load(open("maindf_concat2_aug22.pkl", 'rb'))


maindf_concat2.reset_index(drop =True, inplace = True)
df = pd.DataFrame({'freq': maindf_concat2.LOS_IP.tolist()})
df.groupby('freq', as_index=False).size().plot(kind='bar')
plt.show()

def up_2(df_f):

    df_majority = df_f[df_f.label==0]
    df_minority1 = df_f[df_f.label==1]
#     df_minority3 = df_f[df_f.label==3]

    # Upsample minority class
    df_minority_upsampled1 = resample(df_minority1, 
                                     replace=True,     # sample with replacement
                                     n_samples=len(df_majority),    # to match majority class
                                     random_state=42) # reproducible results
    df_upsampled = pd.concat([df_majority, df_minority_upsampled1])

    df_upsampled = df_upsampled.sample(frac=1)
    return df_upsampled

Thresholds = [0,1,2,4,7]

for THRES in Thresholds:
    def change_los(x):
        if x[0] == 1.:
            return 1.
        else:
            if x[1] <= THRES:
                return 0.
            else:
                return 1.

    Y_multi1 = maindf_concat2[['DIED_VISIT','LOS_IP']].values
    Y_multi1 = np.apply_along_axis(change_los, 1, Y_multi1)


    maindf_concat_model2 = maindf_concat2.iloc[:,:72]


    df_f2 = maindf_concat_model2.copy(deep =True)

    df_f2['label'] = Y_multi1

    df_upsampled2 = up_2(df_f2)


    le = LabelEncoder()
    Y_out2 = le.fit_transform(df_upsampled2.label)

    X2 = df_upsampled2.iloc[:,:-1].values
    Y2= pd.get_dummies(Y_out2).values
    X2 = np.asarray(X2).astype('float32')
    Y2 = np.asarray(Y2).astype('float32')

    pickle.dump(X2, open("X_aug22_small_up_m_"+str(THRES)+"thr.pkl", 'wb'), protocol=4)
    pickle.dump(Y2, open("Y_aug22_small_up_m_"+str(THRES)+"thr.pkl", 'wb'), protocol=4)
    pickle.dump(df_upsampled2, open("df_upsampled_aug22_m_"+str(THRES)+"thr.pkl", 'wb'), protocol=4)

