#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pyreadstat

df =  pd.read_spss('Syncope_primary diagnosis_2016_to_2019.sav')

plt.hist(df['AGE'].values, density=True, bins=100)  # density=False would make counts
plt.ylabel('Frequency')
plt.xlabel('Age');


def age_cal(x):
    if x <=40:
        return 1.0
    elif 40<x<=60:
        return 2.0
    elif 60<x<=75:
        return 3.0
    else:
        return 4.0



df['age_cluster'] = df['AGE'].apply(age_cal)


cols4 = ['ynel1', 'ynel2', 'ynel3', 'ynel4',
       'ynel5', 'ynel6', 'ynel7', 'ynel8', 'ynel9', 'ynel10', 'ynel11',
       'ynel12', 'ynel13', 'ynel14', 'ynel15', 'ynel16', 'ynel17',
       'ynel18', 'ynel19', 'ynel20', 'ynel21', 'ynel22', 'ynel23',
       'ynel24', 'ynel25', 'ynel26', 'ynel27', 'ynel28', 'ynel29',
       'ynel30', 'ynel31','AMONTH', 'AWEEKEND','age_cluster','FEMALE',
        'HOSP_CONTROL', 'HOSP_REGION','HOSP_UR_TEACH', 'HOSP_URCAT4','TOTAL_EDVisits','DIED_VISIT','LOS_IP']

icd10_df_main =  df[cols4]


YNELS = ['ynel1', 'ynel2', 'ynel3', 'ynel4',
       'ynel5', 'ynel6', 'ynel7', 'ynel8', 'ynel9', 'ynel10', 'ynel11',
       'ynel12', 'ynel13', 'ynel14', 'ynel15', 'ynel16', 'ynel17',
       'ynel18', 'ynel19', 'ynel20', 'ynel21', 'ynel22', 'ynel23',
       'ynel24', 'ynel25', 'ynel26', 'ynel27', 'ynel28', 'ynel29',
       'ynel30', 'ynel31']



def change_d(x):
    if x == 'Absent':
        return 0.0
    else:
        return 1.0


for i in YNELS:
    icd10_df_main[i] = icd10_df_main[i].apply(change_d)



icd10_df_main['LOS_IP'] = icd10_df_main['LOS_IP'].fillna(0)


icd10_df_main['AMONTH'] = icd10_df_main['AMONTH'].fillna(13)
icd10_df_main['FEMALE'] = icd10_df_main['FEMALE'].fillna(2)

icd10_df_main.drop_duplicates(keep = 'last', inplace = True)

icd10_df_main.reset_index(inplace = True, drop = True)


plt.hist(icd10_df_main['LOS_IP'].values, density=True, bins=100)  # density=False would make counts
plt.ylabel('Frequency')
plt.xlabel('Length of Stay');




cols_rel = ['ynel1', 'ynel2', 'ynel3', 'ynel4',
       'ynel5', 'ynel6', 'ynel7', 'ynel8', 'ynel9', 'ynel10', 'ynel11',
       'ynel12', 'ynel13', 'ynel14', 'ynel15', 'ynel16', 'ynel17',
       'ynel18', 'ynel19', 'ynel20', 'ynel21', 'ynel22', 'ynel23',
       'ynel24', 'ynel25', 'ynel26', 'ynel27', 'ynel28', 'ynel29',
       'ynel30', 'ynel31','AMONTH', 'AWEEKEND','age_cluster','FEMALE',
        'HOSP_CONTROL', 'HOSP_REGION','HOSP_UR_TEACH', 'HOSP_URCAT4','LOS_IP']


nf = icd10_df_main[cols_rel].assign(combined=icd10_df_main[cols_rel].agg(list, axis=1))

def fl_c(x):
    x = [int(a) for a in x]
    x = [str(a) for a in x]
    
    return ''.join(x)

nf['combined']= nf['combined'].apply(fl_c)
val = nf['combined'].value_counts().index
cnt = nf['combined'].value_counts().to_list()

maj_sub = val[:48344]
nons = nf[~(nf.combined.isin(maj_sub))].index
nf2 =  nf[nf.combined.isin(maj_sub)]

def index_sub(x):
    if x in maj_sub:
        return list(nf2[nf2 == x].index[:2])
    else:
        return list(nf2[nf2== x].index)

trues = nf2.groupby('combined').head(2).index

final_inx = list(trues)+list(nons)


icd10_df_main_fulls = icd10_df_main[icd10_df_main.index.isin(final_inx)]
icd10_df_main_fulls.reset_index(inplace = True, drop = True)

age_values = pd.get_dummies(icd10_df_main_fulls.values[:,33], prefix='age_val')
Female_values = pd.get_dummies(icd10_df_main_fulls.values[:,34], prefix='female_val')
month = pd.get_dummies(icd10_df_main_fulls.values[:,31], prefix='month_val')
week = pd.get_dummies(icd10_df_main_fulls.values[:,32], prefix='week_val')
hosp_cont =  pd.get_dummies(icd10_df_main_fulls.values[:,35], prefix='hc_val')
hosp_reg =  pd.get_dummies(icd10_df_main_fulls.values[:,36], prefix='hr_val')
hosp_urt =  pd.get_dummies(icd10_df_main_fulls.values[:,37], prefix='hur_t_val')
hosp_urcat =  pd.get_dummies(icd10_df_main_fulls.values[:,38], prefix='hurcat_val')

icd10_df_main_fulls.TOTAL_EDVisits = (icd10_df_main_fulls.TOTAL_EDVisits - icd10_df_main_fulls.TOTAL_EDVisits.mean()) / (icd10_df_main_fulls.TOTAL_EDVisits.max() - icd10_df_main_fulls.TOTAL_EDVisits.min())



firsts = icd10_df_main_fulls.iloc[:,:31]
lasts = icd10_df_main_fulls.iloc[:,-3:]



maindf_concat = pd.concat([firsts,age_values, Female_values,month, week, hosp_cont, hosp_reg, hosp_urt, hosp_urcat, lasts], axis=1)
maindf_concat['DIED_VISIT'] = maindf_concat['DIED_VISIT'].replace(2, 1)

maindf_concat = maindf_concat[maindf_concat['DIED_VISIT'].notnull()]
maindf_concat.loc[((maindf_concat['DIED_VISIT']==1) & (maindf_concat['LOS_IP'].isna())), 'LOS_IP'] = 1

maindf_concat2 = maindf_concat[maindf_concat['LOS_IP'].notnull()]


pickle.dump(maindf_concat, open("maindf_concat_aug22.pkl", 'wb'), protocol=4)
pickle.dump(maindf_concat2, open("maindf_concat2_aug22.pkl", 'wb'), protocol=4)

maindf_concat = pickle.load(open("maindf_concat_aug22.pkl", 'rb'))
maindf_concat2 = pickle.load(open("maindf_concat2_aug22.pkl", 'rb'))

