import dill
from random import shuffle
import random
import os
try:
    os.chdir(os.path.join(os.getcwd(), 'data'))
    print(os.getcwd())
except:
    pass

import pandas as pd
from collections import defaultdict
import numpy as np

PATH_RAW_DATA = "raw/"
MED_CSV = 'PRESCRIPTIONS.csv'
DIAG_CSV = 'DIAGNOSES_ICD.csv'
NDC2ATC_CSV = 'ndc2atc_level4.csv'
MAPPING_TXT = 'ndc2rxnorm_mapping.txt'
MULTI_VISITS = 'data_final.pkl'

def etl_med():
    med_df = pd.read_csv(os.path.join(PATH_RAW_DATA, MED_CSV), dtype={'NDC': 'category'})

    # filter raw data
    unused_cols = ['ROW_ID', 'DRUG_TYPE', 'DRUG_NAME_POE', 'DRUG_NAME_GENERIC',
                         'FORMULARY_DRUG_CD', 'GSN', 'PROD_STRENGTH', 'DOSE_VAL_RX',
                         'DOSE_UNIT_RX', 'FORM_VAL_DISP', 'FORM_UNIT_DISP', 'FORM_UNIT_DISP',
                         'ROUTE', 'ENDDATE', 'DRUG']
    med_df.drop(columns=unused_cols, axis = 1, inplace = True)
    med_df = med_df[med_df['NDC'] != '0']
    med_df.fillna(method='pad', inplace=True)
    med_df.dropna(inplace=True)
    med_df.drop_duplicates(inplace=True)
    # format cols
    med_df['STARTDATE'] = pd.to_datetime(med_df['STARTDATE'], format='%Y-%m-%d %H:%M:%S')
    med_df['ICUSTAY_ID'] = med_df['ICUSTAY_ID'].astype('int64')
    med_df.astype({'ICUSTAY_ID': 'int64'})
    med_df.sort_values(by=['SUBJECT_ID', 'HADM_ID','ICUSTAY_ID', 'STARTDATE'], inplace=True)
    med_df = med_df.reset_index(drop=True)
    # fetch first row for each group
    temp = med_df.drop(columns=['NDC'])
    temp = temp.groupby(by=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID']).head([1]).reset_index(drop=True)
    med_df = pd.merge(temp, med_df, on=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'STARTDATE'])
    med_df = med_df.drop(columns=['ICUSTAY_ID'])
    med_df = med_df.drop(columns=['STARTDATE'])
    med_df = med_df.drop_duplicates()
    med_df = med_df.reset_index(drop=True)
    return med_df

def etl_diag():
    dia_df = pd.read_csv(os.path.join(PATH_RAW_DATA, DIAG_CSV))
    # filter
    dia_df.dropna(inplace=True)
    dia_df.drop(columns=['ROW_ID', 'SEQ_NUM'], inplace=True)
    dia_df.drop_duplicates(inplace = True)
    dia_df.sort_values(by=['SUBJECT_ID', 'HADM_ID'], inplace=True)
    dia_df = dia_df.reset_index(drop=True)
    return dia_df

def etl_ndc2atc4(med_df):
    # load data
    with open(os.path.join(PATH_RAW_DATA, MAPPING_TXT), 'r') as file:
        mapping = eval(file.read())
    med_df['RXCUI'] = med_df['NDC'].map(mapping)
    med_df.dropna(inplace=True)

    # mapping
    ndc2atc = pd.read_csv(os.path.join(PATH_RAW_DATA, NDC2ATC_CSV))
    ndc2atc = ndc2atc.drop(columns=['YEAR', 'MONTH', 'NDC'])
    ndc2atc.drop_duplicates(subset=['RXCUI'], inplace=True)
    med_df.drop(index=med_df[med_df['RXCUI'].isin([''])].index, axis=0, inplace=True)
    med_df = med_df.astype({'RXCUI': 'int64'})
    med_df = med_df.reset_index(drop=True)
    med_df = med_df.merge(ndc2atc, on=['RXCUI'])
    med_df.drop(columns=['NDC', 'RXCUI'], inplace=True)
    def substring(x):
        return x[:5]
    med_df['ATC4'] = med_df['ATC4'].map(lambda x: substring(x))
    med_df = med_df.drop_duplicates()
    med_df = med_df.reset_index(drop=True)
    return med_df

def single_visit_filter(med_df):
    filter_df = med_df[['SUBJECT_ID', 'HADM_ID']].groupby(by='SUBJECT_ID')['HADM_ID'].unique().reset_index()
    filter_df['HADM_ID_COUNT'] = filter_df['HADM_ID'].map(lambda x: len(x))
    filter_df = filter_df[(filter_df['HADM_ID_COUNT'] >= 1) & (filter_df['HADM_ID_COUNT'] < 2)]
    filter_df = filter_df.reset_index(drop = True)
    # merge
    med_df = med_df.merge(filter_df[['SUBJECT_ID']], on='SUBJECT_ID', how='inner')
    med_df = med_df.reset_index(drop = True)
    return med_df

def diag_filter(dia_df):
    count_df = dia_df.groupby(by=['ICD9_CODE']).size().reset_index()
    count_df = count_df.rename(columns={0: 'COUNT'}).sort_values(by=['COUNT'], ascending=False).reset_index(drop=True)

    dia_df = dia_df[dia_df['ICD9_CODE'].isin(count_df.loc[:128, 'ICD9_CODE'])]
    dia_df = dia_df.reset_index(drop = True)
    return dia_df

def visits_filter(visits):
    drops = []
    for subject_id in visits['SUBJECT_ID'].unique():
        subject = visits[visits['SUBJECT_ID'] == subject_id]
        for index, row in subject.iterrows():
            dx_len = len(list(row['ICD9_CODE']))
            rx_len = len(list(row['ATC4']))
            if dx_len < 2 or dx_len > np.inf:
                drops.append(subject_id)
                break
            if rx_len < 2 or rx_len > np.inf:
                drops.append(subject_id)
                break
    visits.drop(index=visits[visits['SUBJECT_ID'].isin(drops)].index, axis=0, inplace=True)
    visits = visits.reset_index(drop = True)
    return visits

def etl_med_diag():
    # get etled med and diag
    med_df = etl_med()
    med_df = etl_ndc2atc4(med_df)
    med_df = single_visit_filter(med_df)
    dia_df = etl_diag()
    dia_df = diag_filter(dia_df)

    # get unique visits
    unique_med_df = med_df[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
    unique_dia_df = dia_df[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
    unique_visit = unique_med_df.merge(unique_dia_df, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    med_df = med_df.merge(unique_visit, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    dia_df = dia_df.merge(unique_visit, on=['SUBJECT_ID', 'HADM_ID'], how='inner')

    # group 
    med_df = med_df.groupby(by=['SUBJECT_ID', 'HADM_ID'])['ATC4'].unique().reset_index()
    dia_df = dia_df.groupby(by=['SUBJECT_ID', 'HADM_ID'])['ICD9_CODE'].unique().reset_index()
    med_df['ATC4'] = med_df['ATC4'].map(lambda x: list(x))
    dia_df['ICD9_CODE'] = dia_df['ICD9_CODE'].map(lambda x: list(x))
    etl_visits = dia_df.merge(med_df, on=['SUBJECT_ID', 'HADM_ID'], how='inner')

    # filter visits
    filter_visits = visits_filter(etl_visits)

    # save unique codes
    diag_codes = filter_visits['ICD9_CODE'].values
    med_codes = filter_visits['ATC4'].values
    unique_diag_codes = set([j for i in diag_codes for j in list(i)])
    unique_med_codes = set([j for i in med_codes for j in list(i)])

    return filter_visits, unique_diag_codes, unique_med_codes

def load_multi_visits():
    multi_visits = pd.read_pickle(os.path.join(PATH_RAW_DATA, MULTI_VISITS))
    multi_visits.rename(columns={'NDC': 'ATC4'}, inplace=True)
    multi_visits.drop(columns=['PRO_CODE', 'NDC_Len'], axis=1, inplace=True)

    # unique code save
    diag_codes = multi_visits['ICD9_CODE'].values
    med_codes = multi_visits['ATC4'].values
    unique_diag_codes = set([j for i in diag_codes for j in list(i)])
    unique_med_codes = set([j for i in med_codes for j in list(i)])

    return multi_visits, unique_diag_codes, unique_med_codes

def split_dataset():
    dataset = pd.read_pickle('data-multi-visit.pkl')
    sample_id = dataset['SUBJECT_ID'].unique()
    random_number = [i for i in range(len(sample_id))]

    train_id = sample_id[random_number[:int(len(sample_id)*2/3)]]
    eval_id = sample_id[random_number[int(
        len(sample_id)*2/3): int(len(sample_id)*5/6)]]
    test_id = sample_id[random_number[int(len(sample_id)*5/6):]]

    def save_file(list_data, file_name):
        with open(file_name, 'w') as file:
            for item in list_data:
                file.write(str(item) + '\n')

    save_file(train_id, 'train-id.txt')
    save_file(eval_id, 'eval-id.txt')
    save_file(test_id, 'test-id.txt')

    print('train size: %d, eval size: %d, test size: %d' %
          (len(train_id), len(eval_id), len(test_id)))

def main():
    # load data
    print('load single visits and multi visits')
    single_visits, single_diag, single_med = etl_med_diag()
    multi_visits, multi_diag, multi_med = load_multi_visits()

    # save data
    unique_diag_codes = single_diag | multi_diag
    unique_med_codes = single_med | multi_med
    with open('dx-vocab.txt', 'w') as file:
        for code in unique_diag_codes:
            file.write(code + '\n')
    with open('rx-vocab.txt', 'w') as file:
        for code in unique_med_codes:
            file.write(code + '\n')

    with open('dx-vocab-multi.txt', 'w') as file:
        for code in multi_diag:
            file.write(code + '\n')
    with open('rx-vocab-multi.txt', 'w') as file:
        for code in multi_med:
            file.write(code + '\n')

    single_visits.to_pickle('data-single-visit.pkl')
    multi_visits.to_pickle('data-multi-visit.pkl')

    # split data and save
    split_dataset()


if __name__ == '__main__':
    main()