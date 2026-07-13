import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def to_binary(status):
    if status in ['0', 'X', 'C']:
        return 1
    return 0

def preprocess_data():
    applicant_df = pd.read_csv('application_record.csv')
    credit_df = pd.read_csv('credit_record.csv')
    
    applicant_df.drop_duplicates(subset=[
        'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN',
        'AMT_INCOME_TOTAL', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
        'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'DAYS_BIRTH',
        'DAYS_EMPLOYED', 'FLAG_MOBIL', 'FLAG_WORK_PHONE', 'FLAG_PHONE',
        'FLAG_EMAIL', 'OCCUPATION_TYPE', 'CNT_FAM_MEMBERS'
    ], keep='first', inplace=True)
    
    if 'OCCUPATION_TYPE' in applicant_df.columns:
        applicant_df.drop('OCCUPATION_TYPE', axis=1, inplace=True)
        
    applicant_df['DAYS_BIRTH'] = applicant_df['DAYS_BIRTH'].abs()
    applicant_df['DAYS_EMPLOYED'] = applicant_df['DAYS_EMPLOYED'].abs()
    
    applicant_df['FAMILY_DEPENDENCY'] = applicant_df['CNT_FAM_MEMBERS'] + applicant_df['CNT_CHILDREN']
    
    le = LabelEncoder()
    categorical_cols = [
        'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_INCOME_TYPE', 
        'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE'
    ]
    for col in categorical_cols:
        applicant_df[col] = le.fit_transform(applicant_df[col].astype(str))
        
    credit_df['STATUS_BIN'] = credit_df['STATUS'].astype(str).apply(to_binary)
    
    credit_grouped = credit_df.groupby('ID').agg(
        open_month=('MONTHS_BALANCE', 'min'),
        end_month=('MONTHS_BALANCE', 'max'),
        STATUS_BIN=('STATUS_BIN', 'min')
    ).reset_index()
    credit_grouped['window'] = credit_grouped['end_month'] - credit_grouped['open_month']
    
    final_df = applicant_df.merge(credit_grouped, how='left', on='ID')
    final_df.fillna(0, inplace=True)
    
    final_df.to_csv('processed_dataset.csv', index=False)
    print("Preprocessing complete. Output saved to 'processed_dataset.csv'")

if __name__ == "__main__":
    preprocess_data()
