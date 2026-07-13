import pandas as pd
import numpy as np
import os

def generate_datasets(num_applicants=150000, output_dir='.'):
    """
    Generates synthetic datasets for Credit Card Approval Prediction.
    Produces two files: application_record.csv and credit_record.csv.
    """
    np.random.seed(42)
    
    # 1. Generate Application Records
    applicant_ids = np.arange(5000000, 5000000 + num_applicants)
    genders = np.random.choice(['M', 'F'], num_applicants)
    own_car = np.random.choice(['Y', 'N'], num_applicants)
    own_realty = np.random.choice(['Y', 'N'], num_applicants)
    children = np.random.poisson(0.5, num_applicants)
    income_total = np.random.lognormal(mean=11.5, sigma=0.5, size=num_applicants).round(2)
    
    income_types = ['Working', 'Commercial associate', 'Pensioner', 'State servant', 'Student']
    income_type = np.random.choice(income_types, num_applicants, p=[0.5, 0.25, 0.15, 0.09, 0.01])
    
    edu_types = ['Secondary / secondary special', 'Higher education', 'Incomplete higher', 'Lower secondary', 'Academic degree']
    education_type = np.random.choice(edu_types, num_applicants, p=[0.7, 0.2, 0.05, 0.04, 0.01])
    
    family_status = ['Married', 'Single / not married', 'Civil marriage', 'Separated', 'Widow']
    family_type = np.random.choice(family_status, num_applicants, p=[0.6, 0.2, 0.1, 0.05, 0.05])
    
    housing_types = ['House / apartment', 'With parents', 'Municipal apartment', 'Rented apartment', 'Office apartment', 'Co-op apartment']
    housing_type = np.random.choice(housing_types, num_applicants, p=[0.85, 0.05, 0.03, 0.03, 0.02, 0.02])
    
    days_birth = np.random.randint(-25000, -7000, num_applicants)
    days_employed = np.where(income_type == 'Pensioner', 365243, np.random.randint(-10000, -100, num_applicants))
    
    occupations = ['Laborers', 'Core staff', 'Sales staff', 'Managers', 'Drivers', 'High skill tech staff', 'Accountants', 'Medicine staff', 'Security staff', 'Cleaning staff', 'Private service staff', 'Low-skill Laborers', 'Secretaries', 'Waiters/barmen staff', 'HR staff', 'Realty agents', 'IT staff']
    occupation_probs = [0.25, 0.15, 0.14, 0.12, 0.09, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.01, 0.01, 0.005, 0.005, 0.005, 0.005]
    occupation_type = np.random.choice(occupations, num_applicants, p=occupation_probs)
    
    flag_mobil = np.ones(num_applicants, dtype=int)
    flag_work_phone = np.random.choice([0, 1], num_applicants, p=[0.8, 0.2])
    flag_phone = np.random.choice([0, 1], num_applicants, p=[0.7, 0.3])
    flag_email = np.random.choice([0, 1], num_applicants, p=[0.9, 0.1])
    cnt_fam_members = children + np.where(np.isin(family_type, ['Married', 'Civil marriage']), 2, 1)

    application_df = pd.DataFrame({
        'ID': applicant_ids,
        'CODE_GENDER': genders,
        'FLAG_OWN_CAR': own_car,
        'FLAG_OWN_REALTY': own_realty,
        'CNT_CHILDREN': children,
        'AMT_INCOME_TOTAL': income_total,
        'NAME_INCOME_TYPE': income_type,
        'NAME_EDUCATION_TYPE': education_type,
        'NAME_FAMILY_STATUS': family_type,
        'NAME_HOUSING_TYPE': housing_type,
        'DAYS_BIRTH': days_birth,
        'DAYS_EMPLOYED': days_employed,
        'FLAG_MOBIL': flag_mobil,
        'FLAG_WORK_PHONE': flag_work_phone,
        'FLAG_PHONE': flag_phone,
        'FLAG_EMAIL': flag_email,
        'OCCUPATION_TYPE': occupation_type,
        'CNT_FAM_MEMBERS': cnt_fam_members
    })
    
    app_path = os.path.join(output_dir, 'application_record.csv')
    application_df.to_csv(app_path, index=False)
    
    # 2. Generate Credit Records
    credit_records = []
    applicants_with_history = np.random.choice(applicant_ids, int(num_applicants * 0.8), replace=False)
    
    statuses = ['C', 'X', '0', '1', '2', '3', '4', '5']
    status_probs = [0.4, 0.2, 0.3, 0.05, 0.02, 0.01, 0.01, 0.01]
    
    for app_id in applicants_with_history:
        history_length = np.random.randint(5, 41)
        months = np.arange(0, -history_length, -1)
        user_statuses = np.random.choice(statuses, size=history_length, p=status_probs)
        
        for month, status in zip(months, user_statuses):
            credit_records.append({'ID': app_id, 'MONTHS_BALANCE': month, 'STATUS': status})
            
    credit_df = pd.DataFrame(credit_records)
    
    credit_path = os.path.join(output_dir, 'credit_record.csv')
    credit_df.to_csv(credit_path, index=False)

if __name__ == '__main__':
    generate_datasets()
