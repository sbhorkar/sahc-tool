#import streamlit as st
import pandas as pd
import numpy as np
import os
import random

DIR = os.getcwd()

DATA_DIR = DIR + '/data/'
NHANES_FILE = os.path.join(DATA_DIR, 'nhanes_merged_data.csv')
SAHC_DATA_DIR = DIR + '/sahc_data/'

SAHC_RENAME_FILE = os.path.join(SAHC_DATA_DIR, 'renamed_merged_data_noPID.csv')
df_nhanes = pd.read_csv(NHANES_FILE)
df_sahc = pd.read_csv(SAHC_RENAME_FILE)


genderOptions = {'Male': 1, 'Female': 2}
ageOptions = {'19-33': 19, '34-48': 34, '49-64': 49, '65-78': 65,'79+': 79}
dataSelection = ['NHANES', 'SAHC']

gender = None
age_group = None
#ethnicity = None
ethnicity = 'South Asians only'
medChol = 'No'
medDiab = 'No'
medBP = 'No'

medCholOptions = {'Yes': 1, 'No': 2}
medDiabOptions = {'Yes': 1, 'No': 2}
medBPOptions = {'Yes': 1, 'No': 2}

DROPDOWN_SELECTION = {
    'Total Cholesterol': 'LBXTC', 'LDL': 'LBDLDL', 'HDL': 'LBDHDD',
    'Triglycerides': 'LBXTR', 'Total Cholesterol:HDL': 'TotHDLRat',
    'Fasting Glucose': 'LBXGLU', 'HbA1c': 'LBXGH',
    'Body Mass Index': 'BMXBMI', 'Systolic BP': 'BPXOSY1', 'Diastolic BP': 'BPXODI1', 
}

NAME_MAP = {
    'LBXTC': 'Total Cholesterol', 'LBDLDL': 'LDL', 'LBDHDD': 'HDL', 
    'LBXTR': 'Triglycerides', 'TotHDLRat': 'Total Cholesterol:HDL', 
    'LBXGLU': 'Fasting Glucose', 'LBXGH': 'HbA1c',
    'BMXBMI': 'Body Mass Index', 'BPXOSY1': 'Systolic BP', 'BPXODI1': 'Diastolic BP', 
}
AHA_RANGES = {
    'Triglycerides (mg/dL)': ("Optimal", 150, "Borderline", 200, "At risk", None, None),
    'HDL (mg/dL)': ("At risk", 40, "Optimal", None, None, None, None),
    'LDL (mg/dL)': ("Optimal", 100, "Borderline", 160, "At risk", None, None),
    'Total Cholesterol (mg/dL)': ("Optimal", 200, "Borderline", 240, "At risk", None, None),
    'Fasting Glucose (mg/dL)': ("Optimal", 100, "Borderline", 125, "At risk", None, None),
    'Systolic BP (mmHg)': ("Optimal", 120, "At risk", None, None, None, None),
    'Diastolic BP (mmHg)': ("Optimal", 80, "At risk", None, None, None, None),
    'Total Cholesterol:HDL ()':("Optimal", 3.5, "Borderline", 5, "At risk", None, None),
    'HbA1c (%)': ("Optimal", 5.7, "Borderline", 6.4, "At risk", None, None),
    'Body Mass Index': ("Low", 18.5, "Optimal", 25, "Borderline", 30, "At risk")
}

UNITS_MAP = {
    'LBDHDD': "mg/dL", 'LBDLDL': "mg/dL", 'LBXTC': "mg/dL", 'LBXTR': "mg/dL", 'LBXGH': "%",
    'LBXGLU': "mg/dL", 'BPXOSY1': "mmHg", 'BPXODI1': "mmHg", 'LBXHGB': "g/dL", 'TotHDLRat': "", 'BMXBMI': ""
}

STEP_SIZE = {
        'LBDHDD': 1,
        'LBDLDL': 1,
        'LBXTC': 1,
        'LBXTR': 1,
        'LBXGLU': 1,
        'BPXOSY1': 1,
        'BPXODI1': 1,
        'TotHDLRat': .1,
        'LBXGH': .1,
        'BMXBMI': 1,
    }

def ui_choose(df):    
    
    df2 = df.copy()

    if gender is not None:
        genderFilter = [genderOptions[gender]]
        df2 = df2[df2['RIAGENDR'].isin(genderFilter)]
    if age_group is not None:
        ageFilter = [ageOptions[age_group]]
        df2 = df2[df2['Age_Group'].isin(ageFilter)]
    
    if ethnicity is not None and ethnicity == 'South Asians only':
        medCholFilter = [medCholOptions[medChol]]
        df2 = df2[df2['cholMeds'].isin(medCholFilter)]
        medDiabFilter = [medDiabOptions[medDiab]]
        df2 = df2[df2['diabMeds'].isin(medDiabFilter)]
        medBPFilter = [medBPOptions[medBP]]
        df2 = df2[df2['bpMeds'].isin(medBPFilter)]
    elif ethnicity is None or ethnicity != 'South Asians only':
        medCholFilter = [medCholOptions[medChol]]
        if medChol == 'Yes':
            df2 = df2[df2['BPQ100D'].isin(medCholFilter)]
        elif medChol == 'No':
            df2 = df2[df2['BPQ090D'].isin(medCholFilter) | df2['BPQ100D'].isin(medCholFilter)]
        medDiabFilter = [medDiabOptions[medDiab]]
        df2 = df2[df2['DIQ070'].isin(medDiabFilter)]
        medBPFilter = [medBPOptions[medBP]]
        if medBP == 'Yes':
            df2 = df2[df2['BPQ040A'].isin(medBPFilter)]
        elif medBP == 'No':
            df2 = df2[df2['BPQ020'].isin(medBPFilter) | df2['BPQ040A'].isin(medBPFilter)]
    
    
    if len(df2) < 15:
        print(f"Not enough records found to compare. Please remove medication usage and try again.") 
    
    # Edit the BMI and HDL numbers for ethnicity/gender
    if ethnicity == 'South Asians only':
        AHA_RANGES['Body Mass Index'] = ("Low", 18.5, "Optimal", 23, "Borderline", 30, "At risk")
    if gender == 'Female':
        AHA_RANGES['HDL (mg/dL)'] = ("At risk", 50, "Optimal", None, None, None, None)

    return df2

def dump_percentile(metric, column, df, user_input, random_input = True):
    array = df[column].dropna()

    sorted_array = np.sort(array)
    #print(user_input)

    if random_input:
        if STEP_SIZE[metric] == 1:
            user_input = random.randint(int(sorted_array[0]),int(sorted_array[-1]))
        else:
            user_input = round(random.random()*(sorted_array[-1] - sorted_array[0]), 2)

    user_percentile = int(np.mean(sorted_array <= user_input) * 100)
    if user_percentile == 100:
        user_percentile = 99

            
    selected_data = [float(x) for x in sorted_array if x <= user_input]
    #actual_percentile = len(selected_data)/len(sorted_array)
    return {'Metric':metric, 'Data Sample': sorted_array, 'Length': len(sorted_array),
            'User Input':user_input, 
            'User Data': selected_data, 'User Data Length': len (selected_data), 
            'User Percentile': user_percentile
            #, 'Actual Percentile': actual_percentie, 
            #'Difference': ((actual_percentie*100)-user_percentile) 
            }
        
if ethnicity is None:
    df_d = ui_choose(df_nhanes)
else: 
    df_d = ui_choose(df_sahc)

#print(DROPDOWN_SELECTION.values())
#seqns = df_d['SEQN']
#print (seqns)
#selected_seqn = random.sample(seqns.tolist(), 10)
#print (selected_seqn)
#new_df_d = df_d[df_d['SEQN'].isin(selected_seqn)]
#print(new_df_d)
records = []
for metric in DROPDOWN_SELECTION.values():
    
    column = DROPDOWN_SELECTION[NAME_MAP[metric]]
    
    boundary_user_input = [0, 100]

    #column = DROPDOWN_SELECTION[metric]

    if column == 'BMXBMI':
        columnName = NAME_MAP[metric]
    else:
        columnName = NAME_MAP[metric] + f" ({UNITS_MAP[column]})"
    range_values_dict = AHA_RANGES[columnName]
    range_values = [x for x in range_values_dict if isinstance(x, (int, float))]
    for x in range_values:
        boundary_user_input.append(x)

    for user_input in boundary_user_input: 
        records.append(dump_percentile(metric, column, df_d, user_input, False))

    for i in range(100):
        #records.append(dump_percentile(metric, column, columnName, df_d))
        # Find the user percentile
        
        records.append(dump_percentile(metric, column, df_d, user_input))
df_percentile = pd.DataFrame(records)
print(len(df_percentile))
df_percentile.to_csv('SAHC_Percentile_3.csv')