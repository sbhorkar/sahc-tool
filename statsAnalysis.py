#import streamlit as st
import pandas as pd
import numpy as np
import os
import random

DIR = os.getcwd()

DATA_DIR = DIR + '/data/'
NHANES_FILE = os.path.join(DATA_DIR, 'nhanes_merged_data.csv')
df_nhanes = pd.read_csv(NHANES_FILE)


genderOptions = {'Male': 1, 'Female': 2}
ageOptions = {'19-33': 19, '34-48': 34, '49-64': 49, '65-78': 65,'79+': 79}
dataSelection = ['NHANES', 'SAHC']

gender = None
age_group = None
ethnicity = None
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
    #global gender, age_group,ethnicity, medChol,medDiab,medBP
    
    
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

def dump_percentile(metric, column, columnName, df):
    array = df[column].dropna()
    low_number = AHA_RANGES[columnName][1]
    high_number = AHA_RANGES[columnName][3]
    extra_high_number = AHA_RANGES[columnName][5]
    extra_high_percentile = None

    sorted_array = np.sort(array)

    # Calculate the percentile
    if high_number == None:
        high_number = 1000
        high_percentile = 99
        limit = high_number
    else:
        high_percentile = int(np.mean(sorted_array <= high_number) * 100)
        limit = high_number
        if extra_high_number is not None:
            extra_high_percentile = int(np.mean(sorted_array <= extra_high_number) * 100)
            limit = extra_high_number

    low_percentile = int(np.mean(sorted_array <= low_number) * 100)

    # Find the user percentile
    
    if STEP_SIZE[metric] == 1:
        user_input = random.randint(0,limit)
    else:
        user_input = round(random.random()*10, 2)
    user_percentile = int(np.mean(sorted_array <= user_input) * 100)
    if user_percentile == 100:
        user_percentile = 99
    return {'Metric':metric, 'Low Number': low_number, 'Low Percentile': low_percentile, 
            'High Number': high_number,'High Percentile':high_percentile, 
            'Extra High Number': extra_high_number,'Extra High Percentile':extra_high_percentile, 
            'User Input':user_input, 
            'User Percentile': user_percentile }
    #print(low_number, low_percentile, high_number, high_percentile, user_input, user_percentile)
    #if extra_high_number is not None:
    #    print(extra_high_number, extra_high_percentile)
        

print(len(df_nhanes))
df_d = ui_choose(df_nhanes)
print(len(df_d))

#print(DROPDOWN_SELECTION.values())
df_percentile = pd.DataFrame(columns=['Metric', 'Low Number', 'Low Percentile', 'High Number', 
                                      'High Percentile', 'Extra High Number', 'Extra High Percentile',
                                      'User Input', 'User Percentile'])
records =
for metric in DROPDOWN_SELECTION.values():
    print(metric)
    column = DROPDOWN_SELECTION[NAME_MAP[metric]]
    if metric != 'BMXBMI':
        columnName = NAME_MAP[metric] + f" ({UNITS_MAP[metric]})"
    else:
        columnName = NAME_MAP[metric]
    print (column, columnName)
    for i in range(10):
        df_percentile.append(dump_percentile(metric, column, columnName, df_d))
    len(df_percentile)