import streamlit as st
import pandas as pd
import os
import math

DIR=os.getcwd()
DATA_DIR=DIR+'/data/'
OUTPUT_DIR=DIR+'/output/'

USER_FILE=os.path.join(DATA_DIR, 'DEMO_P.XPT')
DIQ_FILE=os.path.join(DATA_DIR, 'P_DIQ.XPT')
BPQ_FILE=os.path.join(DATA_DIR, 'P_BPQ.XPT')
HDL_FILE=os.path.join(DATA_DIR, 'P_HDL.XPT')
TGL_FILE=os.path.join(DATA_DIR, 'P_TRIGLY.XPT')
TCH_FILE=os.path.join(DATA_DIR, 'P_TCHOL.XPT')
GLU_FILE=os.path.join(DATA_DIR, 'P_GLU.XPT')
GHB_FILE=os.path.join(DATA_DIR, 'P_GHB.XPT')
BPX_FILE=os.path.join(DATA_DIR, 'P_BPXO.XPT')
MCQ_FILE=os.path.join(DATA_DIR, 'P_MCQ.XPT')

NAME_MAP = {'LBXTR': 'Triglycerides', 'LBDHDD': 'HDL', 'LBDLDL': 'LDL', 'LBXTC': 'Total Cholesterol', 'LBXGLU': 'Fasting Glucose', 'LBXGH': 'Glycohemoglobin',
            'BPXOSY1': 'Systolic', 'BPXODI1': 'Diastolic', 'BPXOPLS1': 'Pulse'}

GENDER_MAP=['Unknown','Male','Female']
COMMON_YES_NO_MAP=['Unknown','Yes','No','Yes']


AGE_MAP = [
    {'min':0,  'max':17, 'name':'0-17','id':0},
    {'min':18, 'max':35, 'name':'18-35','id':1},
    {'min':36, 'max':55, 'name':'36-45','id':2},
    {'min':46, 'max':55, 'name':'46-55','id':3},
    {'min':56, 'max':65, 'name':'56-65','id':4},
    {'min':66, 'max':79, 'name':'66-79','id':5},
    {'min':80, 'max':500, 'name':'80+','id':6}
]

def group_for_age(age):
    for a in AGE_MAP:
        if a['min'] <= age <= a['max']:
            return a['name']
    return 'Unknown Age Group for age={}'.format(age)

def group_base(x):
    if math.isnan(x):
        return 'Others'
    y=round(x)
    if y in [1]:
        return 'Yes'
    elif y in [2]:
        return 'No'
    else:
        return 'Others'

def group_common(x):
    if math.isnan(x):
        return 'Others'
    y=round(x)
    if y in [1,3]:
        return 'Yes'
    elif y in [2]:
        return 'No'
    else:
        return 'Others'
    

def group_gender(x):
    if math.isnan(x):
        return 'Others'
    y=round(x)
    if y in [1]:
        return 'Male'
    elif y in [2]:
        return 'Female'
    else:
        return 'Others'

@st.cache_data
def load_and_combine_files(debugging):
    df_user = pd.read_sas(USER_FILE, format='xport')
    df_diq = pd.read_sas(DIQ_FILE, format='xport')
    df_bpq = pd.read_sas(BPQ_FILE, format='xport')
    df_hdl = pd.read_sas(HDL_FILE, format='xport')
    df_tgl = pd.read_sas(TGL_FILE, format='xport')
    df_tch = pd.read_sas(TCH_FILE, format='xport')
    df_glu = pd.read_sas(GLU_FILE, format='xport')
    df_ghb = pd.read_sas(GHB_FILE, format='xport')
    df_bpx = pd.read_sas(BPX_FILE, format='xport')
    df_mcq = pd.read_sas(MCQ_FILE, format='xport')
    #
    df_combined = df_user[['SEQN','RIAGENDR','RIDAGEYR']]
    df_combined = pd.merge(df_combined, df_diq[['SEQN', 'DIQ010', 'DIQ160','DIQ050','DIQ070']], on='SEQN', how='left')
    df_combined = pd.merge(df_combined, df_bpq[['SEQN', 'BPQ090D', 'BPQ100D','BPQ040A','BPQ050A']], on='SEQN', how='left')
    df_combined = pd.merge(df_combined, df_hdl[['SEQN', 'LBDHDD']], on='SEQN', how='left')
    df_combined = pd.merge(df_combined, df_tgl[['SEQN', 'LBXTR','LBDLDL']], on='SEQN', how='left')
    df_combined = pd.merge(df_combined, df_tch[['SEQN', 'LBXTC']], on='SEQN', how='left')
    df_combined = pd.merge(df_combined, df_glu[['SEQN', 'LBXGLU']], on='SEQN', how='left')
    df_combined = pd.merge(df_combined, df_ghb[['SEQN', 'LBXGH']], on='SEQN', how='left')
    df_combined = pd.merge(df_combined, df_bpx[['SEQN', 'BPXOSY1', 'BPXODI1', 'BPXOPLS1','BPXOSY2', 'BPXODI2', 'BPXOPLS2','BPXOSY3', 'BPXODI3', 'BPXOPLS3' ]], on='SEQN', how='left')
    df_combined = pd.merge(df_combined, df_mcq[['SEQN', 'MCQ160C', 'MCQ160D', 'MCQ160E']], on='SEQN', how='left')
    #
    #
    if debugging:
        st.dataframe(df_combined,hide_index=True)
        st.dataframe(df_user,hide_index=True)
        st.dataframe(df_diq,hide_index=True)
        st.dataframe(df_bpq,hide_index=True)
        st.dataframe(df_hdl,hide_index=True)
        st.dataframe(df_tgl,hide_index=True)
        st.dataframe(df_tch,hide_index=True)
        st.dataframe(df_glu,hide_index=True)
        st.dataframe(df_ghb,hide_index=True)
        st.dataframe(df_bpx,hide_index=True)
        st.dataframe(df_mcq,hide_index=True)
    df_combined.to_csv(os.path.join(OUTPUT_DIR, 'nhanes_combined.csv'),index=False)
    return df_combined

a="""
    genderList = st.sidebar.multiselect('Gender',list(genderOptions.keys()))
    ageList = st.sidebar.multiselect('Age groups',list(ageOptions.keys()))
    diabetesList = st.sidebar.multiselect('Diabetes status',options=list(diabetesOptions.keys()))

    medCholList = st.sidebar.multiselect('Cholesterol medication',options=list(medCholOptions.keys()))
    medDiabList = st.sidebar.multiselect('Diabetes medication',options=list(medDiabOptions.keys()))
    medBPList = st.sidebar.multiselect('Blood pressure medication',options=list(medBPOptions.keys()))


"""

def augment_columns(df):
    exclude_columns = ['LBXGH']
    columns_to_convert = df.columns.difference(exclude_columns)
    df[columns_to_convert] = df[columns_to_convert].applymap(lambda x: pd.to_numeric(x, errors='coerce'))
    df['GP_Age'] = df['RIDAGEYR'].apply(lambda age: group_for_age(age))
    df['GP_Gender'] = df['RIAGENDR'].apply(lambda gender: group_common(gender))
    df['GP_StatusDiab'] = df['DIQ010'].apply(lambda x: group_common(x))
    df['GP_StatusPreDiab'] = df['DIQ160'].apply(lambda x: group_common(x))
    df['GP_StatusCAD1'] = df['MCQ160C'].apply(lambda x: group_base(x))
    df['GP_StatusCAD2'] = df['MCQ160D'].apply(lambda x: group_base(x))
    df['GP_StatusCAD3'] = df['MCQ160E'].apply(lambda x: group_base(x))
    df['GP_MedChol1'] = df['BPQ090D'].apply(lambda x: group_base(x))
    df['GP_MedChol2'] = df['BPQ100D'].apply(lambda x: group_base(x))
    df['GP_MedDiab1'] = df['DIQ050'].apply(lambda x: group_base(x))
    df['GP_MedDiab2'] = df['DIQ070'].apply(lambda x: group_base(x))
    df['GP_MedBP1'] = df['BPQ040A'].apply(lambda x: group_base(x))
    df['GP_MedBP2'] = df['BPQ050A'].apply(lambda x: group_base(x))

    #
    #
    #
    #
    df.to_csv(os.path.join(OUTPUT_DIR, 'nhanes_augmented.csv'),index=False)
    return df




#
#
#
df_c=load_and_combine_files(True)
df_aug=augment_columns(df_c)
st.write("# Completed loading, combining and augmenting files. \n ## Check the output directory for nhanes_augmented.csv. ")
st.dataframe(df_aug,hide_index=True)