import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

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

NAME_MAP = {'LBXTR': 'Triglycerides', 'LBDHDD': 'HDL', 'LBDLDL': 'LDL', 'LBXTC': 'Total Cholesterol', 'LBXGLU': 'Fasting Glucose', 'LBXGH': 'Glycohemoglobin',
            'BPXOSY1': 'Systolic', 'BPXODI1': 'Diastolic', 'BPXOPLS1': 'Pulse'}

@st.cache_data
def load_files(debugging):
    df_user = pd.read_sas(USER_FILE, format='xport')
    df_diq = pd.read_sas(DIQ_FILE, format='xport')
    df_bpq = pd.read_sas(BPQ_FILE, format='xport')
    df_hdl = pd.read_sas(HDL_FILE, format='xport')
    df_tgl = pd.read_sas(TGL_FILE, format='xport')
    df_tch = pd.read_sas(TCH_FILE, format='xport')
    df_glu = pd.read_sas(GLU_FILE, format='xport')
    df_ghb = pd.read_sas(GHB_FILE, format='xport')
    df_bpx = pd.read_sas(BPX_FILE, format='xport')
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
    df_combined['Age_Group'] = df_combined['RIDAGEYR'].apply(lambda age: 20 * int(age/20))
    exclude_columns = ['LBXGH']
    columns_to_convert = df_combined.columns.difference(exclude_columns)
    df_combined[columns_to_convert] = df_combined[columns_to_convert].applymap(lambda x: pd.to_numeric(x, errors='coerce'))
    #
    if debugging:
        st.dataframe(df_combined,hide_index=True)
    if debugging:
        st.dataframe(df_user,hide_index=True)
        st.dataframe(df_diq,hide_index=True)
        st.dataframe(df_bpq,hide_index=True)
        st.dataframe(df_hdl,hide_index=True)
        st.dataframe(df_tgl,hide_index=True)
        st.dataframe(df_tch,hide_index=True)
        st.dataframe(df_glu,hide_index=True)
        st.dataframe(df_ghb,hide_index=True)
        st.dataframe(df_bpx,hide_index=True)
    if debugging:
        df_combined.to_csv(os.path.join(OUTPUT_DIR, 'nhanes_combined.csv'),index=False)
    return df_combined

def filter_df(df,genderList,ageList,diabetesSelect,medCholSelect,medDiabSelect,medBPSelect):
    df2=df.copy()
    st.sidebar.write(f"Intial, records= {df2.shape}")
    df2=df2[df2['RIAGENDR'].isin(genderList)]
    st.sidebar.write(f"After Gender, records= {df2.shape}")
    df2=df2[df2['Age_Group'].isin(ageList)]
    st.sidebar.write(f"After Age, records= {df2.shape}")
    df2=df2[df2['DIQ010']==diabetesSelect]
    st.sidebar.write(f"After Diabetes diagnosis, records= {df2.shape}")
    df2=df2[df2['DIQ160']==medDiabSelect]
    st.sidebar.write(f"After diabetes meds, records= {df2.shape}")
    df2=df2[df2['BPQ090D']==medCholSelect]
    st.sidebar.write(f"After Chol meds, records= {df2.shape}")
    df2=df2[df2['BPQ100D']==medBPSelect]
    st.sidebar.write(f"After BP meds, records= {df2.shape}")
    return df2

def ui_choose(df, debugging):
    genderOptions={'Male':1,'Female':2}
    ageOptions={'<20':0,'20-40':20,'40-60':40,'60-80':60,'80+':80}
    diabetesOptions={'Yes':1,'No':2,'Borderline':3}
    medCholOptions={'Yes':1,'No':2}
    medDiabOptions={'Yes':1,'No':2}
    medBPOptions={'Yes':1,'No':2}

    genderList = st.sidebar.multiselect('Gender (can select multiple)',list(genderOptions.keys()))
    ageList = st.sidebar.multiselect('Age groups (can select multiple)',list(ageOptions.keys()))
    diabetesSelect = st.sidebar.selectbox('Diabetes status (pick one)',options=list(diabetesOptions.keys()))

    medCholSelect = st.sidebar.selectbox('Cholesterol medication (pick one)',options=list(medCholOptions.keys()))
    medDiabSelect = st.sidebar.selectbox('Diabetes medication (pick one)',options=list(medDiabOptions.keys()))
    medBPSelect = st.sidebar.selectbox('Blood pressure medication (pick one)',options=list(medBPOptions.keys()))

    if debugging:
        st.write(genderList)
        st.write(ageList)
        st.write(diabetesSelect)
        st.write(medCholSelect)
        st.write(medDiabSelect)
        st.write(medBPSelect)

    genderFilter=[genderOptions[i] for i in genderList]
    ageFilter=[ageOptions[i] for i in ageList]
    diabetesFilter=diabetesOptions[diabetesSelect]
    medCholFilter=medCholOptions[medCholSelect]
    medDiabFilter=medDiabOptions[medDiabSelect]
    medBPFilter=medBPOptions[medBPSelect]

    df2 = filter_df(df,genderFilter,ageFilter,diabetesFilter,medCholFilter,medDiabFilter,medBPFilter)
    return df2

def show_analysis(df):
    st.write("# Raw Data")
    st.dataframe(df, hide_index=True)

    for column in ['LBDHDD', 'LBXTR','LBDLDL', 'LBXTC', 'LBXGLU', 'BPXOSY1', 'BPXODI1', 'BPXOPLS1']:
        columnName=NAME_MAP[column]
        plt.figure(figsize=(10, 6))
        df[column].plot(kind='box')
        plt.title(f'Boxplot for {columnName}')
        plt.xlabel('Values')
        plt.ylabel(columnName)
        st.pyplot(plt.gcf())  # Show the plot in Streamlit

    plt.figure(figsize=(10, 6))
    # Add code for scatter plot
    st.pyplot(plt.figure())  # Show the plot in a separate window

    plt.figure(figsize=(10, 6))
    # Add code for histogram
    st.pyplot(plt.figure())  # Show the plot in a separate window

df_c=load_files(False)
df_d=ui_choose(df_c,False)
show_analysis(df_d)