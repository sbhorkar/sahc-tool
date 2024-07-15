import streamlit as st
import pandas as pd
import os
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

DIR = os.getcwd()
DATA_DIR = DIR + '/data/'
OUTPUT_DIR = DIR + '/output/'

USER_FILE = os.path.join(DATA_DIR, 'DEMO_P.XPT')
DIQ_FILE = os.path.join(DATA_DIR, 'P_DIQ.XPT')
BPQ_FILE = os.path.join(DATA_DIR, 'P_BPQ.XPT')
HDL_FILE = os.path.join(DATA_DIR, 'P_HDL.XPT')
TGL_FILE = os.path.join(DATA_DIR, 'P_TRIGLY.XPT')
TCH_FILE = os.path.join(DATA_DIR, 'P_TCHOL.XPT')
GLU_FILE = os.path.join(DATA_DIR, 'P_GLU.XPT')
GHB_FILE = os.path.join(DATA_DIR, 'P_GHB.XPT')
BPX_FILE = os.path.join(DATA_DIR, 'P_BPXO.XPT')

NAME_MAP = {
    'LBXTR': 'Triglycerides (mg/dL)', 'LBDHDD': 'HDL (mg/dL)', 'LBDLDL': 'LDL (mg/dL)',
    'LBXTC': 'Total Cholesterol (mg/dL)', 'LBXGLU': 'Fasting Glucose (mg/dL)',
    'BPXOSY1': 'Systolic (mmHg)', 'BPXODI1': 'Diastolic (mmHg)', 'BPXOPLS1': 'Pulse'
}

# Uniform AHA ranges regardless of gender and age
AHA_RANGES = {
    'Triglycerides (mg/dL)': (None, 150),
    'HDL (mg/dL)': (40, None),
    'LDL (mg/dL)': (None, 100),
    'Total Cholesterol (mg/dL)': (140, 160),
    'Fasting Glucose (mg/dL)': (None, 100),
    'Systolic (mmHg)': (None, 120),
    'Diastolic (mmHg)': (None, 80),
    'Pulse': (60, 100)
}

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

    df_combined = df_user[['SEQN', 'RIAGENDR', 'RIDAGEYR', 'RIDRETH3']]
    df_combined = pd.merge(df_combined, df_diq[['SEQN', 'DIQ010', 'DIQ160', 'DIQ050', 'DIQ070']], on='SEQN', how='left')
    df_combined = pd.merge(df_combined, df_bpq[['SEQN', 'BPQ090D', 'BPQ100D', 'BPQ040A', 'BPQ050A']], on='SEQN', how='left')
    df_combined = pd.merge(df_combined, df_hdl[['SEQN', 'LBDHDD']], on='SEQN', how='left')
    df_combined = pd.merge(df_combined, df_tgl[['SEQN', 'LBXTR', 'LBDLDL']], on='SEQN', how='left')
    df_combined = pd.merge(df_combined, df_tch[['SEQN', 'LBXTC']], on='SEQN', how='left')
    df_combined = pd.merge(df_combined, df_glu[['SEQN', 'LBXGLU']], on='SEQN', how='left')
    df_combined = pd.merge(df_combined, df_bpx[['SEQN', 'BPXOSY1', 'BPXODI1', 'BPXOPLS1', 'BPXOSY2', 'BPXODI2', 'BPXOPLS2', 'BPXOSY3', 'BPXODI3', 'BPXOPLS3']], on='SEQN', how='left')
    df_combined['Age_Group'] = df_combined['RIDAGEYR'].apply(lambda age: 20 * int(age / 20))

    if debugging:
        st.dataframe(df_combined, hide_index=True)
        st.dataframe(df_user, hide_index=True)
        st.dataframe(df_diq, hide_index=True)
        st.dataframe(df_bpq, hide_index=True)
        st.dataframe(df_hdl, hide_index=True)
        st.dataframe(df_tgl, hide_index=True)
        st.dataframe(df_tch, hide_index=True)
        st.dataframe(df_glu, hide_index=True)
        st.dataframe(df_ghb, hide_index=True)
        st.dataframe(df_bpx, hide_index=True)

    if debugging:
        df_combined.to_csv(os.path.join(OUTPUT_DIR, 'nhanes_combined.csv'), index=False)
    return df_combined

def ui_choose(df, debugging):
    genderOptions = {'Male': 1, 'Female': 2}
    ageOptions = {'20-40': 20, '40-60': 40, '60-80': 60, '80+': 80}
    medCholOptions = {'Yes': 1, 'No': 2}
    medDiabOptions = {'Yes': 1, 'No': 2}
    medBPOptions = {'Yes': 1, 'No': 2}

    gender = st.sidebar.selectbox('Gender', list(genderOptions.keys()), placeholder="Choose an option")
    age_group = st.sidebar.selectbox('Age group', list(ageOptions.keys()), placeholder="Choose an option")
    medChol = st.sidebar.selectbox('Cholesterol medication', options=list(medCholOptions.keys()), placeholder="Choose an option")
    medDiab = st.sidebar.selectbox('Diabetes medication', options=list(medDiabOptions.keys()), placeholder="Choose an option")
    medBP = st.sidebar.selectbox('Blood pressure medication', options=list(medBPOptions.keys()), placeholder="Choose an option")

    if debugging:
        st.write(gender)
        st.write(age_group)
        st.write(medChol)
        st.write(medDiab)
        st.write(medBP)

    genderFilter = [genderOptions[gender]]
    ageFilter = [ageOptions[age_group]]
    medCholFilter = [medCholOptions[medChol]]
    medDiabFilter = [medDiabOptions[medDiab]]
    medBPFilter = [medBPOptions[medBP]]

    df2 = df.copy()
    df2 = df2[df2['RIAGENDR'].isin(genderFilter)]
    df2 = df2[df2['Age_Group'].isin(ageFilter)]
    df2 = df2[df2['BPQ090D'].isin(medCholFilter)]
    df2 = df2[df2['DIQ160'].isin(medDiabFilter)]
    df2 = df2[df2['BPQ100D'].isin(medBPFilter)]
    st.sidebar.write(f"After BP meds, records= {df2.shape}")
    return df2, gender, age_group

def create_legend():
    fig, ax = plt.subplots(figsize=(10, 1))  # Create a separate figure for the legend
    ax.axis('off')  # Hide axes

    # Create custom legend lines
    custom_lines = [
        Line2D([0], [0], color='grey', lw=10, alpha=0.5, label='Percentile Range'),
        Line2D([0], [0], color='green', lw=10, label='Healthy Range'),
        Line2D([0], [0], color='blue', marker='o', linestyle='None', label='Your Input')
    ]

    # Add the custom legend
    ax.legend(handles=custom_lines, loc='center', ncol=3, frameon=False)
    st.pyplot(fig)
    plt.close()

def show_analysis(df, gender, age_group):
    input_labels = {
        'LBDHDD': "Enter your value for HDL (mg/dL)",
        'LBDLDL': "Enter your value for LDL (mg/dL)",
        'LBXTC': "Enter your value for Total Cholesterol (mg/dL)",
        'LBXTR': "Enter your value for Triglycerides (mg/dL)",
        'LBXGLU': "Enter your value for Fasting Glucose (mg/dL)",
        'BPXOSY1': "Enter your value for Systolic Blood Pressure, the top number in a BP reading (mmHg)",
        'BPXODI1': "Enter your value for Diastolic Blood Pressure, the bottom number in a BP reading (mmHg)",
        'BPXOPLS1': "Enter your value for Pulse"
    }

    user_inputs = {}

    for key, label in input_labels.items():
        user_inputs[key] = st.sidebar.number_input(label, key=key, step=1)

    genderOptions = {'Male': 1, 'Female': 2}
    ageOptions = {'<20': 0, '20-40': 20, '40-60': 40, '60-80': 60, '80+': 80}

    for column in ['LBDHDD', 'LBDLDL', 'LBXTC', 'LBXTR', 'LBXGLU', 'BPXOSY1', 'BPXODI1', 'BPXOPLS1']:
        columnName = NAME_MAP[column]
        user_input = user_inputs[column]  # Reference the user input from the dictionary

        st.write(f"### {columnName}")

        if columnName in AHA_RANGES:
            low_value = AHA_RANGES[columnName][0]
            high_value = AHA_RANGES[columnName][1]  # The second value in the tuple is the normal range

            if low_value is not None:
                low_number = normal_value
            if high_value is not None:
                high_number = normal_value
            elif low_value is None:
                low_number = 0
        else:
            st.write(f"No AHA range available for {columnName}.")
            continue

        array = df[column].dropna()
        sorted_array = np.sort(array)

        # Calculate the percentile
        percentile = np.mean(sorted_array <= low_number) * 100

        if percentile == 0:
            st.write(f"No data available for {columnName}.")
            continue

        fig, ax = plt.subplots(figsize=(10, .5))

        # Grey line from 0 to 100
        ax.plot([0, 100], [.9, .9], color='grey', lw=10, alpha=0.5, label='Percentile Range')

        # Green line from percentile to 100
        ax.plot([percentile, 100], [0.65, 0.65], color='green', lw=10, label='Healthy Range')

        # Blue dot at user input position
        user_percentile = np.mean(sorted_array <= user_input) * 100
        ax.scatter(user_percentile, 0.9, color='blue', zorder=5, label='Your Input')

        ax.set_xlim(0, 100)
        ax.set_ylim(0.4, 1.1)
        ax.set_yticks([])
        ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        ax.set_xlabel('Percentile (%)')

        st.pyplot(fig)
        plt.close()

# Main execution
create_legend()
df_c = load_files(False)
df_d, gender, age = ui_choose(df_c, False)
show_analysis(df_d, gender, age)
