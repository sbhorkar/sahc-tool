import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

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

AHA_RANGES = {
    (1, 0): {
        'Triglycerides (mg/dL)': (70, 150, 199, 499),
        'HDL (mg/dL)': (None, 40, 60, None),
        'LDL (mg/dL)': (None, 100, 129, 159),
        'Total Cholesterol (mg/dL)': (None, 200, 239, None),
        'Fasting Glucose (mg/dL)': (70, 99, 125, None),
        'Systolic (mmHg)': (90, 120, 129, 139),
        'Diastolic (mmHg)': (60, 80, 89, 99),
        'Pulse': (60, 100, None, None),
    },
    (1, 20): {
        'Triglycerides (mg/dL)': (70, 150, 199, 499),
        'HDL (mg/dL)': (None, 40, 60, None),
        'LDL (mg/dL)': (None, 100, 129, 159),
        'Total Cholesterol (mg/dL)': (None, 200, 239, None),
        'Fasting Glucose (mg/dL)': (70, 99, 125, None),
        'Systolic (mmHg)': (90, 120, 129, 139),
        'Diastolic (mmHg)': (60, 80, 89, 99),
        'Pulse': (60, 100, None, None),
    },
    (1, 40): {
        'Triglycerides (mg/dL)': (70, 150, 199, 499),
        'HDL (mg/dL)': (None, 40, 60, None),
        'LDL (mg/dL)': (None, 100, 129, 159),
        'Total Cholesterol (mg/dL)': (None, 200, 239, None),
        'Fasting Glucose (mg/dL)': (70, 99, 125, None),
        'Systolic (mmHg)': (90, 120, 129, 139),
        'Diastolic (mmHg)': (60, 80, 89, 99),
        'Pulse': (60, 100, None, None),
    },
    (1, 60): {
        'Triglycerides (mg/dL)': (70, 150, 199, 499),
        'HDL (mg/dL)': (None, 40, 60, None),
        'LDL (mg/dL)': (None, 100, 129, 159),
        'Total Cholesterol (mg/dL)': (None, 200, 239, None),
        'Fasting Glucose (mg/dL)': (70, 99, 125, None),
        'Systolic (mmHg)': (90, 120, 129, 139),
        'Diastolic (mmHg)': (60, 80, 89, 99),
        'Pulse': (60, 100, None, None),
    },
    (1, 80): {
        'Triglycerides (mg/dL)': (70, 150, 199, 499),
        'HDL (mg/dL)': (None, 40, 60, None),
        'LDL (mg/dL)': (None, 100, 129, 159),
        'Total Cholesterol (mg/dL)': (None, 200, 239, None),
        'Fasting Glucose (mg/dL)': (70, 99, 125, None),
        'Systolic (mmHg)': (90, 120, 129, 139),
        'Diastolic (mmHg)': (60, 80, 89, 99),
        'Pulse': (60, 100, None, None),
    },
    (2, 0): {
        'Triglycerides (mg/dL)': (70, 150, 199, 499),
        'HDL (mg/dL)': (None, 50, 70, None),
        'LDL (mg/dL)': (None, 100, 129, 159),
        'Total Cholesterol (mg/dL)': (None, 200, 239, None),
        'Fasting Glucose (mg/dL)': (70, 99, 125, None),
        'Systolic (mmHg)': (90, 120, 129, 139),
        'Diastolic (mmHg)': (60, 80, 89, 99),
        'Pulse': (60, 100, None, None),
    },
    (2, 20): {
        'Triglycerides (mg/dL)': (70, 150, 199, 499),
        'HDL (mg/dL)': (None, 50, 70, None),
        'LDL (mg/dL)': (None, 100, 129, 159),
        'Total Cholesterol (mg/dL)': (None, 200, 239, None),
        'Fasting Glucose (mg/dL)': (70, 99, 125, None),
        'Systolic (mmHg)': (90, 120, 129, 139),
        'Diastolic (mmHg)': (60, 80, 89, 99),
        'Pulse': (60, 100, None, None),
    },
    (2, 40): {
        'Triglycerides (mg/dL)': (70, 150, 199, 499),
        'HDL (mg/dL)': (None, 50, 70, None),
        'LDL (mg/dL)': (None, 100, 129, 159),
        'Total Cholesterol (mg/dL)': (None, 200, 239, None),
        'Fasting Glucose (mg/dL)': (70, 99, 125, None),
        'Systolic (mmHg)': (90, 120, 129, 139),
        'Diastolic (mmHg)': (60, 80, 89, 99),
        'Pulse': (60, 100, None, None),
    },
    (2, 60): {
        'Triglycerides (mg/dL)': (70, 150, 199, 499),
        'HDL (mg/dL)': (None, 50, 70, None),
        'LDL (mg/dL)': (None, 100, 129, 159),
        'Total Cholesterol (mg/dL)': (None, 200, 239, None),
        'Fasting Glucose (mg/dL)': (70, 99, 125, None),
        'Systolic (mmHg)': (90, 120, 129, 139),
        'Diastolic (mmHg)': (60, 80, 89, 99),
        'Pulse': (60, 100, None, None),
    },
    (2, 80): {
        'Triglycerides (mg/dL)': (70, 150, 199, 499),
        'HDL (mg/dL)': (None, 50, 70, None),
        'LDL (mg/dL)': (None, 100, 129, 159),
        'Total Cholesterol (mg/dL)': (None, 200, 239, None),
        'Fasting Glucose (mg/dL)': (70, 99, 125, None),
        'Systolic (mmHg)': (90, 120, 129, 139),
        'Diastolic (mmHg)': (60, 80, 89, 99),
        'Pulse': (60, 100, None, None),
    }
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
    df_combined = df_combined[df_combined['RIDRETH3'] == 6]
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
    ageOptions = {'<20': 0, '20-40': 20, '40-60': 40, '60-80': 60, '80+': 80}
    medCholOptions = {'Yes': 1, 'No': 2}
    medDiabOptions = {'Yes': 1, 'No': 2}
    medBPOptions = {'Yes': 1, 'No': 2}

    gender = st.sidebar.selectbox('Gender', list(genderOptions.keys()), placeholder="Choose an option")
    age_group = st.sidebar.selectbox('Age groups', list(ageOptions.keys()), placeholder="Choose an option")
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
    return df2, gender, age_group

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

        # NHANES Boxplot
        st.write(f"## {columnName}")
        plt.figure(figsize=(10, 3))  # Adjusted height for the box plot
        plt.boxplot(df[column].dropna(), vert=False)

        stats = df[column].describe()
        Q1 = stats["25%"]
        Q3 = stats["75%"]
        IQR = Q3 - Q1
        median = stats["50%"]

        if user_input < Q1:
            plt.scatter(user_input, 1, color='red', marker='v', label='Your Value')
        elif Q1 <= user_input <= Q3:
            plt.scatter(user_input, 1, color='red', marker='o', label='Your Value')
        else:
            plt.scatter(user_input, 1, color='red', marker='^', label='Your Value')

        plt.axvline(x=user_input, color='red', linestyle='--')
        plt.legend()

        plt.title(f'NHANES Data Boxplot for {columnName}')
        plt.xlabel(columnName)

        lower_whisker = Q1 - 1.5 * IQR
        upper_whisker = Q3 + 1.5 * IQR
        lower_whisker_val = df[column][df[column] >= lower_whisker].min()
        upper_whisker_val = df[column][df[column] <= upper_whisker].max()

        plt.annotate(f'Median: {median:.2f}', xy=(median, 1), xytext=(median, 1.1),
                     arrowprops=dict(facecolor='green', shrink=0.05, width=0.5), horizontalalignment='center')
        plt.annotate(f'25th percentile: {Q1:.2f}', xy=(Q1, 1), xytext=(Q1, 0.8),
                     arrowprops=dict(facecolor='yellow', shrink=0.05, width=0.5), horizontalalignment='center')
        plt.annotate(f'75th percentile: {Q3:.2f}', xy=(Q3, 1), xytext=(Q3, 1.2),
                     arrowprops=dict(facecolor='yellow', shrink=0.05, width=0.5), horizontalalignment='center')
        plt.annotate(f'Low: {lower_whisker_val:.2f}', xy=(lower_whisker_val, 1), xytext=(lower_whisker_val, 0.7),
                     arrowprops=dict(facecolor='red', shrink=0.05, width=0.5), horizontalalignment='center')
        plt.annotate(f'High: {upper_whisker_val:.2f}', xy=(upper_whisker_val, 1), xytext=(upper_whisker_val, 1.3),
                     arrowprops=dict(facecolor='red', shrink=0.05, width=0.5), horizontalalignment='center')

        st.pyplot(plt.gcf())
        plt.close()

        # AHA Boxplot
        
        key = (genderOptions[gender], ageOptions[age_group])

        if key in AHA_RANGES and columnName in AHA_RANGES[key]:
            low, normal, high, very_high = AHA_RANGES[key][columnName]

            plt.figure(figsize=(10, 3))  # Adjusted height for the box plot

            data = [val for val in [low, normal, high, very_high] if val is not None]
            if len(data) < 2:
                st.write(f"Insufficient data for {columnName}")
                continue

            plt.boxplot(data, vert=False, showmeans=False, showfliers=False,
                        whiskerprops={'linestyle': '--'}, medianprops={'color': 'green'})

            plt.axvline(x=user_input, color='red', linestyle='--')
            plt.scatter(user_input, 1, color='red', marker='o', label='Your Value')
            plt.legend()

            plt.title(f'American Heart Association (AHA) Data Boxplot for {columnName}')
            plt.xlabel(columnName)

            if low is not None:
                plt.annotate(f'Low: {low:.2f}', xy=(low, 1), xytext=(low, 1.2),
                             arrowprops=dict(facecolor='orange', shrink=0.05), horizontalalignment='center')
            if normal is not None:
                plt.annotate(f'Normal: {normal:.2f}', xy=(normal, 1), xytext=(normal, 1.2),
                             arrowprops=dict(facecolor='green', shrink=0.05), horizontalalignment='center')
            if high is not None:
                plt.annotate(f'High: {high:.2f}', xy=(high, 1), xytext=(high, 1.2),
                             arrowprops=dict(facecolor='orange', shrink=0.05), horizontalalignment='center')
            if very_high is not None:
                plt.annotate(f'Very High: {very_high:.2f}', xy=(very_high, 1), xytext=(very_high, 1.2),
                             arrowprops=dict(facecolor='red', shrink=0.05), horizontalalignment='center')

            st.pyplot(plt.gcf())
            plt.close()
        else:
            st.write(f"No data available for {columnName} for the selected gender and age group.")

# Main execution
df_c = load_files(False)
df_d, gender, age = ui_choose(df_c, False)
show_analysis(df_d, gender, age)
