import streamlit as st
import pandas as pd
import os
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="SAHC Comparison Tool", page_icon=":anatomical_heart:", layout="wide")

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

def create_legend():
    fig, ax = plt.subplots(figsize=(20, 1))  # Create a separate figure for the legend
    ax.axis('off')  # Hide axes

    # Create custom legend lines
    custom_lines = [
        Line2D([0], [0], color='grey', lw=10, alpha=0.5, label='Percentile Range'),
        Line2D([0], [0], color='green', lw=10, label='AHA Reference Range'),
        Line2D([0], [0], color='red', lw=10, label='High'),
        Line2D([0], [0], color='orange', lw=10, label='Low'),
        Line2D([0], [0], color='blue', marker='o', linestyle='None', label='Your Input')
    ]

    # Add the custom legend
    ax.legend(handles=custom_lines, loc='center', ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.5),
              handletextpad=2, columnspacing=2, fontsize=14)
    st.pyplot(fig)
    plt.close()

create_legend()

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

col1, col2, col3, col4, col5 = st.columns(5)

def ui_choose(df, debugging):
    genderOptions = {'Male': 1, 'Female': 2}
    ageOptions = {'20-40': 20, '40-60': 40, '60-80': 60, '80+': 80}

    with col1:
        gender = st.selectbox('Gender', list(genderOptions.keys()), placeholder="Choose an option")
    with col2:
        age_group = st.selectbox('Age group', list(ageOptions.keys()), placeholder="Choose an option")
    
    medCholOptions = {'Yes': 1, 'No': 2}
    medDiabOptions = {'Yes': 1, 'No': 2}
    medBPOptions = {'Yes': 1, 'No': 2}

    with col3:
        medChol = st.selectbox('Cholesterol medication', options=list(medCholOptions.keys()), placeholder="Choose an option")
    with col4:
        medDiab = st.selectbox('Diabetes medication', options=list(medDiabOptions.keys()), placeholder="Choose an option")
    with col5:
        medBP = st.selectbox('Blood pressure medication', options=list(medBPOptions.keys()), placeholder="Choose an option")

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
    return df2


# Number of elements
num_elements = 8

# Create placeholders for columns
cols = [st.empty() for _ in range(num_elements)]

def show_analysis(df):
    validation_labels = {
        'LBDHDD': "Value shold be between 20-100",
        'LBDLDL': "Value shuld be between 30-300",
        'LBXTC': "Value should be between 140-320",
        'LBXTR': "Value should be between 50-500",
        'LBXGLU': "Value should be between 50-150",
        'BPXOSY1': "Value should be between 90-200",
        'BPXODI1': "Value should be between 60-130",
        'BPXOPLS1': "Value should be between 50-150"
    }

    user_inputs = {}

    for i, column in enumerate(['LBDHDD', 'LBDLDL', 'LBXTC', 'LBXTR', 'LBXGLU', 'BPXOSY1', 'BPXODI1', 'BPXOPLS1']):

        with cols[i].container():

            col6, col7 = st.columns([2, 5])

            with col6:
                key = column
                columnName = NAME_MAP[column]

                global header
                header = columnName

                global header_color
                header_color = "black"

                global placeholder
                placeholder = st.empty()

                placeholder.write(f"#### {header}")

                unique_key = f"{key}_{i}"
                user_inputs[key] = st.number_input(f"{validation_labels[key]}", key=unique_key, step=1)

            with col7:
                columnName = NAME_MAP[column]
                user_input = user_inputs[column]  # Reference the user input from the dictionary

                # st.write('This is a <span style="color:red;">red text</span>', unsafe_allow_html=True)

                st.write(f"###")

                if columnName in AHA_RANGES:
                    low_value = AHA_RANGES[columnName][0]
                    high_value = AHA_RANGES[columnName][1]

                    if low_value is not None:
                        low_number = low_value
                    if high_value is not None:
                        high_number = high_value
                    if low_value is None:
                        low_number = None
                    if high_value is None:
                        high_number = None
                else:
                    st.write(f"No AHA prescribed range available for {columnName}.")
                    continue

                array = df[column].dropna()
                # st.dataframe(array,hide_index=True)
                sorted_array = np.sort(array)

                # Calculate the percentile
                if high_number == None:
                    high_percentile = 100
                else:
                    high_percentile = np.mean(sorted_array <= high_number) * 100
                if low_number == None:
                    low_percentile = 0
                else:
                    low_percentile = np.mean(sorted_array <= low_number) * 100

                if len(array.index) < 5:
                    st.markdown(f"Not enough data available for {columnName}.")
                    continue

                fig, ax = plt.subplots(figsize=(10, .5))

                # Grey line from 0 to 100
                ax.plot([0, 100], [.9, .9], color='grey', lw=10, alpha=0.5, label='Percentile Range')

                # Green line from low_percentile to high_percentile
                ax.plot([low_percentile, high_percentile], [0.65, 0.65], color='green', lw=10, label='Healthy Range')

                # Green line from low_percentile to high_percentile
                ax.plot([low_percentile, high_percentile], [0.65, 0.65], color='green', lw=10, label='Healthy Range')

                # Red line from high_percentile to 100
                if high_percentile < 100:
                    ax.plot([high_percentile, 100], [0.65, 0.65], color='red', lw=10, label='High')
                if low_percentile > 0:
                    ax.plot([0, low_percentile], [0.65, 0.65], color='orange', lw=10, label="Low")

                # Blue dot at user input position
                user_percentile = np.mean(sorted_array <= user_input) * 100
                if user_percentile > high_percentile or user_percentile < low_percentile:
                    ax.scatter(user_percentile, 0.9, color='red', zorder=5, label='Your Input')
                    header_color = "red"
                    placeholder.markdown(f"#### <span style='color:{header_color};'>{header}</span>", unsafe_allow_html=True)
                else:
                    ax.scatter(user_percentile, 0.9, color='blue', zorder=5, label='Your Input')


                ax.set_xlim(0, 100)
                ax.set_ylim(0.4, 1.1)
                ax.set_yticks([])
                ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
                ax.set_xlabel('Percentile (%)')

                st.pyplot(fig)
                plt.close()

# Main execution
df_c = load_files(False)
df_d = ui_choose(df_c, False)
show_analysis(df_d)