import streamlit as st
import pandas as pd
import os
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from fpdf import FPDF
import base64
import yagmail
import datetime

st.set_page_config(page_title="SAHC Comparison Tool", page_icon=":anatomical_heart:", layout="wide")

st.write('####')

DIR = os.getcwd()
PLOT_DIR = DIR + '/plots'
LOGO_DIR = DIR + '/logo/'
DATA_DIR = DIR + '/data/'
OUTPUT_DIR = DIR + '/output/'
SAHC_DATA_DIR = DIR + '/sahc_data/'

image_path = os.path.join(LOGO_DIR, 'CORE caps.svg')

def create_download_link(val, filename):
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'

header = st.container()
with header:
    col_image, col_empty, col_pdf = st.columns([1, 3, 1], vertical_alignment='bottom')
    with col_image:
        st.image(image_path, width=500)
    # with col_pdf:
    #     email = st.text_input("Download a PDF report!", placeholder='Email')

    #     if '@' in email:
    #         button_disabled = False
    #     else:
    #         button_disabled = True

    #     export_as_pdf = st.button("Export Report", disabled=button_disabled)

    #     if export_as_pdf:
    #         pdf = FPDF('P','mm','A4');
    #         pdf.add_page()
    #         pdf.set_font('Arial', 'B', 16)
    #         # pdf.cell(40, 10, report_text)
    #         pdf.image('LBDHDD.jpeg', x=5, y=30, w=200, h=25.633)  # Adjust 'x', 'y', 'w', and 'h' as needed

    #         pdf_file_path = 'test.pdf'  # Adjust the path and filename as needed
    #         pdf.output(pdf_file_path)
            
    #         # html = create_download_link(pdf.output(dest="S").encode("latin-1"), "test")
    #         # st.markdown(html, unsafe_allow_html=True)

    #         # Send the PDF via email with yagmail
    #         yag = yagmail.SMTP('sanaa.bhorkar@gmail.com', 'txhamunwqrefciwl', host='smtp.gmail.com', port=587, smtp_starttls=True, smtp_ssl=False)

    #         # Enclose the PDF
    #         yag.send(
    #             to=email,
    #             subject="Your SAHA Report",
    #             contents="Report attached.",
    #             attachments=['test.pdf']
    #         )

    #         # Close SMTP connection
    #         yag.close()

    #         with open("emails.txt", "a") as f: # save emails to a text file
    #             date = datetime.datetime.now()
    #             f.write(f"{date}, {email}\n")
    st.write("Use this tool to assess your risk for cardiovascular disease. Enter your cardiometabolic metrics and compare your markers with your peers.")

header.write("""<div class='fixed-header'/>""", unsafe_allow_html=True)

st.markdown(
    """
<style>
    div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
        position: sticky;
        top: 2.875rem;
        background-color: white;
        z-index: 999;
    }
    .fixed-header {
        border-bottom: 1px solid lightgrey;
    }
</style>
    """,
    unsafe_allow_html=True
)

darkgreen = "#75975e"
regugreen = "#95bb72"
lightgreen = "#c7ddb5"

USER_FILE = os.path.join(DATA_DIR, 'DEMO_P.XPT')
DIQ_FILE = os.path.join(DATA_DIR, 'P_DIQ.XPT')
BPQ_FILE = os.path.join(DATA_DIR, 'P_BPQ.XPT')
HDL_FILE = os.path.join(DATA_DIR, 'P_HDL.XPT')
TGL_FILE = os.path.join(DATA_DIR, 'P_TRIGLY.XPT')
TCH_FILE = os.path.join(DATA_DIR, 'P_TCHOL.XPT')
GLU_FILE = os.path.join(DATA_DIR, 'P_GLU.XPT')
GHB_FILE = os.path.join(DATA_DIR, 'P_GHB.XPT')
BPX_FILE = os.path.join(DATA_DIR, 'P_BPXO.XPT')
CBC_FILE = os.path.join(DATA_DIR, 'P_CBC.XPT')
BMX_FILE = os.path.join(DATA_DIR, 'P_BMX.XPT')
# SAHC_FILE = os.path.join(SAHC_DATA_DIR, 'tblCleanedConcatNoPID.csv')
SAHC_FILE = os.path.join(SAHC_DATA_DIR, 'merged_data_noPID.csv')

UNITS_MAP = {
    'LBDHDD': "mg/dL", 'LBDLDL': "mg/dL", 'LBXTC': "mg/dL", 'LBXTR': "mg/dL", 'LBXGH': "%",
    'LBXGLU': "mg/dL", 'BPXOSY1': "mmHg", 'BPXODI1': "mmHg", 'LBXHGB': "g/dL", 'TotHDLRat': "", 'BMXBMI': "kg/m\u00b2"
}


NAME_MAP = {
    'LBXTR': 'Triglycerides', 'LBDHDD': 'HDL', 'LBDLDL': 'LDL',
    'LBXTC': 'Total Cholesterol', 'LBXGLU': 'Fasting Glucose',
    'BPXOSY1': 'Systolic Blood Pressure', 'BPXODI1': 'Diastolic Blood Pressure', 
    'TotHDLRat': 'Total Cholesterol to HDL Ratio', 'LBXHGB': 'Hemoglobin', 'LBXGH': 'Hemoglobin A1C', 
    'BMXBMI': 'Body Mass Index'
}

AHA_RANGES = {
    'Triglycerides (mg/dL)': (None, 150),
    'HDL (mg/dL)': (40, 60),
    'LDL (mg/dL)': (None, 100),
    'Total Cholesterol (mg/dL)': (140, 160),
    'Fasting Glucose (mg/dL)': (None, 100),
    'Systolic Blood Pressure (mmHg)': (None, 120),
    'Diastolic Blood Pressure (mmHg)': (None, 80),
    'Total Cholesterol to HDL Ratio': (3.5, 5),
    'Hemoglobin A1C (%)': (None, 5.7),
    'Body Mass Index (kg/m\u00b2)': (None, 25)
}

AHA_RANGES = {
    'Triglycerides (mg/dL)': ("Normal", 150, "Borderline", 200, "High"),
    'HDL (mg/dL)': ("Low", 40, "Normal", 60, "High"),
    'LDL (mg/dL)': ("Optimal", 100, "", 160, "High"),
    'Total Cholesterol (mg/dL)': ("Optimal", 150, "", 200, "High"),
    'Fasting Glucose (mg/dL)': ("Normal", 100, "Borderline", 125, "High"),
    'Systolic Blood Pressure (mmHg)': ("Normal", 120, "High", None, None, None),
    'Diastolic Blood Pressure (mmHg)': ("Normal", 80, "High", None, None, None),
    'Total Cholesterol to HDL Ratio': ("Low", 3.5, "Normal", 5, "High"),
    'Hemoglobin A1C (%)': ("Normal", 5.7, "Borderline", 6.4, "High"),
    'Body Mass Index (kg/m\u00b2)': ("Low", 18.5, "Normal", 25, "High")
}

def map_age_to_group(age):
    if 0 <= age <= 2:
        return 1
    elif 3 <= age <= 5:
        return 4
    elif 6 <= age <= 13:
        return 9
    elif 14 <= age <= 18:
        return 16
    elif 19 <= age <= 33:
        return 19
    elif 34 <= age <= 48:
        return 34
    elif 49 <= age <= 64:
        return 49
    elif 65 <= age <= 78:
        return 65
    else:
        return 79

aboutMe_placeholder = st.empty()
aboutMe_placeholder.markdown(f"### <u>About Me</u>", unsafe_allow_html=True)

aboutMe_expand = st.expander("Enter your information here", expanded=True)

genderOptions = {'Male': 1, 'Female': 2}
ageOptions = {'19-33': 19, '34-48': 34, '49-64': 49, '65-78': 65,'79+': 79}
dataSelection = ['NHANES', 'SAHC']

with aboutMe_expand:
    col1, col12, col2, col22, col3, col32, col4, col42 = st.columns([.2, .1, .2, .1, .2, .1, .2, .1], vertical_alignment='top')

    medCholOptions = {'Yes': 1, 'No': 2}
    medDiabOptions = {'Yes': 1, 'No': 2}
    medBPOptions = {'Yes': 1, 'No': 2}


    with col1:
        gender = st.selectbox('Gender', list(genderOptions.keys()))
    with col2:
        age_group = st.selectbox('Age group', list(ageOptions.keys()), index=2)
    with col3:
        ethnicity = st.selectbox('Ethnicity', ['Other', 'South Asian'])
    with col4:
        medications_select = st.multiselect(label="Select your medication use", options=['Cholesterol', 'Diabetes', 'Blood Pressure'])

        medChol = 'No'
        medDiab = 'No'
        medBP = 'No'

        if 'Cholesterol' in medications_select:
            medChol = 'Yes'
        if 'Diabetes' in medications_select:
            medDiab = 'Yes'
        if 'Blood Pressure' in medications_select:
            medBP = 'Yes'

aboutMe_label = f"About me: {gender}, {age_group}, {ethnicity}, Cholesterol meds: {medChol}, Diabetes meds: {medDiab}, Blood Pressure meds: {medBP}"
aboutMe_placeholder.markdown(f"### <u>{aboutMe_label}</u>", unsafe_allow_html=True)
                                            
# @st.cache_data
def load_files(debugging):
    if ethnicity == 'Other':
        df_user = pd.read_sas(USER_FILE, format='xport')
        df_diq = pd.read_sas(DIQ_FILE, format='xport')
        df_bpq = pd.read_sas(BPQ_FILE, format='xport')
        df_hdl = pd.read_sas(HDL_FILE, format='xport')
        df_tgl = pd.read_sas(TGL_FILE, format='xport')
        df_tch = pd.read_sas(TCH_FILE, format='xport')
        df_glu = pd.read_sas(GLU_FILE, format='xport')
        df_ghb = pd.read_sas(GHB_FILE, format='xport')
        df_bpx = pd.read_sas(BPX_FILE, format='xport')
        df_bmx = pd.read_sas(BMX_FILE, format='xport')

        df_combined = df_user[['SEQN', 'RIAGENDR', 'RIDAGEYR', 'RIDRETH3']]
        df_combined = pd.merge(df_combined, df_diq[['SEQN', 'DIQ010', 'DIQ160', 'DIQ050', 'DIQ070']], on='SEQN', how='left')
        df_combined = pd.merge(df_combined, df_bpq[['SEQN', 'BPQ090D', 'BPQ100D', 'BPQ040A', 'BPQ050A', 'BPQ020']], on='SEQN', how='left')
        df_combined = pd.merge(df_combined, df_hdl[['SEQN', 'LBDHDD']], on='SEQN', how='left')
        df_combined = pd.merge(df_combined, df_tgl[['SEQN', 'LBXTR', 'LBDLDL']], on='SEQN', how='left')
        df_combined = pd.merge(df_combined, df_tch[['SEQN', 'LBXTC']], on='SEQN', how='left')
        df_combined = pd.merge(df_combined, df_glu[['SEQN', 'LBXGLU']], on='SEQN', how='left')
        df_combined = pd.merge(df_combined, df_ghb[['SEQN', 'LBXGH']], on='SEQN', how='left')
        df_combined = pd.merge(df_combined, df_bpx[['SEQN', 'BPXOSY1', 'BPXODI1', 'BPXOPLS1', 'BPXOSY2', 'BPXODI2', 'BPXOPLS2', 'BPXOSY3', 'BPXODI3', 'BPXOPLS3']], on='SEQN', how='left')
        df_combined = pd.merge(df_combined, df_bmx[['SEQN', 'BMXBMI']], on='SEQN', how='left')

        # df_combined['Age_Group'] = df_combined['RIDAGEYR'].apply(lambda age: 20 * int(age / 20))
        df_combined['Age_Group'] = df_combined['RIDAGEYR'].apply(map_age_to_group)
        df_combined['TotHDLRat'] =  df_combined['LBXTC'] / df_combined['LBDHDD']

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
    else:
        df_combined = pd.read_csv(SAHC_FILE)

        df_combined = df_combined.rename(columns={
            'gender': 'RIAGENDR',
            'race': 'RIDRETH3',
            'age': 'RIDAGEYR',
            'bpSyst': 'BPXOSY1',
            'bpDiast': 'BPXODI1',
            'cholTot': 'LBXTC',
            'cholLDL': 'LBDLDL',
            'cholHDL': 'LBDHDD',
            'cholTrig': 'LBXTR',
            'Total:HDL': 'TotHDLRat',
            'bloodSugar': 'LBXGLU',
            'hg1ac': 'LBXGH',
            'bmi': 'BMXBMI'
        })

        df_combined['Age_Group'] = df_combined['RIDAGEYR'].apply(map_age_to_group)

        return df_combined

def get_next_age_group(age_group, direction='up'):
    age_groups = list(ageOptions.keys())
    current_index = age_groups.index(age_group)
    
    if direction == 'up':
        if current_index < len(age_groups) - 1:
            return age_groups[current_index + 1]
        else:
            return age_groups[current_index - 1]
    elif direction == 'down':
        if current_index > 0:
            return age_groups[current_index - 1]
        else:
            return age_groups[current_index + 1]

def ui_choose(df, debugging):
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

    if ethnicity == 'South Asian':
        df2 = df2[df2['cholMeds'].isin(medCholFilter)]
    elif medChol == 'Yes':
        df2 = df2[df2['BPQ100D'].isin(medCholFilter)]
    elif medChol == 'No':
        df2 = df2[df2['BPQ090D'].isin(medCholFilter) | df2['BPQ100D'].isin(medCholFilter)]
    
    if ethnicity == 'South Asian':
        df2 = df2[df2['diabMeds'].isin(medDiabFilter)]
    else:
        df2 = df2[df2['DIQ070'].isin(medDiabFilter)]

    if ethnicity == 'South Asian':
        df2 = df2[df2['bpMeds'].isin(medBPFilter)]
    elif medBP == 'Yes':
        df2 = df2[df2['BPQ040A'].isin(medBPFilter)]
    elif medBP == 'No':
        df2 = df2[df2['BPQ020'].isin(medBPFilter) | df2['BPQ040A'].isin(medBPFilter)]

    # st.write(f"{len(df2)} records found.")

    if len(df2) < 15:
        next_age_group = get_next_age_group(age_group)
        ageFilter.append(ageOptions[next_age_group])
        df2 = df[df['RIAGENDR'].isin(genderFilter)]
        df2 = df2[df2['Age_Group'].isin(ageFilter)]

        if ethnicity == 'South Asian':
            df2 = df2[df2['cholMeds'].isin(medCholFilter)]
        elif medChol == 'Yes':
            df2 = df2[df2['BPQ100D'].isin(medCholFilter)]
        elif medChol == 'No':
            df2 = df2[df2['BPQ090D'].isin(medCholFilter) | df2['BPQ100D'].isin(medCholFilter)]
        
        if ethnicity == 'South Asian':
            df2 = df2[df2['diabMeds'].isin(medDiabFilter)]
        else:
            df2 = df2[df2['DIQ070'].isin(medDiabFilter)]

        if ethnicity == 'South Asian':
            df2 = df2[df2['bpMeds'].isin(medBPFilter)]
        elif medBP == 'Yes':
            df2 = df2[df2['BPQ040A'].isin(medBPFilter)]
        elif medBP == 'No':
            df2 = df2[df2['BPQ020'].isin(medBPFilter) | df2['BPQ040A'].isin(medBPFilter)]

        # st.write(f"{len(df2)} records found after expanding age group.")

    return df2

@st.dialog("Your marker compared to your peers")
def popup(acro, column, user_input, user_percentile, gender, age_range, med, on_med, prob, value, p25, p50, p75, p90, side, suffix):
    
    if side == 'greater':
        st.write(f"The estimated probability of an optimal {column} ≥ {value} {UNITS_MAP[acro]} for a {gender.lower()} aged between {age_range} years is {prob * 100: .0f}%.")
    else:
        st.write(f"The estimated probability of an optimal {column} ≤ {value} {UNITS_MAP[acro]} for a {gender.lower()} aged between {age_range} years is {prob * 100: .0f}%.")
    if on_med:
        st.write(f"An {column} of {user_input} is at the {user_percentile: .0f}{suffix} percentile for a {gender.lower()} aged between {age_range} years who is on {med}-lowering medication.")
    else:
        st.write(f"An {column} of {user_input} is at the {user_percentile: .0f}{suffix} percentile for a {gender.lower()} aged between {age_range} years who is not on {med}-lowering medication.")

    data = {
        'Percentile': [f'{column} Level'],
        '25th': [p25],
        '50th': [p50],
        '75th': [p75],
        '90th': [p90]
    }

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(data)
    st.dataframe(df, hide_index=True)

    st.write("Based on the AHA guidlelines, the optimal value for HDL >= 40 for males. “>= 45 for females")

def show_analysis(df):
    st.markdown(f"### <u>My risk profile markers</u>", unsafe_allow_html=True)
    
    # Number of elements
    num_elements = 10

    # Create placeholders for columns
    cols = [st.empty() for _ in range(num_elements)]
        
    validation_labels = {
        'LBDHDD': "Value should be between 20-100",
        'LBDLDL': "Value should be between 30-300",
        'LBXTC': "Value should be between 100-320",
        'LBXTR': "Value should be between 50-500",
        'LBXGLU': "Value should be between 50-150",
        'BPXOSY1': "Value should be between 90-200",
        'BPXODI1': "Value should be between 60-130",
        'TotHDLRat': "Value should be between 0.5-10",
        'LBXGH': "Value should be between 1 and 20",
        'BMXBMI': "Value should be between 10 and 50"
    }

    user_inputs = {}

    for i, column in enumerate(['LBXTC', 'LBDLDL', 'LBDHDD', 'LBXTR', 'TotHDLRat', 'LBXGLU', 'LBXGH', 'BMXBMI', 'BPXOSY1', 'BPXODI1']):

        with cols[i].container():

            # col6, col7, col8, col9, col10, col11 = st.columns([.25, 1.6, .3, 5, 5, .25], vertical_alignment='bottom')
            # col6, col7, col8, col9, col10, col11 = st.columns([0.1, 0.15, 0.05, 0.3, 0.3, 0.1], vertical_alignment='bottom')
            # col6, col7, col8, col9, col10, col11 = st.columns([0.1, 0.12, 0.03, 0.325, 0.325, 0.1], vertical_alignment='bottom')
            col6, col7, col8, col9, col10 = st.columns([0.05, 0.17, 0.05, 0.68, 0.05], vertical_alignment='center')
            
            columnName = f"{NAME_MAP[column]} ({UNITS_MAP[column]})"
            if column == 'TotHDLRat':
                columnName = f"{NAME_MAP[column]}"

            with col7:
                key = column


                global header
                header = columnName

                global header_color
                header_color = "black"

                global placeholder
                placeholder = st.empty()

                placeholder.write(f"#### {header}")

                unique_key = f"{key}_{i}"
                user_inputs[key] = st.number_input('user input', key=unique_key, step=1, label_visibility="collapsed")

                st.caption(f'{validation_labels[key]}')
            
            with col8:

                
                st.markdown("""
                    <style>
                        button[kind="primary"] {
                            background-color: white;
                            font-weight: bold;
                            color: black;
                            width: 50px;
                            border: 0px solid green;
                        }

                        button[kind="primary"]:hover {
                            background-color: white; /* Slightly different background on hover */
                            color: #5D2C82; /* Ensure text color remains visible on hover */
                        }
                    </style>
                    """, unsafe_allow_html=True)
                # st.write("#### hi")
                more_info = st.button(label='ⓘ', key=column, type='primary')
                # st.caption(' Press for more info')

            with col9:
                user_input = user_inputs[column]  # Reference the user input from the dictionary

                array = df[column].dropna()
                if len(array.index) < 5:
                    st.markdown(f"Not enough data available for {columnName}.")
                    continue

                if user_input == 0:
                    # st.write(f"### ")
                    st.write(f"Please enter a value for {columnName}.")
                    continue

                # st.write(f"####")

                if columnName in AHA_RANGES:
                    low_number = AHA_RANGES[columnName][1]
                    high_number = AHA_RANGES[columnName][3]
                else:
                    st.write(f"No AHA prescribed range available for {columnName}.")
                    continue
                if columnName == 'HDL (mg/dL)' and gender == 'Female':
                    low_number = 50

                sorted_array = np.sort(array)

                # Calculate the percentile
                if high_number == None:
                    high_number = 1000
                    high_percentile = 99
                else:
                    high_percentile = np.mean(sorted_array <= high_number) * 100

                low_percentile = np.mean(sorted_array <= low_number) * 100

                fig, ax = plt.subplots(figsize=(16, 1))

                # dot at user input position
                user_percentile = np.mean(sorted_array <= user_input) * 100
                if int(user_percentile) == 100:
                    user_percentile == 99.99
                if (user_input > high_number or user_input < low_number) and column != 'LBDHDD':
                    # ax.scatter(user_percentile, 0.85, color='red', zorder=5, label='Your Input')
                    header_color = "red"
                    placeholder.markdown(f"#### <span style='color:{header_color};'>{header}</span>", unsafe_allow_html=True)
                # else:

                if AHA_RANGES[columnName][0] == 'Optimal':
                    ax.plot([high_percentile + 1, 100], [0.775, 0.775], color=darkgreen, lw=20, label='High')
                    ax.plot([0, low_percentile - 1], [0.775, 0.775], color=regugreen, lw=20, label="Low")
                    ax.plot([low_percentile, high_percentile], [0.775, 0.775], color=lightgreen, lw=20, label='Healthy Range')

                    if user_percentile < low_percentile:
                        ax.scatter(user_percentile, 0.9, color=regugreen, zorder=5, label='Your Input', s=600, edgecolors=['black'])
                    elif user_percentile > high_percentile:
                        ax.scatter(user_percentile, 0.9, color=darkgreen, zorder=5, label='Your Input', s=600, edgecolors=['black'])
                    else:
                        ax.scatter(user_percentile, 0.9, color=lightgreen, zorder=5, label='Your Input', s=600, edgecolors=['black'])

                else:
                    # lines from low_percentile to high_percentile
                    ax.plot([low_percentile, high_percentile], [0.775, 0.775], color=regugreen, lw=20, label='Healthy Range')

                    if high_percentile < 100:
                        ax.plot([high_percentile + 1, 100], [0.775, 0.775], color=darkgreen, lw=20, label='High')
                    if low_percentile > 0:
                        ax.plot([0, low_percentile - 1], [0.775, 0.775], color=lightgreen, lw=20, label="Low")

                    if user_percentile < low_percentile:
                        ax.scatter(user_percentile, 0.9, color=lightgreen, zorder=5, label='Your Input', s=600, edgecolors=['black'])
                    elif user_percentile > high_percentile:
                        ax.scatter(user_percentile, 0.9, color=darkgreen, zorder=5, label='Your Input', s=600, edgecolors=['black'])
                    else:
                        ax.scatter(user_percentile, 0.9, color=regugreen, zorder=5, label='Your Input', s=600, edgecolors=['black'])

                ax.set_xlim(0, 100)
                ax.set_ylim(0.4, 1.1)
                ax.set_yticks([])
                # ax.xaxis.tick_top()  
                ax.xaxis.set_label_position('top') 
                # tick_positions = [0, 100]

                # tick_labels = ['0%ile', '99%ile']
                ax.set_xticks([])
                # ax.set_xticklabels(tick_labels)

                # ax.set_xlabel('Percentile (%)')

                ax.set_title(columnName)

                plt.annotate(f'{user_input}', xy=(user_percentile, 0.875), xytext=(user_percentile, 0.85),
                              horizontalalignment='center', color = 'black', zorder=10, weight='bold')
                if int(user_percentile) == 100:
                    user_percentile = 99
                plt.annotate(f'({user_percentile: .0f}%ile)', xy=(user_percentile, 0.9), xytext=(user_percentile + 5, 0.935), 
                             horizontalalignment='center')

                if high_number < 1000:
                    plt.annotate(f'{AHA_RANGES[columnName][4]}', xy=(high_percentile, 0.675), xytext=(high_percentile, 0.5),
                                  horizontalalignment='left', weight='bold')
                    plt.annotate(f'>{high_number}', xy=(high_percentile, 0.675), xytext=(high_percentile, 0.35),
                                  horizontalalignment='left')
                if low_number > 0:
                    plt.annotate(f'{AHA_RANGES[columnName][0]}', xy=(low_percentile - 1, 0.675), xytext=(low_percentile - 1, 0.5),
                                  horizontalalignment='right', weight='bold')
                    plt.annotate(f'<{low_number}', xy=(low_percentile - 1, 0.657), xytext=(low_percentile - 1, 0.35),
                                  horizontalalignment='right')
                plt.annotate(f'{AHA_RANGES[columnName][2]}', xy=(low_percentile, 0.675), xytext=(low_percentile, 0.5),
                                  horizontalalignment='left', weight='bold')
                plt.annotate(f'{low_number}-{high_number}', xy=(low_percentile, 0.675), xytext=(low_percentile, 0.35),
                                  horizontalalignment='left')

                for spine in ax.spines.values():
                    spine.set_visible(False)

                # fig.patch.set_alpha(0) make background transparent
                # ax.patch.set_alpha(0)

                st.pyplot(fig)

                fig.savefig(f'{column}.jpeg', bbox_inches='tight')
                plt.close(fig)

                digit = user_percentile % 10
                suffix = 'th'

                if digit == 1:
                    suffix = 'st'
                elif digit == 2:
                    suffix == 'nd'
                elif digit == 3:
                    suffix == 'rd'

                kde = sm.nonparametric.KDEUnivariate(array)
                kde.fit(bw='scott')  # You can also specify bandwidth manually, e.g., bw=0.5

                # Generate a range of HDL values to estimate the CDF
                hdl_range = np.linspace(min(array), max(array), 1000)

                # Evaluate the KDE over the range
                pdf_values = kde.evaluate(hdl_range)

                # Calculate the CDF by integrating the PDF
                cdf_values = np.cumsum(pdf_values) * (hdl_range[1] - hdl_range[0])

                # Find the CDF value at HDL = 35
                if "HDL" in columnName:
                    value = low_number
                else:
                    value = high_number
                
                cdf_at_35 = np.interp(value, hdl_range, cdf_values)

                # Probability of HDL >= 35
                if columnName == 'HDL (mg/dL)':
                    prob = 1 - cdf_at_35
                else:
                    prob = cdf_at_35

                # # Display the result in Streamlit
                # st.write(f"Probability of HDL >= {value}: {probability_gte_35 * 100: .0f}%")
                # st.write(f"Probability of HDL = {value}: {prob_density_value[0]}")

                percentile_25 = int(np.percentile(sorted_array, 25))
                percentile_50 = int(np.percentile(sorted_array, 50))
                percentile_75 = int(np.percentile(sorted_array, 75))
                percentile_90 = int(np.percentile(sorted_array, 90))

                popup_column = NAME_MAP[column]

                if more_info:
                    if "HDL" in columnName:
                        popup(column, popup_column, user_input, user_percentile, gender, age_group, "cholesterol", medChol, prob, value, 
                              percentile_25, percentile_50, percentile_75, percentile_90, "greater", suffix)
                    elif "DL" in columnName or "Trig" in columnName or "Chol" in columnName:
                        popup(column, popup_column, user_input, user_percentile, gender, age_group, "cholesterol", medChol, prob, value, 
                              percentile_25, percentile_50, percentile_75, percentile_90, "less", suffix)
                    elif "Glucose" in columnName:
                        popup(column, popup_column, user_input, user_percentile, gender, age_group, "blood sugar", medDiab, prob, value, 
                              percentile_25, percentile_50, percentile_75, percentile_90, "less", suffix)   
                    else:
                        popup(column, popup_column, user_input, user_percentile, gender, age_group, "blood pressure", medBP, prob, value, 
                              percentile_25, percentile_50, percentile_75, percentile_90, "less", suffix)

# Main execution
df_c = load_files(False)
df_d = ui_choose(df_c, False)
show_analysis(df_d)

st.divider()
st.markdown('<div style="text-align: center"> Please email <a href="mailto:sanaa.bhorkar@gmail.com">sanaa.bhorkar@gmail.com</a> with any feedback! </div>', unsafe_allow_html=True)
st.markdown('<div style="text-align: center"> Version 0.3 </div>', unsafe_allow_html=True)