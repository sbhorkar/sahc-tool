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
from matplotlib.patches import Rectangle
from collections import deque 

st.set_page_config(page_title="CORE Comparison Tool", page_icon=":anatomical_heart:", layout="wide")

# st.caption('####')
@st.cache_resource()
def update_visit_count():
    count++
    return count
#     return deque()

DIR = os.getcwd()
PLOT_DIR = DIR + '/plots'
LOGO_DIR = DIR + '/logo/'
DATA_DIR = DIR + '/data/'
OUTPUT_DIR = DIR + '/output/'
SAHC_DATA_DIR = DIR + '/sahc_data/'
VERSION = 1.6

image_path = os.path.join(LOGO_DIR, 'CORE larger size.svg')

def create_download_link(val, filename):
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'

header = st.container()
with header:
    col_image, col_empty, col_color = st.columns([1, 2, 1], vertical_alignment='top')
    with col_image:
        st.image(image_path)
    with col_color:
        colorblind_mode = st.toggle("High Contrast Mode")
        st.caption("Contrast and colorblindness improvements")
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
    st.write("CORE evaluates your cardiometabolic risk profile and compares your markers against peers based on your gender, age, and ethnicity.")
    st.divider()
header.write("""<div class='fixed-header'/>""", unsafe_allow_html=True)

st.markdown(""" 
        <style>
        *:not(fixed-header)::before, 
            *|not(fixed-header)::after {
           box-sizing: inherit;
        }
        </style>""", unsafe_allow_html=True)

# st.markdown(
#     """
# <style>
#     div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
#         position: sticky;
#         top: 3.3rem;
#         background-color: white;
#         z-index: 9999;
#     }
#     .fixed-header {
#         border-bottom: 1px solid lightgrey;
#     }
# </style>
#     """,
#     unsafe_allow_html=True
# )

red = "#ee3942"
orange = "#fec423"
green = "#67bd4a"
darkgreen = "#75975e"
regugreen = "#9dba82"
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
MCQ_FILE = os.path.join(DATA_DIR, 'P_MCQ.XPT')
SAHC_FILE = os.path.join(SAHC_DATA_DIR, 'merged_data_noPID.csv')

UNITS_MAP = {
    'LBDHDD': "mg/dL", 'LBDLDL': "mg/dL", 'LBXTC': "mg/dL", 'LBXTR': "mg/dL", 'LBXGH': "%",
    'LBXGLU': "mg/dL", 'BPXOSY1': "mmHg", 'BPXODI1': "mmHg", 'LBXHGB': "g/dL", 'TotHDLRat': "", 'BMXBMI': ""
}


NAME_MAP = {
    'LBXTC': 'Total Cholesterol', 'LBDLDL': 'LDL', 'LBDHDD': 'HDL', 
    'LBXTR': 'Triglycerides', 'TotHDLRat': 'Total Cholesterol to HDL Ratio', 
    'LBXGLU': 'Fasting Glucose', 'LBXGH': 'Hemoglobin A1C',
    'BMXBMI': 'Body Mass Index', 'BPXOSY1': 'Systolic Blood Pressure', 'BPXODI1': 'Diastolic Blood Pressure', 
}

DROPDOWN_SELECTION = {
    'Total Cholesterol': 'LBXTC', 'LDL': 'LBDLDL', 'HDL': 'LBDHDD',
    'Triglycerides': 'LBXTR', 'Total Cholesterol to HDL Ratio': 'TotHDLRat',
    'Fasting Glucose': 'LBXGLU', 'Hemoglobin A1C': 'LBXGH',
    'Body Mass Index': 'BMXBMI', 'Systolic Blood Pressure': 'BPXOSY1', 'Diastolic Blood Pressure': 'BPXODI1', 
}

AHA_RANGES = {
    'Triglycerides (mg/dL)': ("Optimal", 150, "Borderline", 200, "At risk", None, None),
    'HDL (mg/dL)': ("At risk", 40, "Optimal", None, None, None, None),
    'LDL (mg/dL)': ("Optimal", 100, "Borderline", 160, "At risk", None, None),
    'Total Cholesterol (mg/dL)': ("Optimal", 150, "Borderline", 200, "At risk", None, None),
    'Fasting Glucose (mg/dL)': ("Optimal", 100, "Borderline", 125, "At risk", None, None),
    'Systolic Blood Pressure (mmHg)': ("Optimal", 120, "At risk", None, None, None, None),
    'Diastolic Blood Pressure (mmHg)': ("Optimal", 80, "At risk", None, None, None, None),
    'Total Cholesterol to HDL Ratio ()': ("Optimal", 3.5, "Borderline", 5, "At risk", None, None),
    'Hemoglobin A1C (%)': ("Optimal", 5.7, "Borderline", 6.4, "At risk", None, None),
    'Body Mass Index': ("Low", 18.5, "Optimal", 25, "Borderline", 30, "At risk")
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

aboutMe_expand = st.expander("Please enter your details below", expanded=True)

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
        ethnicity = st.selectbox('Ethnicity', ['Non-South Asian', 'South Asian'])
    with col4:
        medications_select = st.multiselect(label="Select your medication use", options=['None', 'Cholesterol', 'Diabetes', 'Blood Pressure'])

        medChol = 'No'
        medDiab = 'No'
        medBP = 'No'

        if 'Cholesterol' in medications_select:
            medChol = 'Yes'
        if 'Diabetes' in medications_select:
            medDiab = 'Yes'
        if 'Blood Pressure' in medications_select:
            medBP = 'Yes'

meds = ", ".join(medications_select)
if meds == '':
    meds = "None"

aboutMe_label = f"About me: {gender}, Age: {age_group}, {ethnicity}, Current medications: {meds}"
aboutMe_placeholder.markdown(f"### <u>{aboutMe_label}</u>", unsafe_allow_html=True)
                                            
# @st.cache_data
def load_files(debugging):
    if ethnicity == 'Non-South Asian':
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
        df_mcq = pd.read_sas(MCQ_FILE, format='xport')

        df_combined = df_user[['SEQN', 'RIAGENDR', 'RIDAGEYR', 'RIDRETH3']]
        df_combined = pd.merge(df_combined, df_diq[['SEQN', 'DIQ010', 'DIQ160', 'DIQ050', 'DIQ070']], on='SEQN', how='left')
        df_combined = pd.merge(df_combined, df_bpq[['SEQN', 'BPQ090D', 'BPQ100D', 'BPQ040A', 'BPQ020']], on='SEQN', how='left')
        df_combined = pd.merge(df_combined, df_hdl[['SEQN', 'LBDHDD']], on='SEQN', how='left')
        df_combined = pd.merge(df_combined, df_tgl[['SEQN', 'LBXTR', 'LBDLDL']], on='SEQN', how='left')
        df_combined = pd.merge(df_combined, df_tch[['SEQN', 'LBXTC']], on='SEQN', how='left')
        df_combined = pd.merge(df_combined, df_glu[['SEQN', 'LBXGLU']], on='SEQN', how='left')
        df_combined = pd.merge(df_combined, df_ghb[['SEQN', 'LBXGH']], on='SEQN', how='left')
        df_combined = pd.merge(df_combined, df_bpx[['SEQN', 'BPXOSY1', 'BPXODI1', 'BPXOPLS1', 'BPXOSY2', 'BPXODI2', 'BPXOPLS2', 'BPXOSY3', 'BPXODI3', 'BPXOPLS3']], on='SEQN', how='left')
        df_combined = pd.merge(df_combined, df_bmx[['SEQN', 'BMXBMI']], on='SEQN', how='left')
        df_combined = pd.merge(df_combined, df_mcq[['SEQN', 'MCQ160E']], on='SEQN', how='left')

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

    # st.dataframe(df, hide_index=True)

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
    
    if len(df2) == 0:
        st.write(f"No records found to compare.")
    else:
        st.write(f"{len(df2)} records found.")

    return df2

@st.dialog(" ")
def popup(acro, column, user_input, user_percentile, gender, race, age_range, med, on_med, prob, value, p25, p50, p75, p90, side, suffix, low_number, high_number):
    st.header(f"Your {column} value compared to others in your peer group")
    if on_med == 'Yes':
        st.write(f"Your {column} of {user_input:.1f} is at the {user_percentile: .0f}{suffix} percentile for a {race} {gender.lower()} aged between {age_range} years who is ON {med}-lowering medication.")
    else:
        st.write(f"Your {column} of {user_input:.1f} is at the {user_percentile: .0f}{suffix} percentile for a {race} {gender.lower()} aged between {age_range} years who is NOT ON {med}-lowering medication.")
    
    if side == 'greater':
        st.write(f"The estimated probability of an optimal {column} ≥ {value} {UNITS_MAP[acro]} for a {race} {gender.lower()} aged between {age_range} years is {prob * 100: .0f}%.")
    else:
        st.write(f"The estimated probability of an optimal {column} ≤ {value} {UNITS_MAP[acro]} for a {race} {gender.lower()} aged between {age_range} years is {prob * 100: .0f}%.")
    
    if STEP_SIZE[acro] == 0.1:
        data = {
            'Percentile': [f'{column} Level'],
            '25th': [round(p25, 1)],
            '50th': [round(p50, 1)],
            '75th': [round(p75, 1)],
            '90th': [round(p90, 1)]
        }
    else:
        data = {
        'Percentile': [f'{column} Level'],
        '25th': [int(p25)],
        '50th': [int(p50)],
        '75th': [int(p75)],
        '90th': [int(p90)]
    }

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(data)
    st.dataframe(df, hide_index=True)

    if 'Triglyc' in column or 'LDL' in column or 'Total Cholesterol (mg/dL)' in column or 'Glucose' in column or 'Pressure' in column or 'A1C' in column:
        st.write(f"Based on the American Heart Association guidlelines, the optimal value for {column} is ≤ {low_number} for {gender.lower()}s.")
    elif 'to HDL' in column or 'Body Mass' in column:
        st.write(f"Based on the American Heart Association guidlelines, the optimal value for {column} is between {low_number}-{high_number} for {gender.lower()}s.")
    elif 'HDL' in column:
        st.write(f"Based on the American Heart Association guidlelines, the optimal value for {column} is ≥ {low_number} for {gender.lower()}s.")

# metric_list = get_app_queue()
if 'metric_list' not in st.session_state:
    st.session_state.metric_list = deque()

if 'visit_count' not in st.session_state:
    visit_count = update_visit_count()

st.write(visit_count)

def show_analysis(df):
    # st.write("show_analysis entered", datetime.datetime.now())
    validation_labels = {
        'LBDHDD': "Value should be between 20-100",
        'LBDLDL': "Value should be between 30-300",
        'LBXTC': "Value should be between 100-320",
        'LBXTR': "Value should be between 50-300",
        'LBXGLU': "Value should be between 50-150",
        'BPXOSY1': "Value should be between 90-200",
        'BPXODI1': "Value should be between 60-130",
        'TotHDLRat': "Value should be between 0.5-10",
        'LBXGH': "Value should be between 0 and 20",
        'BMXBMI': "Value should be between 10 and 50"
    }

    validation_range = {
        'LBDHDD': [20, 100],
        'LBDLDL': [30, 300],
        'LBXTC': [100, 320],
        'LBXTR': [50, 300],
        'LBXGLU': [50, 150],
        'BPXOSY1': [90, 200],
        'BPXODI1': [60,130],
        'TotHDLRat': [0.5, 10.0],
        'LBXGH': [0.0, 20.0],
        'BMXBMI': [10, 50]
    }

    user_inputs = {}

    st.markdown(f"### <u>My risk profile markers</u>", unsafe_allow_html=True)

    col_dropdown, col_empty, col_input, col_input2, col_empty = st.columns([0.25, 0.1, 0.2, 0.2, 0.25], gap='medium', vertical_alignment='center')

    with col_dropdown:
        metric = st.selectbox('Metric', list(DROPDOWN_SELECTION.keys()))
    
    column = DROPDOWN_SELECTION[metric]

    if column == 'BMXBMI':
        with col_input:
            weight = st.number_input('Weight (lbs)', key="weight", step=STEP_SIZE[column], value = 1)
        with col_input2:
            height = st.number_input('Height (in)', key="height", step=STEP_SIZE[column], value = 1)

        user_inputs[column] = (weight / (height ** 2)) * 703
        columnName = metric

    else:
        columnName = metric + f" ({UNITS_MAP[column]})"

        with col_input:
            if column == 'TotHDLRat':
                user_inputs[column] = st.number_input(f'{metric}', key='user_input', step=STEP_SIZE[column])
            else:
                user_inputs[column] = st.number_input(f'{metric} ({UNITS_MAP[column]})', key='user_input', step=STEP_SIZE[column])

        with col_input2:
            st.caption(f"{validation_labels[column]}")

    if len(st.session_state.metric_list) == 0:
        st.session_state.metric_list.appendleft({'column':column, 'input': user_inputs[column], 'columnName': columnName})
    else:
        exists = False
        delete = -1
        for index, selection in enumerate(st.session_state.metric_list):
            if selection['column'] == column:
                exists = True
                delete = index
                break
        if exists:
            del st.session_state.metric_list[delete]
        if user_inputs[column] > validation_range[column][0] and user_inputs[column] < validation_range[column][1]:
            st.session_state.metric_list.appendleft({'column':column, 'input': user_inputs[column], 'columnName': columnName})

    # Number of elements
    num_elements = 10

    # Create placeholders for columns
    cols = [st.empty() for _ in range(num_elements)]

    for i, column_dict in enumerate(st.session_state.metric_list):

        global header
        global placeholder
        global header_color

        column = column_dict['column']
        columnName = column_dict['columnName']
        #st.write(column)

        with cols[i].container():
            
            placeholder = st.empty()

            header = f"{NAME_MAP[column]}"
            placeholder.markdown(f"#### {header}", unsafe_allow_html=True)

            col7, col8, col9, col10 = st.columns([0.025, 0.3, 0.65, 0.025], vertical_alignment='center')
            
            with col9:
                #user_input = user_inputs[column]  # Reference the user input from the dictionary
                user_input = column_dict['input']


                array = df[column].dropna()
                if len(array.index) < 5:
                    st.markdown(f"Not enough data available for {columnName}.")
                    continue

                if user_input <= validation_range[column][0] or user_input >= validation_range[column][1]:
                    # st.write(f"### ")
                    st.write(f"Please enter a :red[**valid**] value for {columnName}.")
                    continue
                elif STEP_SIZE[column] == 0.1 or column == 'BMXBMI':
                    header = f"{NAME_MAP[column]}: {user_input:.1f} {UNITS_MAP[column]}"
                else:
                    header = f"{NAME_MAP[column]}: {user_input:.0f} {UNITS_MAP[column]}"

                # st.write(f"####")

                if columnName in AHA_RANGES:
                    low_number = AHA_RANGES[columnName][1]
                    high_number = AHA_RANGES[columnName][3]
                    if ethnicity == 'South Asian' and columnName == 'Body Mass Index':
                        high_number = 23
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
                    extra_high_number = AHA_RANGES[columnName][5]
                    if extra_high_number is not None:
                        extra_high_percentile = np.mean(sorted_array <= extra_high_number) * 100

                low_percentile = np.mean(sorted_array <= low_number) * 100

                fig, ax = plt.subplots(figsize=(16, 1.15))

                # dot at user input position
                user_percentile = np.mean(sorted_array <= user_input) * 100
                if int(user_percentile) == 100:
                    user_percentile = 99
                
                if high_percentile < 99:
                    ax.plot([high_percentile, high_percentile], [0.6, 0.95], color='white', lw=1, label='Demarcation', zorder=999)
                ax.plot([low_percentile, low_percentile], [0.6, 0.95], color='white', lw=1, label='Demarcation', zorder=999)
                if columnName == 'Body Mass Index':
                    ax.plot([extra_high_percentile, extra_high_percentile], [0.6, 0.95], color='white', lw=1, label='Demarcation', zorder=999)
                
                if colorblind_mode == True:
                    optimal = lightgreen
                    borderline = darkgreen
                    at_risk = darkgreen
                else:
                    optimal = green
                    borderline = orange
                    at_risk = red

                # if columnName != 'HDL (mg/dL)':
                #     ax.add_patch(Rectangle((0, 0.6), 
                #                     low_percentile, 0.35, 
                #                     color=optimal, fill=True, zorder=90))
                #     ax.add_patch(Rectangle((low_percentile, 0.6), 
                #                 (high_percentile - low_percentile), 0.35, 
                #                 color=at_risk, fill=True, zorder=90))
                # else:
                #     ax.add_patch(Rectangle((0, 0.6), 
                #                     low_percentile, 0.35, 
                #                     color=at_risk, fill=True, zorder=90))
                #     ax.add_patch(Rectangle((low_percentile, 0.6), 
                #                 (high_percentile - low_percentile), 0.35, 
                #                 color=optimal, fill=True, zorder=90))

                # if high_percentile < 99:
                #     ax.add_patch(Rectangle((high_percentile, 0.6), 
                #                     (100 - high_percentile), 0.35, 
                #                     color=at_risk, fill=True, zorder=90))

                if columnName == 'HDL (mg/dL)':
                    ax.add_patch(Rectangle((0, 0.6), 
                                    low_percentile, 0.35, 
                                    color=at_risk, fill=True, zorder=90))
                    ax.add_patch(Rectangle((low_percentile, 0.6), 
                                (high_percentile - low_percentile), 0.35, 
                                color=optimal, fill=True, zorder=90))
                elif "Blood Pressure" in columnName:
                    ax.add_patch(Rectangle((0, 0.6), 
                                    low_percentile, 0.35, 
                                    color=optimal, fill=True, zorder=90))
                    ax.add_patch(Rectangle((low_percentile, 0.6), 
                                (high_percentile - low_percentile), 0.35, 
                                color=at_risk, fill=True, zorder=90))
                elif columnName == 'Body Mass Index':
                    ax.add_patch(Rectangle((0, 0.6), 
                                    low_percentile, 0.35, 
                                    color=at_risk, fill=True, zorder=90))
                    ax.add_patch(Rectangle((low_percentile, 0.6), 
                                (high_percentile - low_percentile), 0.35, 
                                color=optimal, fill=True, zorder=90))
                    ax.add_patch(Rectangle((high_percentile, 0.6), 
                                (extra_high_percentile - high_percentile), 0.35, 
                                color=borderline, fill=True, zorder=90))
                    ax.add_patch(Rectangle((extra_high_percentile, 0.6), 
                                (100 - extra_high_percentile), 0.35, 
                                color=at_risk, fill=True, zorder=90))
                else:
                    ax.add_patch(Rectangle((0, 0.6), 
                                    low_percentile, 0.35, 
                                    color=optimal, fill=True, zorder=90))
                    ax.add_patch(Rectangle((low_percentile, 0.6), 
                                (high_percentile - low_percentile), 0.35, 
                                color=borderline, fill=True, zorder=90))
                    ax.add_patch(Rectangle((high_percentile, 0.6), 
                                (100 - high_percentile), 0.35, 
                                color=at_risk, fill=True, zorder=90))

                if columnName == 'HDL (mg/dL)':
                    if user_percentile < low_percentile:
                        ax.scatter(user_percentile, 0.865, color=at_risk, zorder=1000, label='Your Input', s=950, edgecolors=['black'])
                        header_color = at_risk
                    else:
                        ax.scatter(user_percentile, 0.865, color=optimal, zorder=1000, label='Your Input', s=950, edgecolors=['black'])
                        header_color = optimal
                elif 'Blood Pressure' in columnName:
                    if user_percentile < low_percentile:
                        ax.scatter(user_percentile, 0.865, color=optimal, zorder=1000, label='Your Input', s=950, edgecolors=['black'])
                        header_color = optimal
                    else:
                        ax.scatter(user_percentile, 0.865, color=at_risk, zorder=1000, label='Your Input', s=950, edgecolors=['black'])
                        header_color = at_risk
                elif columnName == 'Body Mass Index':
                    if user_percentile < low_percentile:
                        ax.scatter(user_percentile, 0.865, color=at_risk, zorder=1000, label='Your Input', s=950, edgecolors=['black'])
                        header_color = at_risk
                    elif user_percentile < high_percentile:
                        ax.scatter(user_percentile, 0.865, color=optimal, zorder=1000, label='Your Input', s=950, edgecolors=['black'])
                        header_color = optimal
                    elif user_percentile < extra_high_percentile:
                        ax.scatter(user_percentile, 0.865, color=borderline, zorder=1000, label='Your Input', s=950, edgecolors=['black'])
                        header_color = borderline
                    else:
                        ax.scatter(user_percentile, 0.865, color=at_risk, zorder=1000, label='Your Input', s=950, edgecolors=['black'])
                        header_color = at_risk
                else:
                    if user_percentile < low_percentile:
                        ax.scatter(user_percentile, 0.865, color=optimal, zorder=1000, label='Your Input', s=950, edgecolors=['black'])
                        header_color = optimal
                    elif user_percentile <= high_percentile:
                        ax.scatter(user_percentile, 0.865, color=borderline, zorder=1000, label='Your Input', s=950, edgecolors=['black'])
                        header_color = borderline
                    elif user_percentile > high_percentile:
                        ax.scatter(user_percentile, 0.865, color=at_risk, zorder=1000, label='Your Input', s=950, edgecolors=['black'])
                        header_color = at_risk

                placeholder.markdown(f"#### <span style='color:{header_color};'>{header}</span>", unsafe_allow_html=True)

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

                if STEP_SIZE[column] == .1 or column == 'BMXBMI':
                    ax.annotate(f'{user_input: .1f}', xy=(user_percentile, 0.865), xytext=(user_percentile - 0.2, 0.81),
                                horizontalalignment='center', color = 'black', zorder=1001, weight='bold', fontsize=12)
                else:
                    ax.annotate(f'{user_input: .0f}', xy=(user_percentile, 0.865), xytext=(user_percentile - 0.2, 0.81),
                                horizontalalignment='center', color = 'black', zorder=1001, weight='bold', fontsize=14)
                
                if int(user_percentile) == 100:
                    user_percentile = 99
                if user_percentile > 90:
                    plt.annotate(f'({user_percentile:.0f}%ile)', xy=(user_percentile, 0.9), xytext=(user_percentile - 5, 0.98), 
                             horizontalalignment='center')
                else:
                    plt.annotate(f'({user_percentile:.0f}%ile)', xy=(user_percentile, 0.9), xytext=(user_percentile + 5, 0.98), 
                             horizontalalignment='center')
                    
                if columnName == 'Body Mass Index':
                    plt.annotate(f'{AHA_RANGES[columnName][0]}', xy=(low_percentile, 0.675), xytext=(low_percentile - 1, 0.45),
                                horizontalalignment='right', weight='bold')
                    plt.annotate(f'<18.5', xy=(low_percentile - 1, 0.657), xytext=(low_percentile - 1, 0.3),
                            horizontalalignment='right')
                    if ethnicity == 'South Asian':
                        plt.annotate(f'{AHA_RANGES[columnName][2]}', xy=(low_percentile, 0.675), xytext=(low_percentile, 0.45),
                                horizontalalignment='left', weight='bold')
                        plt.annotate(f'18.5-23', xy=(low_percentile, 0.657), xytext=(low_percentile, 0.3),
                                horizontalalignment='left')
                        plt.annotate(f'{AHA_RANGES[columnName][4]}', xy=(low_percentile, 0.675), xytext=(high_percentile, 0.45),
                                horizontalalignment='left', weight='bold')
                        plt.annotate(f'23-25', xy=(high_percentile - 1, 0.657), xytext=(high_percentile, 0.3),
                                horizontalalignment='left')
                        plt.annotate(f'{AHA_RANGES[columnName][6]}', xy=(low_percentile, 0.675), xytext=(extra_high_percentile, 0.45),
                                horizontalalignment='left', weight='bold')
                        plt.annotate('>25', xy=(extra_high_percentile - 1, 0.657), xytext=(extra_high_percentile, 0.3),
                                horizontalalignment='left')
                    else:
                        plt.annotate(f'{AHA_RANGES[columnName][2]}', xy=(low_percentile, 0.675), xytext=(low_percentile, 0.45),
                                horizontalalignment='left', weight='bold')
                        plt.annotate(f'18.5-25', xy=(low_percentile, 0.657), xytext=(low_percentile, 0.3),
                                horizontalalignment='left')
                        plt.annotate(f'{AHA_RANGES[columnName][4]}', xy=(low_percentile, 0.675), xytext=(high_percentile, 0.45),
                                horizontalalignment='left', weight='bold')
                        plt.annotate(f'25-30', xy=(high_percentile - 1, 0.657), xytext=(high_percentile, 0.3),
                                horizontalalignment='left')
                        plt.annotate(f'{AHA_RANGES[columnName][6]}', xy=(low_percentile, 0.675), xytext=(extra_high_percentile, 0.45),
                                horizontalalignment='left', weight='bold')
                        plt.annotate('>30', xy=(extra_high_percentile - 1, 0.657), xytext=(extra_high_percentile, 0.3),
                                horizontalalignment='left')
                else:
                    if high_number < 1000:
                        plt.annotate(f'{AHA_RANGES[columnName][4]}', xy=(high_percentile, 0.675), xytext=(high_percentile, 0.45),
                                    horizontalalignment='left', weight='bold')
                        plt.annotate(f'>{high_number}', xy=(high_percentile, 0.675), xytext=(high_percentile, 0.3),
                                    horizontalalignment='left')
                    if low_number > 0:
                        plt.annotate(f'{AHA_RANGES[columnName][0]}', xy=(low_percentile, 0.675), xytext=(low_percentile - 1, 0.45),
                                    horizontalalignment='right', weight='bold')
                        plt.annotate(f'<{low_number}', xy=(low_percentile - 1, 0.657), xytext=(low_percentile - 1, 0.3),
                                    horizontalalignment='right')
                    if high_number == 1000:
                        plt.annotate(f'{AHA_RANGES[columnName][2]}', xy=(low_percentile, 0.675), xytext=(low_percentile, 0.45),
                                        horizontalalignment='left', weight='bold')
                        plt.annotate(f'>{low_number}', xy=(low_percentile, 0.675), xytext=(low_percentile, 0.3),
                                        horizontalalignment='left')
                    else:
                        plt.annotate(f'{AHA_RANGES[columnName][2]}', xy=(low_percentile, 0.675), xytext=(low_percentile, 0.45),
                                        horizontalalignment='left', weight='bold')
                        plt.annotate(f'{low_number}-{high_number}', xy=(low_percentile, 0.675), xytext=(low_percentile, 0.3),
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
                if "HDL" in columnName or high_number == 1000:
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

                percentile_25 = np.percentile(sorted_array, 25)
                percentile_50 = np.percentile(sorted_array, 50)
                percentile_75 = np.percentile(sorted_array, 75)
                percentile_90 = np.percentile(sorted_array, 90)

                popup_column = NAME_MAP[column]

            with col8:
                st.markdown("""
                    <style>
                        button[kind="primary"] {
                            background-color: white;
                            color: black;
                            border: 1px solid #7D343C;
                        }

                        button[kind="primary"]:hover {
                            background-color: white; /* Slightly different background on hover */
                            color: #7D343C; /* Ensure text color remains visible on hover */
                        }
                    </style>
                    """, unsafe_allow_html=True)
                
                #user_input = user_inputs[]  # Reference the user input from the dictionary
                user_input = column_dict['input']

                array = df[column].dropna()
                sorted_array = np.sort(array)
                user_percentile = np.mean(sorted_array <= user_input) * 100

                more_info = st.button(label=f'ⓘ {user_percentile: .0f}{suffix} percentile compared to peers in your group', key=column, type='primary')
                # st.caption(' Press for more info')

            if more_info:
                    if "HDL (mg/dL)" in columnName:
                        popup(column, popup_column, user_input, user_percentile, gender, ethnicity, age_group, "cholesterol", medChol, prob, value, 
                              percentile_25, percentile_50, percentile_75, percentile_90, "greater", suffix, low_number, high_number)
                    elif "DL" in columnName or "Trig" in columnName or "Chol" in columnName:
                        popup(column, popup_column, user_input, user_percentile, gender, ethnicity, age_group, "cholesterol", medChol, prob, value, 
                              percentile_25, percentile_50, percentile_75, percentile_90, "less", suffix, low_number, high_number)
                    elif "Glucose" in columnName or "A1C" in columnName:
                        popup(column, popup_column, user_input, user_percentile, gender, ethnicity, age_group, "blood sugar", medDiab, prob, value, 
                              percentile_25, percentile_50, percentile_75, percentile_90, "less", suffix, low_number, high_number)   
                    else:
                        popup(column, popup_column, user_input, user_percentile, gender, ethnicity, age_group, "blood pressure", medBP, prob, value, 
                              percentile_25, percentile_50, percentile_75, percentile_90, "less", suffix, low_number, high_number)
            

# Main execution
df_c = load_files(False)
df_d = ui_choose(df_c, False)
show_analysis(df_d)

st.divider()
st.markdown('<div style="text-align: center"> Please email <a href="mailto:sanaa.bhorkar@gmail.com">sanaa.bhorkar@gmail.com</a> with any feedback! </div>', unsafe_allow_html=True)
st.markdown(f"<div style='text-align: center'> Version {VERSION}</div>", unsafe_allow_html=True)
