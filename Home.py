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

st.set_page_config(page_title="SCORE Comparison Tool", page_icon=":anatomical_heart:", layout="wide")

deploy = True
# Hide the hamburger menu for deployment
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
if deploy:
    st.markdown(hide_menu_style, unsafe_allow_html=True)

# st.caption('####')
# @st.cache_resource(ttl=600)
# def get_app_queue():
#     return deque()

DIR = os.getcwd()
PLOT_DIR = DIR + '/plots'
LOGO_DIR = DIR + '/logo/'
DATA_DIR = DIR + '/data/'
OUTPUT_DIR = DIR + '/output/'
SAHC_DATA_DIR = DIR + '/sahc_data/'
VERSION = 1.7

image_path = os.path.join(LOGO_DIR, 'SCORE.svg')

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
# header.write("""<div class='fixed-header'/>""", unsafe_allow_html=True)

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
    'LBXGLU': 'Fasting Glucose', 'LBXGH': 'HbA1c',
    'BMXBMI': 'Body Mass Index', 'BPXOSY1': 'Systolic Blood Pressure', 'BPXODI1': 'Diastolic Blood Pressure', 
}

DROPDOWN_SELECTION = {
    'Total Cholesterol': 'LBXTC', 'LDL': 'LBDLDL', 'HDL': 'LBDHDD',
    'Triglycerides': 'LBXTR', 'Total Cholesterol to HDL Ratio': 'TotHDLRat',
    'Fasting Glucose': 'LBXGLU', 'HbA1c': 'LBXGH',
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
    'HbA1c (%)': ("Optimal", 5.7, "Borderline", 6.4, "At risk", None, None),
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

# aboutMe_placeholder = st.empty()
# aboutMe_placeholder.markdown(f"### <u>About Me</u>", unsafe_allow_html=True)

expand_label = "About Me"

# aboutMe_expand = st.expander(expand_label, expanded=True)

if 'title_list' not in st.session_state:
    st.session_state.title_list = []

if "title_expander" not in st.session_state:
    st.session_state.title_expander = "About Me"

if "selected_gender" not in st.session_state:
    st.session_state.selected_gender = None

if "selected_age" not in st.session_state:
    st.session_state.selected_age = None

if "selected_ethnicity" not in st.session_state:
    st.session_state.selected_ethnicity = None

if "selected_meds" not in st.session_state:
    st.session_state.selected_meds = []

genderOptions = {'Male': 1, 'Female': 2}
ageOptions = {'19-33': 19, '34-48': 34, '49-64': 49, '65-78': 65,'79+': 79}
dataSelection = ['NHANES', 'SAHC']

def update_title():
    st.session_state.title_list = []  # Clear the title list on each update
    
    selected_gender = st.session_state.selected_gender
    selected_age = st.session_state.selected_age
    selected_ethnicity = st.session_state.selected_ethnicity
    selected_meds = st.session_state.selected_meds
    
    if selected_gender:
        st.session_state.title_list.append(selected_gender)
    if selected_age:
        st.session_state.title_list.append(f"Age: {selected_age}")
    if selected_ethnicity:
        st.session_state.title_list.append(selected_ethnicity)
    if selected_meds:
        if 'None' in selected_meds:
            selected_meds.remove('None')
        if selected_meds != []:
            selected_meds = ', '.join(selected_meds)
            st.session_state.title_list.append(f"Current medications: {selected_meds}")
    
    if st.session_state.title_list:
        st.session_state.title_expander = f"About Me: {', '.join(st.session_state.title_list)}"
    else:
        st.session_state.title_expander = "About Me"

with st.expander(st.session_state.title_expander or expand_label, expanded=True):
    # option = st.selectbox("Select an option", ["One", "Two", "Tres"], key="title_expander")
    col1, col12, col2, col22, col3, col32, col4, col42 = st.columns([.2, .1, .2, .1, .2, .1, .2, .1], vertical_alignment='top')

    medCholOptions = {'Yes': 1, 'No': 2}
    medDiabOptions = {'Yes': 1, 'No': 2}
    medBPOptions = {'Yes': 1, 'No': 2}

    with col1:
        # gender = st.selectbox('Gender', list(genderOptions.keys()), key=None) # fix these SANAA
        # gender = st.selectbox('Gender', list(genderOptions.keys()), key="selected_gender", on_change=update_title(), index=None)
        gender = st.selectbox('Gender', list(genderOptions.keys()), key="selected_gender", on_change=update_title, index=None)
        # index = None will keep default empty
    with col2:
        # age_group = st.selectbox('Age group', list(ageOptions.keys()), index=2, key=None)
        # age_group = st.selectbox('Age group', list(ageOptions.keys()), key="selected_age", on_change=update_title, index=None)
        age_group = st.selectbox('Age group', list(ageOptions.keys()), key="selected_age", on_change=update_title, index=None)
    with col3:
        # ethnicity = st.selectbox('Ethnicity', ['Non-South Asian', 'South Asian'], key=None)
        ethnicity = st.selectbox('Ethnicity', ['Non-South Asian', 'South Asian'], key="selected_ethnicity", on_change=update_title, index=None)
    with col4:
        med_options = ['None', 'Cholesterol', 'Diabetes', 'Blood Pressure']
        medications_select = st.multiselect(label="Select your medication use", options=med_options, key="selected_meds", on_change=update_title)

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

# aboutMe_label = f"About me: {gender}, Age: {age_group}, {ethnicity}, Current medications: {meds}"
# aboutMe_placeholder.markdown(f"### <u>{aboutMe_label}</u>", unsafe_allow_html=True)
                                            
# @st.cache_data
def load_files(debugging):
    if ethnicity is None or ethnicity != 'South Asian':
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

        # st.write(df_combined.shape)
        # st.dataframe(df_combined)

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

    df2 = df.copy()
    # st.write(df2)
    # st.write(df2.shape)

    if gender is not None:
        genderFilter = [genderOptions[gender]]
        df2 = df2[df2['RIAGENDR'].isin(genderFilter)]
        # st.write(df2.shape, "gender")
    if age_group is not None:
        ageFilter = [ageOptions[age_group]]
        df2 = df2[df2['Age_Group'].isin(ageFilter)]
        # st.write(df2.shape, "Age")

    # st.write(df2)
    
    if ethnicity is not None and ethnicity == 'South Asian':
        medCholFilter = [medCholOptions[medChol]]
        df2 = df2[df2['cholMeds'].isin(medCholFilter)]
        medDiabFilter = [medDiabOptions[medDiab]]
        df2 = df2[df2['diabMeds'].isin(medDiabFilter)]
        medBPFilter = [medBPOptions[medBP]]
        df2 = df2[df2['bpMeds'].isin(medBPFilter)]
    elif ethnicity is None or ethnicity != 'South Asian':
        # st.dataframe(df2)
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
        if age_group is not None:
            next_age_group = get_next_age_group(age_group)
            ageFilter.append(ageOptions[next_age_group])

            if gender is not None:
                genderFilter = [genderOptions[gender]]
                df2 = df2[df2['RIAGENDR'].isin(genderFilter)]
            
            if ethnicity is not None and ethnicity == 'South Asian':
                medCholFilter = [medCholOptions[medChol]]
                df2 = df2[df2['cholMeds'].isin(medCholFilter)]
                medDiabFilter = [medDiabOptions[medDiab]]
                df2 = df2[df2['diabMeds'].isin(medDiabFilter)]
                medBPFilter = [medBPOptions[medBP]]
                df2 = df2[df2['bpMeds'].isin(medBPFilter)]
            elif ethnicity is None or ethnicity != 'South Asian':
                if medChol == 'Yes':
                    df2 = df2[df2['BPQ100D'].isin(medCholFilter)]
                elif medChol == 'No':
                    df2 = df2[df2['BPQ090D'].isin(medCholFilter) | df2['BPQ100D'].isin(medCholFilter)]
                df2 = df2[df2['DIQ070'].isin(medDiabFilter)]
                if medBP == 'Yes':
                    df2 = df2[df2['BPQ040A'].isin(medBPFilter)]
                elif medBP == 'No':
                    df2 = df2[df2['BPQ020'].isin(medBPFilter) | df2['BPQ040A'].isin(medBPFilter)]
    
    if len(df2) < 15:
        st.write(f"Not enough records found to compare. Please remove medication usage and try again.")
    else:
        length = len(df2)
        st.markdown(f"<span style='color:white;'>{length} records found.</span>", unsafe_allow_html=True) # SANAA unwhite
        

    return df2

@st.dialog(" ")
def popup(acro, column, user_input, gender, race, age_range, med, on_med, prob, p25, p50, p75, p90, low_number, high_number, status, df3):
    
    st.header(f"Comparing your {column} value to others in your peer group")
    if on_med == 'Yes':
        if age_range is None and race is None and gender is None:
            st.write(f"(Person ON {med}-lowering medication.)")  
        elif gender is None and race is None:
            st.write(f"(Person aged between {age_range} years, ON {med}-lowering medication.)")
        elif age_range is None and gender is None:
            st.write(f"({race}, ON {med}-lowering medication.)")  
        elif age_range is None and race is None:
            st.write(f"({gender}, ON {med}-lowering medication.)")  
        elif gender is None:
            st.write(f"({race}, aged between {age_range} years, ON {med}-lowering medication.)")
        elif race is None:
            st.write(f"({gender}, aged between {age_range} years, ON {med}-lowering medication.)")
        else:
            st.write(f"({race} {gender.lower()}, aged between {age_range} years, ON {med}-lowering medication.)")
    else:
        if age_range is None and race is None and gender is None:
            st.write(f"(Person NOT ON {med}-lowering medication.)")  
        elif gender is None and race is None:
            st.write(f"(Person aged between {age_range} years, NOT ON {med}-lowering medication.)")
        elif age_range is None and gender is None:
            st.write(f"({race}, NOT ON {med}-lowering medication.)")  
        elif age_range is None and race is None:
            st.write(f"({gender}, NOT ON {med}-lowering medication.)")  
        elif gender is None:
            st.write(f"({race}, aged between {age_range} years, NOT ON {med}-lowering medication.)")
        elif race is None:
            st.write(f"({gender}, aged between {age_range} years, NOT ON {med}-lowering medication.)")
        else:
            st.write(f"({race} {gender.lower()}, aged between {age_range} years, NOT ON {med}-lowering medication.)")

    total = df3.shape[0]
    # st.write(total)

    # columnName = column + f" ({UNITS_MAP[acro]})"
    df3 = df3[df3[acro] <= 5.8]
    top = df3.shape[0]  
    # st.write(top)           
    prop = (top / total) * 100

    st.header(f"Your {column}: {user_input:.1f} {UNITS_MAP[acro]}, Risk classification: {status}")
    st.write(f"{prop:.0f}% of individuals in your peer group have {column} <= {user_input:.1f}")

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

    if 'to HDL' in column or 'Body Mass' in column:
        st.write(f"The **optimal** value for {column} is **{low_number} - {high_number}** according to AHA guidelines. In your peer group, the estimated probability of having a sub-optimal {column} of ≤ {low_number} or ≥ {high_number} is {prob:.0f}%.")
    elif 'HDL' in column:
        st.write(f"The **optimal** value for {column} is ≥ **{low_number}** according to AHA guidelines. In your peer group, the estimated probability of having a sub-optimal {column} of ≥ {low_number} is {prob:.0f}%.")
    else:
        st.write(f"The **optimal** value for {column} is ≤ **{low_number}** according to AHA guidelines. In your peer group, the estimated probability of having a sub-optimal {column} of ≥ {low_number} is {prob:.0f}%.")

    if status != 'Optimal':
        st.write("Please check our guidance on next steps to lower the risk posed by this marker at the end of your report.")
    else:
        "Your marker is in optimal range. Continue working on your lifestyle behaviors to keep this marker in range."

# metric_list = get_app_queue()
if 'metric_list' not in st.session_state:
    st.session_state.metric_list = deque()


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

    # st.markdown(f"### <u>My risk profile markers</u>", unsafe_allow_html=True)

    # col_empty1, col_dropdown, col_empty, col_input, col_input2, col_empty = st.columns([0.001, 0.234, 0.115, 0.2, 0.2, 0.25], gap='medium', vertical_alignment='center')
    col_dropdown, col_empty, col_input, col_empty2, col_input2, col_empty3 = st.columns([0.225, 0.025, 0.172, 0.067, 0.175, 0.331], gap='medium', vertical_alignment='center')


    with col_dropdown:
        metric = st.selectbox('My Risk Profile Markers: Select one', list(DROPDOWN_SELECTION.keys()))
    
    column = DROPDOWN_SELECTION[metric]

    if column == 'BMXBMI':
        columnName = metric

        with col_input:
            weight = st.number_input('Weight (lbs)', key="weight", step=STEP_SIZE[column], value=None)
        with col_input2:
            height = st.number_input('Height (in)', key="height", step=STEP_SIZE[column], value=None)

        if weight is not None and height is not None:
            user_inputs[column] = (weight / (height ** 2)) * 703
        else:
            user_inputs[column] = -1

    else:
        columnName = metric + f" ({UNITS_MAP[column]})"

        with col_input:
            if column == 'TotHDLRat':
                user_inputs[column] = st.number_input(f'{metric}', key='user_input', step=STEP_SIZE[column], value=None)
            else:
                user_inputs[column] = st.number_input(f'{metric} ({UNITS_MAP[column]})', key='user_input', step=STEP_SIZE[column], value=None)

        # with col_input2:
        #     st.caption(f"{validation_labels[column]}")

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
        if user_inputs[column] is not None and user_inputs[column] >= 0:
            st.session_state.metric_list.appendleft({'column':column, 'input': user_inputs[column], 'columnName': columnName})

    st.divider()
    
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

            col8, col9, col10 = st.columns([0.65, 0.3, 0.05], vertical_alignment='top', gap='small')
        
            with col8:
                #user_input = user_inputs[column]  # Reference the user input from the dictionary
                user_input = column_dict['input']

                if user_input == None:
                    break

                placeholder.markdown(f"#### {header}", unsafe_allow_html=True)

                array = df[column].dropna()
                if len(array.index) < 5:
                    st.markdown(f"Not enough data available for {columnName}.")
                    continue

                # if user_input <= validation_range[column][0] or user_input >= validation_range[column][1]:
                #     # st.write(f"### ")
                #     st.write(f"Please enter a :red[**valid**] value for {columnName}.")
                #     continue
                # el
                if STEP_SIZE[column] == 0.1 or column == 'BMXBMI':
                    header = f"{NAME_MAP[column]}: {user_input:.1f} {UNITS_MAP[column]}"
                else:
                    header = f"{NAME_MAP[column]}: {user_input:.0f} {UNITS_MAP[column]}"

                # st.write(f"####")

                if columnName in AHA_RANGES:
                    low_number = AHA_RANGES[columnName][1]
                    high_number = AHA_RANGES[columnName][3]
                    if ethnicity == 'South Asian' and columnName == 'Body Mass Index':
                        AHA_RANGES[columnName][3] = 23
                else:
                    st.write(f"No AHA prescribed range available for {columnName}.")
                    continue
                if columnName == 'HDL (mg/dL)' and gender == 'Female':
                    AHA_RANGES[columnName][1] = 50

                sorted_array = np.sort(array)

                # Calculate the percentile
                if high_number == None:
                    high_number = 1000
                    high_percentile = 99
                else:
                    high_percentile = int(np.mean(sorted_array <= high_number) * 100)
                    extra_high_number = AHA_RANGES[columnName][5]
                    if extra_high_number is not None:
                        extra_high_percentile = int(np.mean(sorted_array <= extra_high_number) * 100)

                low_percentile = int(np.mean(sorted_array <= low_number) * 100)

                fig, ax = plt.subplots(figsize=(16, 1.15))

                # dot at user input position
                user_percentile = int(np.mean(sorted_array <= user_input) * 100)
                if user_percentile == 100:
                    user_percentile = 99

                status = 'Optimal'
                
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

                scatter_size = 1750

                if columnName == 'HDL (mg/dL)':
                    if user_percentile < low_percentile:
                        header_color = at_risk
                        status = 'At risk'
                    else:
                        header_color = optimal
                elif 'Blood Pressure' in columnName:
                    if user_percentile < low_percentile:
                        header_color = optimal
                    else:
                        header_color = at_risk
                        status = 'At risk'
                elif columnName == 'Body Mass Index':
                    if user_percentile < low_percentile:
                        header_color = at_risk
                        status = 'At risk'
                    elif user_percentile < high_percentile:
                        header_color = optimal
                    elif user_percentile < extra_high_percentile:
                        header_color = borderline
                        status = 'Borderline'
                    else:
                        header_color = at_risk
                        status = 'At risk'
                else:
                    if user_percentile < low_percentile:
                        header_color = optimal
                    elif user_percentile <= high_percentile:
                        header_color = borderline
                        status = 'Borderline'
                    elif user_percentile > high_percentile:
                        header_color = at_risk
                        status = 'At risk'
                
                ax.scatter(user_percentile, 0.855, zorder=999, s=scatter_size+100, edgecolors='k')
                ax.scatter(user_percentile, 0.855, color=header_color, zorder=1000, label='Your Input', s=scatter_size, edgecolors='w')
                        

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

                # ax.set_title(columnName)

                if STEP_SIZE[column] == .1 or column == 'BMXBMI':
                    ax.annotate(f'{user_input: .1f}', xy=(user_percentile, 0.865), xytext=(user_percentile - 0.25, 0.79),
                                horizontalalignment='center', color = 'black', zorder=1001, weight='bold', fontsize=16)
                else:
                    ax.annotate(f'{user_input: .0f}', xy=(user_percentile, 0.865), xytext=(user_percentile - 0.25, 0.79),
                                horizontalalignment='center', color = 'black', zorder=1001, weight='bold', fontsize=16)
                
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

                df3 = df.copy()

                total = df3.shape[0]

                if "to HDL" in columnName:
                    value = AHA_RANGES[columnName][1]
                    df3 = df3[df3['TotHDLRat'] < AHA_RANGES[columnName][1] & df3['TotHDLRat'] > AHA_RANGES[columnName][3]]
                elif "HDL" in columnName:
                    value = AHA_RANGES[columnName][1]
                    df3 = df3[df3['LBDHDD'] < AHA_RANGES[columnName][1]]
                elif "Body Mass Index" in columnName:
                    value = AHA_RANGES[columnName][3]
                    df3 = df3[df3['BMXBMI'] > AHA_RANGES[columnName][3]]
                else:
                    value = AHA_RANGES[columnName][1]
                    df3 = df3[df3[column] > AHA_RANGES[columnName][1]]

                top = df3.shape[0]
                # st.write(top)
                # st.write(total)                
                prob = (top / total) * 100

                # kde = sm.nonparametric.KDEUnivariate(array)
                # kde.fit(bw='scott')  # You can also specify bandwidth manually, e.g., bw=0.5

                # # Generate a range of HDL values to estimate the CDF
                # hdl_range = np.linspace(min(array), max(array), 1000)

                # # Evaluate the KDE over the range
                # pdf_values = kde.evaluate(hdl_range)

                # # Calculate the CDF by integrating the PDF
                # cdf_values = np.cumsum(pdf_values) * (hdl_range[1] - hdl_range[0])

                # # Find the CDF value at HDL = 35
                # if "HDL" in columnName or high_number == 1000:
                #     value = low_number
                # else:
                #     value = high_number
                
                # cdf_at_35 = np.interp(value, hdl_range, cdf_values)

                # # Probability of HDL >= 35
                # if columnName == 'HDL (mg/dL)':
                #     prob = 1 - cdf_at_35
                # else:
                #     prob = cdf_at_35

                # # Display the result in Streamlit
                # st.write(f"Probability of HDL >= {value}: {probability_gte_35 * 100: .0f}%")
                # st.write(f"Probability of HDL = {value}: {prob_density_value[0]}")

                percentile_25 = np.percentile(sorted_array, 25)
                percentile_50 = np.percentile(sorted_array, 50)
                percentile_75 = np.percentile(sorted_array, 75)
                percentile_90 = np.percentile(sorted_array, 90)

                popup_column = NAME_MAP[column]

            with col9:
                st.markdown("""
                    <style>
                        button[kind="primary"] {
                            background-color: white;
                            color: black;
                            border: 0px solid #7D343C;
                            padding: 5px;
                        }

                        button[kind="primary"]:hover {
                            background-color: #7D343C; /* Slightly different background on hover */
                            color: white; /* Ensure text color remains visible on hover */
                        }
                    </style>
                    """, unsafe_allow_html=True)
                
                #user_input = user_inputs[]  # Reference the user input from the dictionary
                user_input = column_dict['input']

                array = df[column].dropna()
                sorted_array = np.sort(array)
                user_percentile = int(np.mean(sorted_array <= user_input) * 100)

                digit = user_percentile % 10
                # st.write(digit)
                suffix = 'th'

                if digit == 1:
                    suffix = 'st'
                elif digit == 2:
                    suffix == 'nd'
                elif digit == 3:
                    suffix == 'rd'

                more_info = st.button(label=f'ⓘ {user_percentile: .0f}{suffix} percentile compared to peers in your group', key=column, type='primary')
                # st.caption(' Press for more info')

            if more_info:
                    if "HDL (mg/dL)" in columnName or "DL" in columnName or "Trig" in columnName or "Chol" in columnName:
                        popup(column, popup_column, user_input, gender, ethnicity, age_group, "cholesterol", medChol, prob, 
                              percentile_25, percentile_50, percentile_75, percentile_90, low_number, high_number, status, df)
                    elif "Glucose" in columnName or "A1C" in columnName:
                        popup(column, popup_column, user_input, gender, ethnicity, age_group, "blood sugar", medDiab, prob, 
                              percentile_25, percentile_50, percentile_75, percentile_90, low_number, high_number, status, df)   
                    else:
                        popup(column, popup_column, user_input, gender, ethnicity, age_group, "blood pressure", medBP, prob, 
                              percentile_25, percentile_50, percentile_75, percentile_90, low_number, high_number, status, df3)
            


# Main execution
df_c = load_files(False)
df_d = ui_choose(df_c, False)
show_analysis(df_d)

# df_d = df_d['LBDHDD'] check with Amit
# st.write(df_d.shape[0])

# df_d = df_d[df_d < 40]
# st.write(df_d.shape[0])

st.divider()
st.markdown('<div style="text-align: center"> Please email <a href="mailto:sanaa.bhorkar@gmail.com">sanaa.bhorkar@gmail.com</a> with any feedback! </div>', unsafe_allow_html=True)
st.markdown(f"<div style='text-align: center'> Version {VERSION}</div>", unsafe_allow_html=True)