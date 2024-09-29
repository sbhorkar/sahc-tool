
# Import all necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import deque
import sqlite3
from streamlit_extras.stylable_container import stylable_container

########################### PAGE SET UP ##############################
# Setting page initial state
st.set_page_config(page_title="SCORE Comparison Tool", page_icon=":anatomical_heart:", layout="wide", initial_sidebar_state = "auto")

# Set up the page margins to reduce top padding
st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 2rem;
                    padding-right: 1rem;
                }
        </style>
        """, unsafe_allow_html=True)

# Hide the hamburger menu
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

# Adding CSS for the share button logo
st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.6.0/css/all.min.css"/>', unsafe_allow_html=True)

DIR = os.getcwd()
LOGO_DIR = DIR + '/logo/'
DATA_DIR = DIR + '/data/'
SAHC_DATA_DIR = DIR + '/sahc_data/'
VERSION = 3.8 #Fixed issue of column

image_path = os.path.join(LOGO_DIR, 'SCORE Official Logo.svg')

########################### FEEDBACK ##############################
# Initialize the database connection
conn = sqlite3.connect('feedback.db')
c = conn.cursor()

# Create table if it doesn't exist
c.execute('''CREATE TABLE IF NOT EXISTS feedback
            (thumbs_up INTEGER DEFAULT 0, thumbs_down INTEGER DEFAULT 0)''')

# Initialize counts if the table is empty
c.execute("SELECT COUNT(*) FROM feedback")
if c.fetchone()[0] == 0:
    c.execute("INSERT INTO feedback (thumbs_up, thumbs_down) VALUES (0, 0)")
    conn.commit()

# Function to get counts
def get_counts():
    c.execute("SELECT thumbs_up, thumbs_down FROM feedback")
    return c.fetchone()

# Function to update count
def update_count(feedback_type):
    if feedback_type == "thumbs_up":
        c.execute("UPDATE feedback SET thumbs_up = thumbs_up + 1")
    elif feedback_type == "thumbs_down":
        c.execute("UPDATE feedback SET thumbs_down = thumbs_down + 1")
    conn.commit()

@st.dialog(" ")
def header_popup(liked):
    if liked:
        st.write("Glad you liked SCORE! Would you like to send suggestions for improvement?")
        url = 'mailto:sahc@elcaminohealth.org?Subject=Liked%20SCORE%2C%20some%20suggestions%20for%20improvement&Body=Hello%2C%0A%0AHere%20are%20my%20suggestions%20to%20help%20improve%20SCORE%3A%0A%0ABest%2C%0ANAME%0AMOBILE%20%28provide%20if%20you%20would%20like%20the%20South%20Asian%20Heart%20Center%20to%20contact%20you%20regarding%20your%20suggestions%29.%0A%0A'
    else:
        st.write("Would you like to send suggestions for improvement?")
        url = 'mailto:sahc@elcaminohealth.org?Subject=Feedback%20on%20SCORE&Body=Hello%2C%0A%0AHere%20are%20my%20suggestions%20to%20help%20improve%20SCORE%3A%0A%0ABest%2C%0ANAME%0AMOBILE%20%28provide%20if%20you%20would%20like%20the%20South%20Asian%20Heart%20Center%20to%20contact%20you%20for%20further%20details%29.%0A%0A'
    col_button1, col_button2, col_empty = st.columns([0.2, 0.2, 0.6])
    with col_button1:
        st.link_button("Yes", url)
    with col_button2:
        if st.button("No"):
            st.rerun()

########################### FEEDBACK END ##############################

########################### VIEWER COUNT ##############################

def create_db():
    conn = sqlite3.connect('unique_views.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS unique_views (
            user_id TEXT PRIMARY KEY
        )
    ''')
    conn.commit()
    conn.close()

create_db()

import streamlit as st
import hashlib

import uuid

# Generate a unique run_id for the session
if 'run_id' not in st.session_state:
    st.session_state.run_id = str(uuid.uuid4())

def get_user_id():
    session_id = st.session_state.get("user_id")
    if not session_id:
        session_id = hashlib.sha256(str(st.session_state.run_id).encode()).hexdigest()
        st.session_state["user_id"] = session_id
    return session_id

def track_unique_view(user_id):
    conn = sqlite3.connect('unique_views.db')
    cursor = conn.cursor()

    cursor.execute('INSERT OR IGNORE INTO unique_views (user_id) VALUES (?)', (user_id,))
    conn.commit()

    cursor.execute('SELECT COUNT(*) FROM unique_views')
    total_views = cursor.fetchone()[0]

    conn.close()
    return total_views

user_id = get_user_id()
total_unique_views = track_unique_view(user_id)

########################### VIEWER COUNT END ##############################

########################### SIDEBAR ##############################
@st.dialog("Share with your friends and family!")
def share_popup():
    col_mail, col_what, col_mess = st.columns(3, gap='small')
    with col_mail:
        url = 'mailto:?Subject=Checkout%20SCORE%3A%20Compare%20your%20lipid%20and%20glucose%20markers%20with%20others%20similar%20to%20you&Body=Hello%2C%0A%0AI%20recently%20came%20across%20SCORE%2C%20a%20tool%20from%20El%20Camino%20Health%2C%20South%20Asian%20Heart%20Center%20that%20compares%20your%20lipids%20and%20other%20cardio-metabolic%20markers%20against%20your%20peers%2C%20matching%20your%20age%2C%20gender%2C%20ethnicity%2C%20and%20medication%20use.%0A%0AThis%20may%20help%20you%20calibrate%20your%20markers%20and%20take%20steps%20to%20improve%20your%20risk%20profile.%0A%0ACheck%20it%20out%20it%20out%20here%3A%20https%3A//scores.streamlit.app/%0A%0AYou%20may%20read%20more%20about%20the%20work%20of%20El%20Camino%20Health%27s%20South%20Asian%20Heart%20Center%2C%20a%20non-profit%20with%20the%20mission%20to%20reduce%20the%20high%20incidence%20of%20diabetes%20and%20heart%20disease%20with%20evidence-based%2C%20culturally%20tailored%2C%20and%20lifestyle-focused%20prevention%20services%2C%20here%3A%20www.southasianheartcenter.org%0A%0ABest%2C%0A'
        st.link_button(":envelope: Mail", url)
    with col_what:
        url = 'https://wa.me/?text=Hello%2C%0A%0AI%20recently%20came%20across%20SCORE%2C%20a%20tool%20from%20El%20Camino%20Health%2C%20South%20Asian%20Heart%20Center%20that%20compares%20your%20lipids%20and%20other%20cardio-metabolic%20markers%20against%20your%20peers%2C%20matching%20your%20age%2C%20gender%2C%20ethnicity%2C%20and%20medication%20use.%0A%0AThis%20may%20help%20you%20calibrate%20your%20markers%20and%20take%20steps%20to%20improve%20your%20risk%20profile.%0A%0ACheck%20it%20out%20it%20out%20here%3A%20https%3A//scores.streamlit.app/%0A%0AYou%20may%20read%20more%20about%20the%20work%20of%20El%20Camino%20Health%27s%20South%20Asian%20Heart%20Center%2C%20a%20non-profit%20with%20the%20mission%20to%20reduce%20the%20high%20incidence%20of%20diabetes%20and%20heart%20disease%20with%20evidence-based%2C%20culturally%20tailored%2C%20and%20lifestyle-focused%20prevention%20services%2C%20here%3A%20www.southasianheartcenter.org%0A%0ABest%2C%0A'
        st.link_button("Whatsapp", url)
    with col_mess:
        url = 'sms:&body=Hello%2C%0A%0AI%20recently%20came%20across%20SCORE%2C%20a%20tool%20from%20El%20Camino%20Health%2C%20South%20Asian%20Heart%20Center%20that%20compares%20your%20lipids%20and%20other%20cardio-metabolic%20markers%20against%20your%20peers%2C%20matching%20your%20age%2C%20gender%2C%20ethnicity%2C%20and%20medication%20use.%0A%0AThis%20may%20help%20you%20calibrate%20your%20markers%20and%20take%20steps%20to%20improve%20your%20risk%20profile.%0A%0ACheck%20it%20out%20it%20out%20here%3A%20https%3A//scores.streamlit.app/%0A%0AYou%20may%20read%20more%20about%20the%20work%20of%20El%20Camino%20Health%27s%20South%20Asian%20Heart%20Center%2C%20a%20non-profit%20with%20the%20mission%20to%20reduce%20the%20high%20incidence%20of%20diabetes%20and%20heart%20disease%20with%20evidence-based%2C%20culturally%20tailored%2C%20and%20lifestyle-focused%20prevention%20services%2C%20here%3A%20www.southasianheartcenter.org%0A%0ABest%2C%0A'
        st.link_button(":speech_balloon: Messages", url)

def config_sidebar():
    st.markdown(
        """
        <style>
            section[data-testid="stSidebar"] {
                width: 100px !important; # Set the width to your desired value
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.sidebar:
        st.write("Provide Feedback:")
        col_empty, col_up, col_down, col_empty2 = st.columns([0.05,0.1, 0.1,0.2])
        with col_up:
            if st.button("👍", help="Like this"):
                update_count("thumbs_up")
                header_popup(True)
                    
        with col_down:
            if st.button("👎", help="Needs improvement"):
                update_count("thumbs_down")
                header_popup(False)
        
        st.write("Share with others")
        with stylable_container(
            key="container_with_border",
            css_styles=r"""
                button div:before {
                    font-family: "Font Awesome 5 Free";
                    content: '\f14d'; # Get the share icon
                    display: inline-block;
                    padding-right: 0px;
                    vertical-align: middle;
                    font-weight: 900;
                    color: black;
                }
                """,
            ): 
            col_empty, col_up, col_down, col_empty2 = st.columns([0.05, 0.1, 0.1, 0.2])
            with col_up:
                share = st.button("", help='Share with others')
        
                if share:
                    share_popup()

        global colorblind_mode
        st.write("High Contrast Mode")
        col_empty, col_up, col_empty2 = st.columns([0.03,0.25, 0.1])
        with col_up:
            colorblind_mode = st.toggle("On/Off", help='Colorblindess improvements')

########################### PDF CODE FOR LATER ##############################

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
    #         pdf.cell(40, 10, report_text)
    #         pdf.image('LBDHDD.jpeg', x=5, y=30, w=200, h=25.633)  # Adjust 'x', 'y', 'w', and 'h' as needed

    #         pdf_file_path = 'test.pdf'  # Adjust the path and filename as needed
    #         pdf.output(pdf_file_path)
            
    #         html = create_download_link(pdf.output(dest="S").encode("latin-1"), "test")
    #         st.markdown(html, unsafe_allow_html=True)

    #         Send the PDF via email with yagmail
    #         yag = yagmail.SMTP('sanaa.bhorkar@gmail.com', 'txhamunwqrefciwl', host='smtp.gmail.com', port=587, smtp_starttls=True, smtp_ssl=False)

    #         Enclose the PDF
    #         yag.send(
    #             to=email,
    #             subject="Your SAHA Report",
    #             contents="Report attached.",
    #             attachments=['test.pdf']
    #         )

    #         Close SMTP connection
    #         yag.close()

    #         with open("emails.txt", "a") as f: save emails to a text file
    #             date = datetime.datetime.now()
    #             f.write(f"{date}, {email}\n")
    # header.write("""<div class='fixed-header'/>""", unsafe_allow_html=True)

########################### PDF CODE END ##############################

config_sidebar()

########################### SIDEBAR END ##############################

########################### HEADER ##############################
st.image(image_path)
col_score, col_records = st.columns([0.95, 0.05])
with col_score:
    st.write(f"SCORE evaluates your cardiometabolic risk profile and compares your markers against peers based on your gender, age, and ethnicity.")

########################### HEADER END ##############################

########################### PAGE SETTINGS AND CONSTANTS ##############################
if "title_expander" not in st.session_state:
    st.session_state.title_expander = "About Me"

expand_label = "About Me"
aboutMe_expand = st.expander(st.session_state.title_expander or expand_label, expanded=True)
analysis = st.container(border=True)

st.markdown(""" 
        <style>
        *:not(fixed-header)::before, 
            *|not(fixed-header)::after {
           box-sizing: inherit;
        }
        </style>""", unsafe_allow_html=True)

red = "#ee3942"
orange = "#fec423"
green = "#67bd4a"
darkgreen = "#75975e"
regugreen = "#9dba82"
lightgreen = "#c7ddb5"

SAHC_RENAME_FILE = os.path.join(SAHC_DATA_DIR, 'renamed_merged_data_noPID.csv')
NHANES_FILE = os.path.join(DATA_DIR, 'nhanes_merged_data.csv')
df_sahc = pd.read_csv(SAHC_RENAME_FILE)
df_nhanes = pd.read_csv(NHANES_FILE)

UNITS_MAP = {
    'LBDHDD': "mg/dL", 'LBDLDL': "mg/dL", 'LBXTC': "mg/dL", 'LBXTR': "mg/dL", 'LBXGH': "%",
    'LBXGLU': "mg/dL", 'BPXOSY1': "mmHg", 'BPXODI1': "mmHg", 'LBXHGB': "g/dL", 'TotHDLRat': "", 'BMXBMI': ""
}

NAME_MAP = {
    'LBXTC': 'Total Cholesterol', 'LBDLDL': 'LDL', 'LBDHDD': 'HDL', 
    'LBXTR': 'Triglycerides', 'TotHDLRat': 'Total Cholesterol to HDL ratio', 
    'LBXGLU': 'Fasting Glucose', 'LBXGH': 'HbA1c',
    'BMXBMI': 'Body Mass Index', 'BPXOSY1': 'Systolic Blood Pressure', 'BPXODI1': 'Diastolic Blood Pressure', 
}

DROPDOWN_SELECTION = {
    'Total Cholesterol': 'LBXTC', 'LDL': 'LBDLDL', 'HDL': 'LBDHDD',
    'Triglycerides': 'LBXTR', 'Total Cholesterol:HDL': 'TotHDLRat',
    'Fasting Glucose': 'LBXGLU', 'HbA1c': 'LBXGH',
    'Body Mass Index': 'BMXBMI', 'Systolic BP': 'BPXOSY1', 'Diastolic BP': 'BPXODI1', 
}

AHA_RANGES = {
    'Triglycerides (mg/dL)': ("Optimal", 150, "Borderline", 200, "At risk", None, None),
    'HDL (mg/dL)': ("At risk", 40, "Borderline", 60, "At Risk", None, None),
    'LDL (mg/dL)': ("Optimal", 100, "Borderline", 160, "At risk", None, None),
    'Total Cholesterol (mg/dL)': ("Optimal", 200, "Borderline", 240, "At risk", None, None),
    'Fasting Glucose (mg/dL)': ("Optimal", 100, "Borderline", 126, "At risk", None, None),
    'Systolic BP (mmHg)': ("Optimal", 120, "At risk", None, None, None, None),
    'Diastolic BP (mmHg)': ("Optimal", 80, "At risk", None, None, None, None),
    'Total Cholesterol:HDL ()':("Optimal", 3.5, "Borderline", 5.1, "At risk", None, None),
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

########################### PAGE SETTINGS AND CONSTANTS END ##############################

########################### EXPANDER SET UP ##############################

if 'title_list' not in st.session_state:
    st.session_state.title_list = []

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
            selected_meds.clear()
            st.session_state.selected_meds = []
        if selected_meds != []:
            selected_meds = ', '.join(selected_meds)
            st.session_state.title_list.append(f"Current medications: {selected_meds}")
    
    if st.session_state.title_list:
        st.session_state.title_expander = f"About Me: {', '.join(st.session_state.title_list)}"
    else:
        st.session_state.title_expander = "About Me"

with aboutMe_expand:
    col1, col12, col2, col22, col3, col32, col4, col42 = st.columns([.25, .1, .25, .1, .25, .1, .22, .08], vertical_alignment='top')

    medCholOptions = {'Yes': 1, 'No': 2}
    medDiabOptions = {'Yes': 1, 'No': 2}
    medBPOptions = {'Yes': 1, 'No': 2}

    with col1:
        gender = st.selectbox('Gender assigned at birth', list(genderOptions.keys()), key="selected_gender", on_change=update_title, index=None)
    with col2:
        age_group = st.selectbox('Age group', list(ageOptions.keys()), key="selected_age", on_change=update_title, index=None)
    with col3:
        ethnicity = st.selectbox('Ethnicity', ['Any ethnicity', 'South Asians only'], key="selected_ethnicity", on_change=update_title, index=None)
    with col4:
        med_options = ['None', 'Cholesterol', 'Diabetes', 'Blood Pressure']
        medications_select = st.multiselect(label="Medications", options=med_options, key="selected_meds", on_change=update_title)

        medChol = 'No'
        medDiab = 'No'
        medBP = 'No'

        if 'Cholesterol' in medications_select:
            medChol = 'Yes'
        if 'Diabetes' in medications_select:
            medDiab = 'Yes'
        if 'Blood Pressure' in medications_select:
            medBP = 'Yes'

########################### EXPANDER TITLE SET UP END ##############################

def new_load_files():
    global df_nhanes, df_sahc
    if ethnicity is None or ethnicity != 'South Asians only':
        return df_nhanes
    else:
        return df_sahc

########################### DATAFRAME CLEAN UP AND FILTER ##############################

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

def ui_choose(df):
    df2 = df.copy()

    if gender is not None:
        genderFilter = [genderOptions[gender]]
        df2 = df2[df2['RIAGENDR'].isin(genderFilter)]
    
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

    if age_group is not None:
        ageFilter = [ageOptions[age_group]]
        length = len(df2[df2['Age_Group'].isin(ageFilter)])

        if length < 15:
            next_age_group = get_next_age_group(age_group)
            ageFilter.append(ageOptions[next_age_group])
        
        df2 = df2[df2['Age_Group'].isin(ageFilter)]

    length = len(df2)

    with col_records:
        st.markdown(f"<span style='color:white;'>({len(df2)})</span>", unsafe_allow_html=True)
    
    if len(df2) < 15:
        st.write(f"Not enough records found to compare. Please remove medication usage and try again.") 
    
    # Edit the BMI and HDL numbers for ethnicity/gender
    if ethnicity == 'South Asians only':
        AHA_RANGES['Body Mass Index'] = ("Low", 18.5, "Optimal", 23, "Borderline", 25, "At risk")
    if gender == 'Female':
        AHA_RANGES['HDL (mg/dL)'] = ("At risk", 50, "Borderline", 60, "At Risk", None, None)

    return df2

########################### DATAFRAME CLEAN UP AND FILTER END ##############################

########################### INFORMATION POP UP ##############################

@st.dialog(" ", width='large')
def popup(acro, column, user_input, gender, race, age_range, med, on_med, prob, p25, p50, p75, p90, low_number, high_number, status, prop):

    if on_med == 'Yes':
        
        if age_range is None and race is None and gender is None:
            person_label = f"(Person ON {med}-lowering medication)"
        elif gender is None and race is None:
            person_label = f"(Person aged between {age_range} years, ON {med}-lowering medication)"
        elif age_range is None and gender is None:
            person_label = f"({race}, ON {med}-lowering medication)"
        elif age_range is None and race is None:
            person_label = f"({gender}, ON {med}-lowering medication)"
        elif gender is None:
            person_label = f"({race}, aged between {age_range} years, ON {med}-lowering medication)"
        elif race is None:
            person_label = f"({gender}, aged between {age_range} years, ON {med}-lowering medication)"
        else:
            person_label = f"({race} {gender.lower()}, aged between {age_range} years, ON {med}-lowering medication)"
    else:
        if age_range is None and race is None and gender is None:
            person_label = f"(Person NOT ON {med}-lowering medication)"
        elif gender is None and race is None:
            person_label = f"(Person aged between {age_range} years, NOT ON {med}-lowering medication)"
        elif age_range is None and gender is None:
            person_label = f"({race}, NOT ON {med}-lowering medication)"
        elif age_range is None and race is None:
            person_label = f"({gender}, NOT ON {med}-lowering medication)"
        elif gender is None:
            person_label = f"({race}, aged between {age_range} years, NOT ON {med}-lowering medication)"
        elif race is None:
            person_label = f"({gender}, aged between {age_range} years, NOT ON {med}-lowering medication)"
        else:
            person_label = f"({race} {gender.lower()}, aged between {age_range} years, NOT ON {med}-lowering medication)"

    st.write(f"""
            **Your {column} compared to others in your peer group**
            <br>
            {person_label}
            """, unsafe_allow_html=True)

    st.write(f"""
        **Your {column}: {user_input:.1f} {UNITS_MAP[acro]}
        <br>
        Risk classification: {status}**
        """, unsafe_allow_html=True)
    
    if "LDL" in column:
        st.write(f"**{prop:.0f}%** of individuals in your peer group have an {column} < {user_input:.1f}")
    else:
        st.write(f"**{prop:.0f}%** of individuals in your peer group have a {column} < {user_input:.1f}")

    if STEP_SIZE[acro] == 0.1:
        data = pd.DataFrame({
            'Percentile': [f'{column} Level'],
            '25th': [round(p25, 1)],
            '50th': [round(p50, 1)],
            '75th': [round(p75, 1)],
            '90th': [round(p90, 1)]
        })
    else:
        data = pd.DataFrame({
            'Percentile': [f'{column} Level'],
            '25th': [int(p25)],
            '50th': [int(p50)],
            '75th': [int(p75)],
            '90th': [int(p90)]
        })

    html = data.to_html(index=False)
    html = html.replace('<thead>', 
                    '<thead style="background-color: lightgray;">')
    html_parts = html.split('<td>')

    html_parts[1] = '<td style="text-align: left;">' + html_parts[1]

    for i in range(2, 6):
        if i < len(html_parts):
            html_parts[i] = '<td style="text-align: center;">' + html_parts[i]

    html = ''.join(html_parts)

    st.markdown(html, unsafe_allow_html=True)

    if 'Body Mass' in column:
        st.write(f"""
        According to AHA guidelines, the **optimal** value for {column} is **{low_number} - {high_number - 0.1}**.
        <br>
        In your peer group, the estimated probability of having a sub-optimal {column} of < {low_number} or ≥ {high_number} is **{prob:.0f}%.**
        """, unsafe_allow_html=True)
    elif 'HDL (mg/dL)' in column:
        st.write(f"""
            According to AHA guidelines, the **optimal** value for {column} is < **{low_number}**.
            <br>
            In your peer group, the estimated probability of having a sub-optimal {column} of ≤ {low_number} is **{prob:.0f}%.**
            """, unsafe_allow_html=True)
    else:
        st.write(f"""
            According to AHA guidelines, the **optimal** value for {column} is < **{low_number}**.
            <br>
            In your peer group, the estimated probability of having a sub-optimal {column} of ≥ {low_number} is **{prob:.0f}%.**
            """, unsafe_allow_html=True)
    
    if status != 'Optimal':
        st.markdown('To understand your cardio-metabolic risk profile further, and to receive personalized guidance to improve lifestyle behaviors such as diet, exercise, sleep, stress management and more, <a href="https://www.elcaminohealth.org/community/lifestyle-consult">schedule</a> a 60 minute lifestyle medicine consult with El Camino Health.', unsafe_allow_html=True)
    else:
        "Your marker is in optimal range. Continue working on your lifestyle behaviors to keep this marker in range."

########################### INFORMATION POP UP END ##############################

########################### DROPDOWN AND INPUTS FOR METRICS ##############################

if 'metric_list' not in st.session_state:
    st.session_state.metric_list = deque()

# Helper function to fix the boundary condition
def update_header_color(new_status):
    global status, header_color, at_risk, optimal, borderline

    # st.write(new_status)

    if new_status == 'Optimal':
        header_color = optimal
    elif new_status == 'At risk':
        header_color = at_risk
    else:
        header_color = borderline

    status = new_status

def show_analysis(df):
    global status, header_color, at_risk, optimal, borderline
    user_inputs = {}
    
    with analysis:

        col_dropdown, col_empty, col_input, col_empty2, col_input2, col_empty3 = st.columns([.25, .1, .25, .1, .25, .4], vertical_alignment='top')
        st.divider() 

        with col_dropdown:
            metric = st.selectbox('My Risk Profile Markers', list(DROPDOWN_SELECTION.keys()))
        
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

        if len(st.session_state.metric_list) == 0:
            st.session_state.metric_list.appendleft({'column':column, 'input': user_inputs[column], 'columnName': columnName})
        else:
            exists = False
            delete = -1
            saved_input = None
            for index, selection in enumerate(st.session_state.metric_list):
                if selection['column'] == column:
                    exists = True
                    delete = index
                    break

            if exists:
                    saved_input = st.session_state.metric_list[delete]
                    del st.session_state.metric_list[delete]

            if user_inputs[column] is not None and user_inputs[column] >= 0:
                st.session_state.metric_list.appendleft({'column':column, 'input': user_inputs[column], 'columnName': columnName})
            elif saved_input != None:
                st.session_state.metric_list.appendleft(saved_input)

########################### DROPDOWN AND INPUTS FOR METRICS END ##############################

########################### SHOW GRAPHS ##############################

        num_elements = 10 # Number of total metrics available, if user checks all of them

        cols = [st.empty() for _ in range(num_elements)]

        for i, column_dict in enumerate(st.session_state.metric_list):

            column = column_dict['column']
            columnName = column_dict['columnName']

            with cols[i].container():

                placeholder = st.empty()
                header = f"{NAME_MAP[column]}"

                col8, col9 = st.columns([0.65, 0.35], vertical_alignment='top', gap='small')
            
                with col8:
                    user_input = column_dict['input']

                    if user_input == None:
                        break

                    placeholder.markdown(f"#### {header}", unsafe_allow_html=True)

                    array = df[column].dropna()
                    if len(array.index) < 5:
                        st.markdown(f"Not enough data available for {columnName}.")
                        continue

                    if STEP_SIZE[column] == 0.1 or column == 'BMXBMI':
                        header = f"{NAME_MAP[column]}: {user_input:.1f} {UNITS_MAP[column]}"
                    else:
                        header = f"{NAME_MAP[column]}: {user_input:.0f} {UNITS_MAP[column]}"

                    low_number = AHA_RANGES[columnName][1]
                    high_number = AHA_RANGES[columnName][3]

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

                    # Find the user percentile
                    user_percentile = int(np.mean(sorted_array <= user_input) * 100)
                    if user_percentile == 100:
                        user_percentile = 99

                    status = 'Optimal'
                    
                    if high_percentile < 99:
                        ax.plot([high_percentile, high_percentile], [0.6, 0.95], color='white', lw=2, label='Demarcation', zorder=999)
                    ax.plot([low_percentile, low_percentile], [0.6, 0.95], color='white', lw=2, label='Demarcation', zorder=999)
                    if columnName == 'Body Mass Index':
                        ax.plot([extra_high_percentile, extra_high_percentile], [0.6, 0.95], color='white', lw=2, label='Demarcation', zorder=999)
                    
                    if colorblind_mode == True:
                        optimal = lightgreen
                        borderline = darkgreen
                        at_risk = darkgreen
                    else:
                        optimal = green
                        borderline = orange
                        at_risk = red

                    if columnName == 'HDL (mg/dL)':
                        ax.add_patch(Rectangle((0, 0.6), 
                                        low_percentile, 0.35, 
                                        color=at_risk, fill=True, zorder=90))
                        ax.add_patch(Rectangle((low_percentile, 0.6), 
                                    (high_percentile - low_percentile), 0.35, 
                                    color=borderline, fill=True, zorder=90))
                        ax.add_patch(Rectangle((high_percentile, 0.6), 
                                    (100 - high_percentile), 0.35, 
                                    color=optimal, fill=True, zorder=90))
                    elif "BP " in columnName:
                        ax.add_patch(Rectangle((0, 0.6), 
                                        low_percentile, 0.35, 
                                        color=optimal, fill=True, zorder=90))
                        ax.add_patch(Rectangle((low_percentile, 0.6), 
                                    (high_percentile - low_percentile), 0.35, 
                                    color=at_risk, fill=True, zorder=90))
                    elif "Body Mass" in columnName:
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
                            update_header_color('At risk')
                        elif user_percentile < high_percentile:
                            update_header_color('Borderline')
                        else:
                            update_header_color('Optimal')
                    elif 'BP ' in columnName:
                        if user_percentile <= low_percentile:
                            #Boundary condition of percentile
                            if user_input >= low_number:
                                update_header_color('At risk')
                            else:
                                update_header_color('Optimal')
                        else:
                            if user_input <= low_number:
                                update_header_color('Optimal')
                            else:
                                update_header_color('At risk')    
                    elif columnName == 'Body Mass Index':
                        if user_percentile < low_percentile:
                            if user_input >= low_number:
                                update_header_color('Optimal')
                            else:
                                update_header_color('At risk')
                        elif user_percentile <= high_percentile:
                            if user_input > high_number:
                                update_header_color('Borderline')
                            else:
                                update_header_color('Optimal')
                        else:
                            if user_input >= extra_high_number:
                                update_header_color('At risk')
                            else:
                                update_header_color('Borderline')
                    else:
                        if user_percentile <= low_percentile:
                            # Boundary condition of percentile
                            if user_input > low_number:
                                update_header_color('Borderline')
                            else:
                                update_header_color('Optimal')
                        elif user_percentile <= high_percentile:
                            if user_input <= low_number:
                                update_header_color('Optimal')
                            elif user_input >= high_number:
                                update_header_color('At risk')
                            else:
                                update_header_color('Borderline')
                            
                        elif user_percentile >= high_percentile:
                            update_header_color('At risk')
                            if user_input < high_number:
                                update_header_color('Borderline')
                                        
                    # Show the circle for the user input on the graph
                    ax.scatter(user_percentile, 0.85, zorder=999, s=scatter_size+225, edgecolors='k')
                    ax.scatter(user_percentile, 0.85, color=header_color, zorder=1000, label='Your Input', s=scatter_size, edgecolors='w', linewidth=3)

                    placeholder.markdown(f"#### <span style='color:{header_color};'>{header}</span>", unsafe_allow_html=True) # Change the color of the title for each graph

                    ax.set_ylim(0.4, 1.1)
                    ax.set_yticks([]) 
                    ax.set_xticks([])

                    if status == 'Borderline':
                        text_color = 'black'
                    else:
                        text_color = 'white'

                    if STEP_SIZE[column] == .1 or column == 'BMXBMI':
                        ax.annotate(f'{user_input: .1f}', xy=(user_percentile, 0.865), xytext=(user_percentile - 0.25, 0.79),
                                    horizontalalignment='center', color = text_color, zorder=1001, weight='bold', fontsize=16)
                    else:
                        ax.annotate(f'{user_input: .0f}', xy=(user_percentile, 0.865), xytext=(user_percentile - 0.25, 0.79),
                                    horizontalalignment='center', color = text_color, zorder=1001, weight='bold', fontsize=16)
                    
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
                        plt.annotate(f'{AHA_RANGES[columnName][2]}', xy=(low_percentile, 0.675), xytext=(low_percentile, 0.45),
                                horizontalalignment='left', weight='bold')
                        plt.annotate(f'18.5-{AHA_RANGES[columnName][3] - 0.1}', xy=(low_percentile, 0.657), xytext=(low_percentile, 0.3),
                                horizontalalignment='left')
                        plt.annotate(f'{AHA_RANGES[columnName][4]}', xy=(low_percentile, 0.675), xytext=(high_percentile, 0.45),
                                horizontalalignment='left', weight='bold')
                        plt.annotate(f'{AHA_RANGES[columnName][3]}-{AHA_RANGES[columnName][5] - 0.1}', xy=(high_percentile - 1, 0.657), xytext=(high_percentile, 0.3),
                                horizontalalignment='left')
                        plt.annotate(f'{AHA_RANGES[columnName][6]}', xy=(low_percentile, 0.675), xytext=(extra_high_percentile, 0.45),
                                horizontalalignment='left', weight='bold')
                        plt.annotate(f'≥{AHA_RANGES[columnName][5]}', xy=(extra_high_percentile - 1, 0.657), xytext=(extra_high_percentile, 0.3),
                                horizontalalignment='left')
                    else:
                        if high_number < 1000:
                            plt.annotate(f'{AHA_RANGES[columnName][4]}', xy=(high_percentile, 0.675), xytext=(high_percentile, 0.45),
                                        horizontalalignment='left', weight='bold')
                            plt.annotate(f'≥{high_number}', xy=(high_percentile, 0.675), xytext=(high_percentile, 0.3),
                                        horizontalalignment='left')
                        if low_number > 0:
                            plt.annotate(f'{AHA_RANGES[columnName][0]}', xy=(low_percentile, 0.675), xytext=(low_percentile - 1, 0.45),
                                        horizontalalignment='right', weight='bold')
                            plt.annotate(f'<{low_number}', xy=(low_percentile - 1, 0.657), xytext=(low_percentile - 1, 0.3),
                                        horizontalalignment='right')
                        if high_number == 1000:
                            plt.annotate(f'{AHA_RANGES[columnName][2]}', xy=(low_percentile, 0.675), xytext=(low_percentile, 0.45),
                                            horizontalalignment='left', weight='bold')
                            plt.annotate(f'≥{low_number}', xy=(low_percentile, 0.675), xytext=(low_percentile, 0.3),
                                            horizontalalignment='left')
                        else:
                            plt.annotate(f'{AHA_RANGES[columnName][2]}', xy=(low_percentile, 0.675), xytext=(low_percentile, 0.45),
                                            horizontalalignment='left', weight='bold')
                            if columnName == 'HbA1c (%)':
                                plt.annotate(f'{low_number}-{high_number}', xy=(low_percentile, 0.675), xytext=(low_percentile, 0.3),
                                            horizontalalignment='left')
                            elif columnName == 'Total Cholesterol:HDL ()':
                                plt.annotate(f'{low_number}-{high_number}', xy=(low_percentile, 0.675), xytext=(low_percentile, 0.3),
                                            horizontalalignment='left')
                            else:
                                plt.annotate(f'{low_number}-{high_number - 1}', xy=(low_percentile, 0.675), xytext=(low_percentile, 0.3),
                                            horizontalalignment='left')

                    for spine in ax.spines.values():
                        spine.set_visible(False)

                    st.pyplot(fig)
                    plt.close(fig)

########################### SHOW GRAPHS END ##############################

########################### INFORMATION BUTTON ##############################

                    if "to HDL" in columnName:
                        value = AHA_RANGES[columnName][1]
                    elif "HDL" in columnName:
                        value = AHA_RANGES[columnName][1]
                    elif "Body Mass Index" in columnName:
                        value = AHA_RANGES[columnName][3]
                    else:
                        value = AHA_RANGES[columnName][1]

                    prob_percentile = int(np.mean(sorted_array <= value) * 100)
                    prob = 100 - prob_percentile # Get the probability of having a sub-optimal value

                    df4 = df.copy()
                    df4 = df4[column]
                    df4 = df4.dropna()
                    total = df4.shape[0]

                    # Get the percentage of people that have a value <= the entered value
                    df4 = df4[df4 <= user_input]
                    top = df4.shape[0]          
                    prop = (top / total) * 100

                    percentile_25 = int(np.percentile(sorted_array, 25))
                    percentile_50 = int(np.percentile(sorted_array, 50))
                    percentile_75 = int(np.percentile(sorted_array, 75))
                    percentile_90 = int(np.percentile(sorted_array, 90))

                    popup_column = NAME_MAP[column]

                with col9:
                    digit = user_percentile % 10
                    suffix = 'th'

                    if user_percentile not in (11,12,13):
                        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(digit, 'th')
                    
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

                    more_info = st.button(f'ⓘ {user_percentile: .0f}{suffix} percentile compared to peers in your group', key=column, type='primary')

                if more_info:
                        if "HDL (mg/dL)" in columnName or "DL" in columnName or "Trig" in columnName or "Chol" in columnName:
                            popup(column, popup_column, user_input, gender, ethnicity, age_group, "cholesterol", medChol, prob, 
                                percentile_25, percentile_50, percentile_75, percentile_90, low_number, high_number, status, prop)
                        elif "Glucose" in columnName or "A1C" in columnName:
                            popup(column, popup_column, user_input, gender, ethnicity, age_group, "blood sugar", medDiab, prob, 
                                percentile_25, percentile_50, percentile_75, percentile_90, low_number, high_number, status, prop)   
                        else:
                            popup(column, popup_column, user_input, gender, ethnicity, age_group, "blood pressure", medBP, prob, 
                                percentile_25, percentile_50, percentile_75, percentile_90, low_number, high_number, status, prop)

########################### INFORMATION BUTTON END ##############################

########################### MAIN EXECUTION ##############################
df_c = new_load_files()
df_d = ui_choose(df_c)
show_analysis(df_d)

########################### MAIN EXECUTION END ##############################

########################### FOOTER ##############################
up, down = get_counts()
st.markdown(f"<div style='text-align: center'> Total Visitors: {total_unique_views}</div>", unsafe_allow_html=True)
st.markdown(f"<div style='text-align: center'> Version {VERSION}</div>", unsafe_allow_html=True)
st.markdown(f"<div style='text-align: center'><span style='color:white;'>({up}, {down})</span></div>", unsafe_allow_html=True)
conn.close()