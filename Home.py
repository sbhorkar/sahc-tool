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
from io import BytesIO

st.set_page_config(page_title="SAHC Comparison Tool", page_icon=":anatomical_heart:", layout="wide")

DIR = os.getcwd()
PLOT_DIR = DIR + '/plots'
LOGO_DIR = DIR + '/logo/'
DATA_DIR = DIR + '/data/'
OUTPUT_DIR = DIR + '/output/'

image_path = os.path.join(LOGO_DIR, '4.svg')

def create_download_link(val, filename):
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'

header = st.container()
with header:
    col1, col2, col3 = st.columns([1, 3, 1], vertical_alignment='bottom')
    with col1:
        st.image(image_path, width=500)
    with col3:
        report_text = st.text_input("Download a PDF report!", placeholder='Your email')

        if report_text == "":
            button_disabled = True
        else:
            button_disabled = False

        export_as_pdf = st.button("Export Report", disabled=button_disabled)

        if export_as_pdf:
            pdf = FPDF('P','mm','A4');
            pdf.add_page()
            pdf.set_font('Arial', 'B', 16)
            # pdf.cell(40, 10, report_text)
            pdf.image('LBDHDD.jpeg', x=5, y=30, w=200, h=25.633)  # Adjust 'x', 'y', 'w', and 'h' as needed

            pdf_file_path = 'test.pdf'  # Adjust the path and filename as needed
            pdf.output(pdf_file_path)
            
            # html = create_download_link(pdf.output(dest="S").encode("latin-1"), "test")
            # st.markdown(html, unsafe_allow_html=True)

            # Send the PDF via email with yagmail
            yag = yagmail.SMTP('sanaa.bhorkar@gmail.com', 'txhamunwqrefciwl', host='smtp.gmail.com', port=587, smtp_starttls=True, smtp_ssl=False)

            # Enclose the PDF
            yag.send(
                to=report_text,
                subject="Your ARCH Report",
                contents="Report attached.",
                attachments=['test.pdf']
            )

            # Close SMTP connection
            yag.close()

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
        border-bottom: 1px solid grey;
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

NAME_MAP = {
    'LBXTR': 'Triglycerides (mg/dL)', 'LBDHDD': 'HDL (mg/dL)', 'LBDLDL': 'LDL (mg/dL)',
    'LBXTC': 'Total Cholesterol (mg/dL)', 'LBXGLU': 'Fasting Glucose (mg/dL)',
    'BPXOSY1': 'Systolic Blood Pressure (mmHg)', 'BPXODI1': 'Diastolic Blood Pressure (mmHg)', 'BPXOPLS1': 'Pulse'
}

# Uniform AHA ranges regardless of gender and age
AHA_RANGES = {
    'Triglycerides (mg/dL)': (None, 150, None),
    'HDL (mg/dL)': (40, 60, None),
    'LDL (mg/dL)': (None, 100, None),
    'Total Cholesterol (mg/dL)': (140, 160, None),
    'Fasting Glucose (mg/dL)': (None, 100, None),
    'Systolic Blood Pressure (mmHg)': (None, 120, None),
    'Diastolic Blood Pressure (mmHg)': (None, 80, None),
    'Pulse': (60, 100, None)
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

demoExpand = st.expander("My demographics", expanded=True)

genderOptions = {'Male': 1, 'Female': 2}
# ageOptions = {'20-40': 20, '40-60': 40, '60-80': 60, '80+': 80}
ageOptions = {'19-33': 19, '34-48': 34, '49-64': 49, '65-78': 65,'79+': 79}
dataSelection = ['NHANES', 'SAHC']

with demoExpand:
    col1, col2, col3 = st.columns(3, vertical_alignment='top')

    with col1:
        gender_toggle = st.toggle('Male?', value=True)
        if gender_toggle is True:
            gender = 'Male'
        else:
            gender = 'Female'
    with col2:
        age_group = st.selectbox('Age group', list(ageOptions.keys()), index=2)
    with col3:
        data_toggle = st.toggle("South Asian?", disabled=True)
        if data_toggle is True:
            data_select = 'SAHC'
        else:
            data_select = 'NHANES'

medCholOptions = {'Yes': 1, 'No': 2}
medDiabOptions = {'Yes': 1, 'No': 2}
medBPOptions = {'Yes': 1, 'No': 2}

medsExpand = st.expander("### My medication use", expanded=True)

with medsExpand:
    col4, col5, col6 = st.columns(3, vertical_alignment='bottom')

    with col4:
        # medChol = st.selectbox('Cholesterol medication', options=list(medCholOptions.keys()), placeholder="No", disabled=True)
        chol_toggle = st.toggle("On cholesterol-lowering medication?", value=False)
        if chol_toggle is True:
            medChol = 'Yes'
        else:
            medChol = 'No'
    with col5:
        # medDiab = st.selectbox('Diabetes medication', options=list(medDiabOptions.keys()), placeholder="No", disabled=True)
        diab_toggle = st.toggle("On blood sugar-lowering medication?", value=False)
        if diab_toggle is True:
            medDiab = 'Yes'
        else:
            medDiab = 'No'
    with col6:
        # medBP = st.selectbox('Blood pressure medication', options=list(medBPOptions.keys()), placeholder="No", disabled=True)
        bp_toggle = st.toggle("On blood pressure-lowering medication?", value=False)
        if bp_toggle is True:
            medBP = 'Yes'
        else:
            medBP = 'No'

# @st.cache_data
def load_files(debugging):
    if data_select == 'NHANES':
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
        df_combined = pd.merge(df_combined, df_bpq[['SEQN', 'BPQ090D', 'BPQ100D', 'BPQ040A', 'BPQ050A', 'BPQ020']], on='SEQN', how='left')
        df_combined = pd.merge(df_combined, df_hdl[['SEQN', 'LBDHDD']], on='SEQN', how='left')
        df_combined = pd.merge(df_combined, df_tgl[['SEQN', 'LBXTR', 'LBDLDL']], on='SEQN', how='left')
        df_combined = pd.merge(df_combined, df_tch[['SEQN', 'LBXTC']], on='SEQN', how='left')
        df_combined = pd.merge(df_combined, df_glu[['SEQN', 'LBXGLU']], on='SEQN', how='left')
        df_combined = pd.merge(df_combined, df_bpx[['SEQN', 'BPXOSY1', 'BPXODI1', 'BPXOPLS1', 'BPXOSY2', 'BPXODI2', 'BPXOPLS2', 'BPXOSY3', 'BPXODI3', 'BPXOPLS3']], on='SEQN', how='left')
        # df_combined['Age_Group'] = df_combined['RIDAGEYR'].apply(lambda age: 20 * int(age / 20))
        df_combined['Age_Group'] = df_combined['RIDAGEYR'].apply(map_age_to_group)

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

    if medChol == 'Yes':
        df2 = df2[df2['BPQ100D'].isin(medCholFilter)]
    elif medChol == 'No':
        df2 = df2[df2['BPQ090D'].isin(medCholFilter) | df2['BPQ100D'].isin(medCholFilter)]
    
    df2 = df2[df2['DIQ070'].isin(medDiabFilter)]

    if medBP == 'Yes':
        df2 = df2[df2['BPQ040A'].isin(medBPFilter)]
    elif medBP == 'No':
        df2 = df2[df2['BPQ020'].isin(medBPFilter) | df2['BPQ040A'].isin(medBPFilter)]

    st.write(f"{len(df2)} records found.")

    if len(df2) < 15:
        next_age_group = get_next_age_group(age_group)
        ageFilter.append(ageOptions[next_age_group])
        df2 = df[df['RIAGENDR'].isin(genderFilter)]
        df2 = df2[df2['Age_Group'].isin(ageFilter)]

        if medChol == 'Yes':
            df2 = df2[df2['BPQ100D'].isin(medCholFilter)]
        elif medChol == 'No':
            df2 = df2[df2['BPQ090D'].isin(medCholFilter) | df2['BPQ100D'].isin(medCholFilter)]
        
        df2 = df2[df2['DIQ070'].isin(medDiabFilter)]

        if medBP == 'Yes':
            df2 = df2[df2['BPQ040A'].isin(medBPFilter)]
        elif medBP == 'No':
            df2 = df2[df2['BPQ020'].isin(medBPFilter) | df2['BPQ040A'].isin(medBPFilter)]

        st.write(f"{len(df2)} records found after expanding age group.")

    return df2


@st.experimental_dialog("More information")
def popup(column, user_input, user_percentile, gender, age_range, med, on_med, prob, value, p25, p50, p75, p90):
    st.write(f"The estimated probability of an optimal {column} level >= {value} for a {gender.lower()} between {age_range} is {prob * 100: .0f}%.")
    if on_med:
        st.write(f"An {column} of {user_input} is at {user_percentile: .0f}%ile for a {gender.lower()} between {age_range} who is on {med}-lowering medication.")
    else:
        st.write(f"An {column} of {user_input} is at {user_percentile: .0f}%ile for a {gender.lower()} between {age_range} who is not on {med}-lowering medication.")

    data = {
        '25th': [p25],
        '50th': [p50],
        '75th': [p75],
        '90th': [p90]
    }

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(data)

    if column == 'HDL (mg/dL)':
        st.write('### Percentiles and HDL levels:')
        st.dataframe(df, hide_index=True)

def show_analysis(df):
    st.markdown(f"### <u>Your risk profile markers</u>", unsafe_allow_html=True)
    
    # Number of elements
    num_elements = 8

    # Create placeholders for columns
    cols = [st.empty() for _ in range(num_elements)]
        
    validation_labels = {
        'LBDHDD': "Value should be between 20-100",
        'LBDLDL': "Value should be between 30-300",
        'LBXTC': "Value should be between 140-320",
        'LBXTR': "Value should be between 50-500",
        'LBXGLU': "Value should be between 50-150",
        'BPXOSY1': "Value should be between 90-200",
        'BPXODI1': "Value should be between 60-130",
        'BPXOPLS1': "Value should be between 50-150"
    }

    units_map = {
        'LBDHDD': "mg/dL",
        'LBDLDL': "mg/dL",
        'LBXTC': "mg/dL",
        'LBXTR': "mg/dL",
        'LBXGLU': "mg/dL",
        'BPXOSY1': "mmHg",
        'BPXODI1': "mmHg",
        'BPXOPLS1': "bpm"
    }

    user_inputs = {}

    for i, column in enumerate(['LBDHDD', 'LBDLDL', 'LBXTC', 'LBXTR', 'LBXGLU', 'BPXOSY1', 'BPXODI1', 'BPXOPLS1']):

        with cols[i].container():

            # col6, col7, col8, col9, col10, col11 = st.columns([.25, 1.6, .3, 5, 5, .25], vertical_alignment='bottom')
            # col6, col7, col8, col9, col10, col11 = st.columns([0.1, 0.15, 0.05, 0.3, 0.3, 0.1], vertical_alignment='bottom')
            # col6, col7, col8, col9, col10, col11 = st.columns([0.1, 0.12, 0.03, 0.325, 0.325, 0.1], vertical_alignment='bottom')
            col6, col7, col8, col9, col10, col11 = st.columns([0.05, 0.15, 0.03, 0.36, 0.36, 0.05], vertical_alignment='bottom')

            if column == 'LBDHDD':
                col6, col7, col8, col9, col10 = st.columns([0.08, 0.15, 0.03, 0.6, 0.08], vertical_alignment='bottom')
            
            with col7:
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
                            color: #FF4B4B; /* Ensure text color remains visible on hover */
                        }
                    </style>
                    """, unsafe_allow_html=True)
                # st.write("#")
                more_info = st.button(label='ⓘ', key=column, type='primary')

            with col9:
                columnName = NAME_MAP[column]
                user_input = user_inputs[column]  # Reference the user input from the dictionary

                array = df[column].dropna()
                if len(array.index) < 5:
                    st.markdown(f"Not enough data available for {columnName}.")
                    continue

                if user_input == 0:
                    # st.write(f"### ")
                    st.write(f"Please enter a value for {columnName}")
                    continue

                # st.write(f"####")

                if columnName in AHA_RANGES:
                    low_number = AHA_RANGES[columnName][0]
                    high_number = AHA_RANGES[columnName][1]
                else:
                    st.write(f"No AHA prescribed range available for {columnName}.")
                    continue

                sorted_array = np.sort(array)

                # Calculate the percentile
                if high_number == None:
                    high_number = 1000
                    high_percentile = 101
                else:
                    high_percentile = np.mean(sorted_array <= high_number) * 100
                if low_number == None:
                    low_number = 0
                    low_percentile = 0
                else:
                    low_percentile = np.mean(sorted_array <= low_number) * 100

                fig, ax = plt.subplots(figsize=(15, 1))

                # lines from low_percentile to high_percentile
                ax.plot([low_percentile, high_percentile], [0.725, 0.725], color=regugreen, lw=20, label='Healthy Range')

                if high_percentile < 100:
                    ax.plot([high_percentile + 1, 100], [0.725, 0.725], color=darkgreen, lw=20, label='High')
                if low_percentile > 0:
                    ax.plot([0, low_percentile - 1], [0.725, 0.725], color=lightgreen, lw=20, label="Low")

                # dot at user input position
                user_percentile = np.mean(sorted_array <= user_input) * 100
                if int(user_percentile) == 100:
                    user_percentile == 99.99
                if (user_input > high_number or user_input < low_number) and column != 'LBDHDD':
                    # ax.scatter(user_percentile, 0.85, color='red', zorder=5, label='Your Input')
                    header_color = "red"
                    placeholder.markdown(f"#### <span style='color:{header_color};'>{header}</span>", unsafe_allow_html=True)
                # else:

                if user_percentile < low_percentile:
                    ax.scatter(user_percentile, 0.85, color=lightgreen, zorder=5, label='Your Input', s=500, edgecolors=['black'])
                elif user_percentile > high_percentile:
                    ax.scatter(user_percentile, 0.85, color=darkgreen, zorder=5, label='Your Input', s=500, edgecolors=['black'])
                else:
                    ax.scatter(user_percentile, 0.85, color=regugreen, zorder=5, label='Your Input', s=500, edgecolors=['black'])

                ax.set_xlim(0, 100)
                ax.set_ylim(0.4, 1.1)
                ax.set_yticks([])
                ax.xaxis.tick_top()  
                ax.xaxis.set_label_position('top') 
                tick_positions = [0, 100, user_percentile]

                if user_percentile - 2 <= 0 or user_percentile + 2 >= 99:
                    tick_labels = ['0%ile', '99%ile', '']
                else:
                    tick_labels = ['0%ile', '99%ile', f'{user_percentile: .0f}%ile']
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(tick_labels)

                # ax.set_xlabel('Percentile (%)')

                ax.set_title(columnName)

                if user_percentile > high_percentile:
                    plt.annotate(f'{user_input}', xy=(user_percentile, 0.85), xytext=(user_percentile, 0.8),
                              horizontalalignment='center', color = 'white', zorder=10, weight='bold')
                else:
                    plt.annotate(f'{user_input}', xy=(user_percentile, 0.85), xytext=(user_percentile, 0.8),
                              horizontalalignment='center', color = 'black', zorder=10, weight='bold')
                # plt.annotate(f'{user_percentile: .0f}%', xy=(user_percentile, 0.85), xytext=(user_percentile, 0.925),
                #               horizontalalignment='center')

                if high_number < 1000:
                    plt.annotate('High', xy=(high_percentile, 0.65), xytext=(high_percentile, 0.45),
                                  horizontalalignment='left', weight='bold')
                    plt.annotate(f'>{high_number}', xy=(high_percentile, 0.65), xytext=(high_percentile, 0.3),
                                  horizontalalignment='left')
                if low_number > 0:
                    plt.annotate('Low', xy=(low_percentile - 1, 0.65), xytext=(low_percentile - 1, 0.45),
                                  horizontalalignment='right', weight='bold')
                    plt.annotate(f'<{low_number}', xy=(low_percentile - 1, 0.65), xytext=(low_percentile - 1, 0.3),
                                  horizontalalignment='right')
                plt.annotate('Normal', xy=(low_percentile, 0.65), xytext=(low_percentile, 0.45),
                                  horizontalalignment='left', weight='bold')
                plt.annotate(f'{low_number}-{high_number}', xy=(low_percentile, 0.65), xytext=(low_percentile, 0.3),
                                  horizontalalignment='left')

                for spine in ax.spines.values():
                    spine.set_visible(False)

                # fig.patch.set_alpha(0) make background transparent
                # ax.patch.set_alpha(0)

                st.pyplot(fig)

                fig.savefig(f'{column}.jpeg', bbox_inches='tight')
                plt.close(fig)

                # digit = user_input % 10
                # suffix = ''

                # if digit == 0:
                #     suffix == 'th'
                # if digit == 1:
                #     suffix = 'st'
                # elif digit == 2:
                #     suffix == 'nd'
                # elif digit == 3:
                #     suffix == 'rd'
                # else:
                #     suffix == 'th'

                kde = sm.nonparametric.KDEUnivariate(array)
                kde.fit(bw='scott')  # You can also specify bandwidth manually, e.g., bw=0.5

                # Generate a range of HDL values to estimate the CDF
                hdl_range = np.linspace(min(array), max(array), 1000)

                # Evaluate the KDE over the range
                pdf_values = kde.evaluate(hdl_range)

                # Calculate the CDF by integrating the PDF
                cdf_values = np.cumsum(pdf_values) * (hdl_range[1] - hdl_range[0])

                # Find the CDF value at HDL = 35
                value = user_input
                cdf_at_35 = np.interp(value, hdl_range, cdf_values)

                # Probability of HDL >= 35
                prob_greater_than = 1 - cdf_at_35
                prob_at = kde.evaluate(value)

                # # Display the result in Streamlit
                # st.write(f"Probability of HDL >= {value}: {probability_gte_35 * 100: .0f}%")
                # st.write(f"Probability of HDL = {value}: {prob_density_value[0]}")

                percentile_25 = int(np.percentile(sorted_array, 25))
                percentile_50 = int(np.percentile(sorted_array, 50))
                percentile_75 = int(np.percentile(sorted_array, 75))
                percentile_90 = int(np.percentile(sorted_array, 90))

                if more_info:
                    if "DL" in columnName or "Trig" in columnName or "Chol" in columnName:
                        popup(columnName, user_input, user_percentile, gender, age_group, "cholesterol", medChol, prob_greater_than, value, 
                              percentile_25, percentile_50, percentile_75, percentile_90)
                    elif "Glucose" in columnName:
                        popup(columnName, user_input, user_percentile, gender, age_group, "blood sugar", medDiab, prob_greater_than, value, 
                              percentile_25, percentile_50, percentile_75, percentile_90)   
                    else:
                        popup(columnName, user_input, user_percentile, gender, age_group, "blood pressure", medBP, prob_greater_than, value, 
                              percentile_25, percentile_50, percentile_75, percentile_90)
            
            if column != 'LBDHDD':
                with col10:
                    fig, ax = plt.subplots(figsize=(15, 1))

                    percentile_25 = int(np.percentile(sorted_array, 25))
                    percentile_50 = int(np.percentile(sorted_array, 50))
                    percentile_75 = int(np.percentile(sorted_array, 75))
                    percentile_90 = int(np.percentile(sorted_array, 90))

                    # st.write(f"25: {percentile_25}, 50: {percentile_50}, 75: {percentile_75}, 90: {percentile_90}")

                    ax.plot([0, 100], [0.725, 0.725], color='grey', lw=20)

                    ax.set_xlim(0, 100)
                    ax.set_ylim(0.4, 1.1)
                    ax.set_yticks([]) 
                    tick_positions = [0, 25, 50, 75, 90, 99]

                    tick_labels = ['', '25th', '50th', '75th', '90th', '']
                    ax.set_xticks(tick_positions)
                    ax.set_xticklabels(tick_labels)

                    ax.set_xlabel('Percentile (%)')

                    for spine in ax.spines.values():
                        spine.set_visible(False)

                    ax.scatter(25, 0.85, color='grey', zorder=5, label='Your Input', s=500, edgecolors=['black'])
                    plt.annotate(f'{percentile_25}', xy=(25, 0.65), xytext=(25, 0.81),
                                    horizontalalignment='center', weight='bold', color='white', zorder=10)
                    ax.scatter(50, 0.85, color='grey', zorder=5, label='Your Input', s=500, edgecolors=['black'])
                    plt.annotate(f'{percentile_50}', xy=(50, 0.65), xytext=(50, 0.81),
                                    horizontalalignment='center', weight='bold', color='white', zorder=10)
                    ax.scatter(75, 0.85, color='grey', zorder=5, label='Your Input', s=500, edgecolors=['black'])
                    plt.annotate(f'{percentile_75}', xy=(75, 0.65), xytext=(75, 0.81),
                                    horizontalalignment='center', weight='bold', color='white', zorder=10)
                    ax.scatter(90, 0.85, color='grey', zorder=5, label='Your Input', s=500, edgecolors=['black'])
                    plt.annotate(f'{percentile_90}', xy=(90, 0.65), xytext=(90, 0.81),
                                    horizontalalignment='center', weight='bold', color='white', zorder=10)

                    st.pyplot(fig)
                    plt.close()

# Main execution
df_c = load_files(False)
df_d = ui_choose(df_c, False)
show_analysis(df_d)

st.write(f"#")
st.markdown('<div style="text-align: center"> Version 0.2 </div>', unsafe_allow_html=True)