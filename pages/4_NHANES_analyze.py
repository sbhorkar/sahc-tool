import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

DIR=os.getcwd()
DATA_DIR=DIR+'/data/'
OUTPUT_DIR=DIR+'/output/'

SELECT_LIST={'GP_Age':"Age",'GP_Gender':"Gender",'GP_StatusDiab':"Status: Diabetes",'GP_StatusPreDiab':"Status: Pre-diabetes",
             'GP_StatusCAD1':"Told you had CAD",'GP_StatusCAD2':"Told you had Angina",'GP_StatusCAD3':"TOld you had heart attack",
             'GP_MedChol1':"Told to take prescription for cholesterol",'GP_MedChol2':"Now taking cholesterol medication",
             'GP_MedDiab1':"Told to take prescription for diabetes",'GP_MedDiab2':"Now taking diabetes medication",
             'GP_MedBP1':"Told to take prescription for BP",'GP_MedBP2':"Now taking BP medication"}
METRICS_MAP = {'LBXTR': 'Triglycerides', 'LBDHDD': 'HDL', 'LBDLDL': 'LDL', 'LBXTC': 'Total Cholesterol', 'LBXGLU': 'Fasting Glucose', 'LBXGH': 'Glycohemoglobin',
            'BPXOSY1': 'Systolic', 'BPXODI1': 'Diastolic', 'BPXOPLS1': 'Pulse'}

# Temp disable caching @st.cache_data
def load_files(debugging):
    df=pd.read_csv(OUTPUT_DIR+'nhanes_augmented.csv')
    if debugging:
        st.dataframe(df,hide_index=True)
    values_list={}
    for i in SELECT_LIST.keys():
        values_list[i]=df[i].unique()
    if debugging:
        st.write(f"Values list {values_list}")
    return df, values_list

def ui_choose(df, vl, debugging):
    selected_options={}
    for i in SELECT_LIST.keys():
        selected_options[i] = st.sidebar.multiselect(SELECT_LIST[i],vl[i], key=SELECT_LIST[i],default=vl[i])
        if debugging:
            st.write(f"For i {i}, SELECT LIST {SELECT_LIST[i]}, vl {vl[i]}, selected options {selected_options[i]}")
    
    if debugging:
        st.write(f"Selected options {selected_options}")

    for s in selected_options.keys():
        v=selected_options[s]
        df=df[df[s].isin(v)]    

    return df

def show_analysis(df):
    st.write("# Raw Data")
    st.dataframe(df, hide_index=True)
    st.write("# Summary Statistics")
    st.write(df.describe())
    st.write("# Data Distribution")
    

    for column in METRICS_MAP.keys():
        columnName=METRICS_MAP[column]
        plt.figure(figsize=(10, 6))
        df[column].plot(kind='box')
        plt.title(f'{columnName}')
        plt.xlabel('Values')
        plt.ylabel(columnName)
        stats = df[column].describe()
        Q1 = stats["25%"]
        Q3 = stats["75%"]
        IQR = Q3 - Q1
        lower_whisker = Q1 - 1.5 * IQR
        upper_whisker = Q3 + 1.5 * IQR
        lower_whisker_val = df[column][df[column] >= lower_whisker].min()
        upper_whisker_val = df[column][df[column] <= upper_whisker].max()
    
    
        # Annotating the plot with statistical values
        plt.annotate(f'Median: {stats["50%"]:.2f}', xy=(1, stats["50%"]), xytext=(1.1, stats["50%"]),
                 arrowprops=dict(facecolor='blue', shrink=0.05), horizontalalignment='left')
        plt.annotate(f'25th percentile: {stats["25%"]:.2f}', xy=(1, stats["25%"]), xytext=(1.1, stats["25%"]),
                 arrowprops=dict(facecolor='green', shrink=0.05), horizontalalignment='left')
        plt.annotate(f'75th percentile: {stats["75%"]:.2f}', xy=(1, stats["75%"]), xytext=(1.1, stats["75%"]),
                 arrowprops=dict(facecolor='red', shrink=0.05), horizontalalignment='left')
        plt.annotate(f'Lower Whisker: {lower_whisker_val:.2f}', xy=(1, lower_whisker_val), xytext=(1.1, lower_whisker_val - 0.5 * IQR),
                 arrowprops=dict(facecolor='orange', shrink=0.05), horizontalalignment='left')
        plt.annotate(f'Upper Whisker: {upper_whisker_val:.2f}', xy=(1, upper_whisker_val), xytext=(1.1, upper_whisker_val + 0.5 * IQR),
                 arrowprops=dict(facecolor='purple', shrink=0.05), horizontalalignment='left')
    
    
        st.pyplot(plt.gcf())  # Show the plot in Streamlit

    plt.figure(figsize=(10, 6))
    # Add code for scatter plot
    st.pyplot(plt.figure())  # Show the plot in a separate window

    plt.figure(figsize=(10, 6))
    # Add code for histogram
    st.pyplot(plt.figure())  # Show the plot in a separate window

debugging=False
df_c,vl=load_files(debugging)
df_d=ui_choose(df_c,vl,debugging)
show_analysis(df_d)