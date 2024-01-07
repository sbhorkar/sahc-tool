import streamlit as st
import pandas as pd

import os

pd.set_option("styler.render.max_elements", 1468657)
# Demographics file
dir=os.getcwd()
nhanes_files={
    "2013-2014":"DEMO_H.XPT",
    "2015-2016":"DEMO_I.XPT",
    "2017-2018":"DEMO_J.XPT",
    "2017-March 2020 Pre-Pandemic":"DEMO_P.XPT",
}
file_selection = st.selectbox(
   "Select NHANES Data Year",
   list(nhanes_files.keys()),
)

f_demo=nhanes_files[file_selection]
print(f"File selection {file_selection} DEMO {f_demo}")
df_demo=pd.read_sas(dir+'/data/'+f_demo, format='xport')
print(f"DF shape demographics: {df_demo.shape}")
with st.expander(f"# Raw Data : Demographics {df_demo.shape}"):
    st.dataframe(df_demo.style.format(lambda x: f"{x:.0f}"),hide_index=True)
    st.download_button(label="Download CSV",data=df_demo.to_csv(index=False),file_name='demo.csv',mime='text/csv')
df_demo_pro=df_demo
# Select specific demographics
nhanes_demos={
    "Mexican American":1,
    "Other Hispanic":2,
    "Non-Hispanic White":3,
    "Non-Hispanic Black":4,
    "Non-Hispanic Asians":6,
    "Other Race - Including Multi-Racial":7,
}
demo_selection = st.selectbox(
   "Select demographics subset",
   list(nhanes_demos.keys()),
)

demo_id=nhanes_demos[demo_selection]
df_demo_pro=df_demo_pro[df_demo_pro['RIDRETH3']==demo_id]
df_demo_pro.to_csv(dir+'/output/'+f_demo[:-4]+'_pro.csv',index=False)
with st.expander(f"# {demo_selection} : Demographics {df_demo_pro.shape}"):
    st.dataframe(df_demo_pro.style.format(lambda x: f"{x:.0f}"),hide_index=True)
    st.download_button(label="Download CSV",data=df_demo_pro.to_csv(index=False),file_name='demo_pro.csv',mime='text/csv')

to_skip="""
#Questionnaire file
f_diq='nhanes_2020_P_DIQ_output_file.csv' 
df_diq=pd.read_csv(dir+'/data/nhanes/'+f_diq)
print(f"DF shape questionairre: {df_diq.shape}")
with st.expander(f"# Raw Data : Questionnaire {df_diq.shape}"):
    st.dataframe(df_diq,hide_index=True)
df_diq_pro=df_diq
# Select Asians
df_diq_pro = df_diq_pro[df_diq_pro['SEQN'].isin(df_demo_pro['SEQN'])]
df_diq_pro.to_csv(dir+'/data/nhanes/'+f_diq[:-4]+'_pro.csv',index=False)
with st.expander(f"# Asians Only : Questionairre {df_diq_pro.shape}"):
    st.dataframe(df_diq_pro,hide_index=True)
    st.download_button(label="Download CSV",data=df_diq_pro.to_csv(index=False),file_name='diq_pro.csv',mime='text/csv')
"""