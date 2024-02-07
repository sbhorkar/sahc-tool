import streamlit as st
import pandas as pd

import os

# Given two dataframes with users identified by SEQN, compute how many common users they have
def common_users(df1, df2):
    return len(set(df1['SEQN']).intersection(set(df2['SEQN'])))

pd.set_option("styler.render.max_elements", 1468657)
# Demographics file
dir=os.getcwd()
nhanes_files={
    "2013-2014":"DEMO_H.XPT",
    "2015-2016":"DEMO_I.XPT",
    "2017-2018":"DEMO_J.XPT",
    "2017-March 2020 Pre-Pandemic":"DEMO_P.XPT",
}

nhlist = list(nhanes_files[i] for i in nhanes_files.keys())
dfdict={}
for i in nhanes_files.keys():
    fname=nhanes_files[i]
    dfdict[i] = pd.read_sas(dir+'/data/'+fname, format='xport')

for i in nhanes_files.keys():
    st.write(f"### Dataset: {i} has {dfdict[i].shape[0]} records")
    st.dataframe(dfdict[i])

for i in nhanes_files.keys():
    for j in nhanes_files.keys():
        if i != j:
            st.write(f"{common_users(dfdict[i],dfdict[j])} records in common between {i} and {j}")



a= """

file_selection = st.selectbox(
   "Select NHANES Data Year",
   list(nhanes_files.keys()),
)

f_demo=nhanes_files[file_selection]
print(f"File selection {file_selection} DEMO {f_demo}")
df_demo=pd.read_sas(dir+'/data/'+f_demo, format='xport')
"""