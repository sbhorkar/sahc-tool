import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os

def common_users(df1, df2):
    return len(set(df1['SEQN']).intersection(set(df2['SEQN'])))

def create_matrix(df):
    matrix = pd.DataFrame(index=df.keys(), columns=df.keys())
    for i in df.keys():
        for j in df.keys():
            matrix[i][j] = common_users(df[i], df[j])
    matrix = matrix.apply(pd.to_numeric)
    return matrix

dir=os.getcwd()
nhanes_files={
    "2013-2014":"DEMO_H.XPT",
    "2015-2016":"DEMO_I.XPT",
    "2017-2018":"DEMO_J.XPT",
    "2017-March 2020 Pre-Pandemic":"DEMO_P.XPT",
}

dfdict={}
for i in nhanes_files.keys():
    fname=nhanes_files[i]
    dfdict[i] = pd.read_sas(dir+'/data/'+fname, format='xport')

m=create_matrix(dfdict)
print(f"Matrix {m}")
st.write("### Common Users Matrix")
# Plotting the matrix

fig, ax = plt.subplots(figsize=(10, 10))  # Explicitly create a figure and an axes object
sns.heatmap(m, annot=True, cmap='coolwarm', fmt='g', ax=ax)  # Plot on the created axes object
st.pyplot(fig)  # Pass the figure to st.pyplot() to display it
