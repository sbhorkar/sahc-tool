import streamlit as st
import pandas as pd
import os

DIR=os.getcwd()
DATA_DIR=DIR+'/data/'
USER_FILE=os.path.join(DATA_DIR, 'DEMO_P.XPT')
DIQ_FILE=os.path.join(DATA_DIR, 'P_DIQ.XPT')
BPQ_FILE=os.path.join(DATA_DIR, 'P_BPQ.XPT')
HDL_FILE=os.path.join(DATA_DIR, 'P_HDL.XPT')
TGL_FILE=os.path.join(DATA_DIR, 'P_TRIGLY.XPT')
TCH_FILE=os.path.join(DATA_DIR, 'P_TCHOL.XPT')
GLU_FILE=os.path.join(DATA_DIR, 'P_GLU.XPT')
GHB_FILE=os.path.join(DATA_DIR, 'P_GHB.XPT')
BPX_FILE=os.path.join(DATA_DIR, 'P_BPXO.XPT')


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
    #
    df_combined = df_user[['SEQN','RIAGENDR','RIDAGEYR']]
    df_combined = pd.merge(df_combined, df_diq[['SEQN', 'DIQ010', 'DIQ160','DIQ050','DIQ070']], on='SEQN', how='left')
    df_combined = pd.merge(df_combined, df_bpq[['SEQN', 'BPQ090D', 'BPQ100D','BPQ040A','BPQ050A']], on='SEQN', how='left')
    df_combined = pd.merge(df_combined, df_hdl[['SEQN', 'LBDHDD']], on='SEQN', how='left')
    df_combined = pd.merge(df_combined, df_tgl[['SEQN', 'LBXTR','LBDLDL']], on='SEQN', how='left')
    df_combined = pd.merge(df_combined, df_tch[['SEQN', 'LBXTC']], on='SEQN', how='left')
    df_combined = pd.merge(df_combined, df_glu[['SEQN', 'LBXGLU']], on='SEQN', how='left')
    df_combined = pd.merge(df_combined, df_ghb[['SEQN', 'LBXGH']], on='SEQN', how='left')
    df_combined = pd.merge(df_combined, df_bpx[['SEQN', 'BPXOSY1', 'BPXODI1', 'BPXOPLS1','BPXOSY2', 'BPXODI2', 'BPXOPLS2','BPXOSY3', 'BPXODI3', 'BPXOPLS3' ]], on='SEQN', how='left')
    #
    st.dataframe(df_combined,hide_index=True)
    if debugging:
        st.dataframe(df_user,hide_index=True)
        st.dataframe(df_diq,hide_index=True)
        st.dataframe(df_bpq,hide_index=True)
        st.dataframe(df_hdl,hide_index=True)
        st.dataframe(df_tgl,hide_index=True)
        st.dataframe(df_tch,hide_index=True)
        st.dataframe(df_glu,hide_index=True)
        st.dataframe(df_ghb,hide_index=True)
        st.dataframe(df_bpx,hide_index=True)
    df_combined.to_csv(os.path.join(DATA_DIR, 'nhanes_combined.csv'),index=False)
    return df_combined;


df_c=load_files(True);