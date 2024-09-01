import pandas as pd
import os

DIR = os.getcwd()
LOGO_DIR = DIR + '/logo/'
DATA_DIR = DIR + '/data/'
SAHC_DATA_DIR = DIR + '/sahc_data/'
VERSION = 2.0

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
USER_FILE_2017 = os.path.join(DATA_DIR, 'DEMO_2017.XPT')
DIQ_FILE_2017 = os.path.join(DATA_DIR, 'DIQ_2017.XPT')
BPQ_FILE_2017 = os.path.join(DATA_DIR, 'BPQ_2017.XPT')
HDL_FILE_2017 = os.path.join(DATA_DIR, 'HDL_2017.XPT')
TGL_FILE_2017 = os.path.join(DATA_DIR, 'TRIGLY_2017.XPT')
TCH_FILE_2017 = os.path.join(DATA_DIR, 'TCHOL_2017.XPT')
GLU_FILE_2017 = os.path.join(DATA_DIR, 'GLU_2017.XPT')
GHB_FILE_2017 = os.path.join(DATA_DIR, 'GHB_2017.XPT')
BPX_FILE_2017 = os.path.join(DATA_DIR, 'BPXO_2017.XPT')
CBC_FILE_2017 = os.path.join(DATA_DIR, 'CBC_2017.XPT')
BMX_FILE_2017 = os.path.join(DATA_DIR, 'BMX_2017.XPT')
MCQ_FILE_2017 = os.path.join(DATA_DIR, 'MCQ_2017.XPT')
USER_FILE_2015 = os.path.join(DATA_DIR, 'DEMO_2015.XPT')
DIQ_FILE_2015 = os.path.join(DATA_DIR, 'DIQ_2015.XPT')
BPQ_FILE_2015 = os.path.join(DATA_DIR, 'BPQ_2015.XPT')
HDL_FILE_2015 = os.path.join(DATA_DIR, 'HDL_2015.XPT')
TGL_FILE_2015 = os.path.join(DATA_DIR, 'TRIGLY_2015.XPT')
TCH_FILE_2015 = os.path.join(DATA_DIR, 'TCHOL_2015.XPT')
GLU_FILE_2015 = os.path.join(DATA_DIR, 'GLU_2015.XPT')
GHB_FILE_2015 = os.path.join(DATA_DIR, 'GHB_2015.XPT')
BPX_FILE_2015 = os.path.join(DATA_DIR, 'BPXO_2015.XPT')
BMX_FILE_2015 = os.path.join(DATA_DIR, 'BMX_2015.XPT')
MCQ_FILE_2015 = os.path.join(DATA_DIR, 'MCQ_2015.XPT')
USER_FILE_2013 = os.path.join(DATA_DIR, 'DEMO_2013.XPT')
DIQ_FILE_2013 = os.path.join(DATA_DIR, 'DIQ_2013.XPT')
BPQ_FILE_2013 = os.path.join(DATA_DIR, 'BPQ_2013.XPT')
HDL_FILE_2013 = os.path.join(DATA_DIR, 'HDL_2013.XPT')
TGL_FILE_2013 = os.path.join(DATA_DIR, 'TRIGLY_2013.XPT')
TCH_FILE_2013 = os.path.join(DATA_DIR, 'TCHOL_2013.XPT')
GLU_FILE_2013 = os.path.join(DATA_DIR, 'GLU_2013.XPT')
GHB_FILE_2013 = os.path.join(DATA_DIR, 'GHB_2013.XPT')
BPX_FILE_2013 = os.path.join(DATA_DIR, 'BPXO_2013.XPT')
BMX_FILE_2013 = os.path.join(DATA_DIR, 'BMX_2013.XPT')
MCQ_FILE_2013 = os.path.join(DATA_DIR, 'MCQ_2013.XPT')

SAHC_FILE = os.path.join(SAHC_DATA_DIR, 'merged_data_noPID.csv')
SAHC_RENAME_FILE = os.path.join(SAHC_DATA_DIR, 'renamed_merged_data_noPID.csv')

NHANES_FILE = os.path.join(DATA_DIR, 'nhanes_merged_data.csv')

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
    
def export_nhanes_files():

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
    df_user_2017 = pd.read_sas(USER_FILE_2017, format='xport')
    df_diq_2017 = pd.read_sas(DIQ_FILE_2017, format='xport')
    df_bpq_2017 = pd.read_sas(BPQ_FILE_2017, format='xport')
    df_hdl_2017 = pd.read_sas(HDL_FILE_2017, format='xport')
    df_tgl_2017 = pd.read_sas(TGL_FILE_2017, format='xport')
    df_tch_2017 = pd.read_sas(TCH_FILE_2017, format='xport')
    df_glu_2017 = pd.read_sas(GLU_FILE_2017, format='xport')
    df_ghb_2017 = pd.read_sas(GHB_FILE_2017, format='xport')
    df_bpx_2017 = pd.read_sas(BPX_FILE_2017, format='xport')
    df_bmx_2017 = pd.read_sas(BMX_FILE_2017, format='xport')
    df_mcq_2017 = pd.read_sas(MCQ_FILE_2017, format='xport')
    df_user_2015 = pd.read_sas(USER_FILE_2015, format='xport')
    df_diq_2015 = pd.read_sas(DIQ_FILE_2015, format='xport')
    df_bpq_2015 = pd.read_sas(BPQ_FILE_2015, format='xport')
    df_hdl_2015 = pd.read_sas(HDL_FILE_2015, format='xport')
    df_tgl_2015 = pd.read_sas(TGL_FILE_2015, format='xport')
    df_tch_2015 = pd.read_sas(TCH_FILE_2015, format='xport')
    df_glu_2015 = pd.read_sas(GLU_FILE_2015, format='xport')
    df_ghb_2015 = pd.read_sas(GHB_FILE_2015, format='xport')
    df_bpx_2015 = pd.read_sas(BPX_FILE_2015, format='xport')
    df_bmx_2015 = pd.read_sas(BMX_FILE_2015, format='xport')
    df_mcq_2015 = pd.read_sas(MCQ_FILE_2015, format='xport')
    df_user_2013 = pd.read_sas(USER_FILE_2013, format='xport')
    df_diq_2013 = pd.read_sas(DIQ_FILE_2013, format='xport')
    df_bpq_2013 = pd.read_sas(BPQ_FILE_2013, format='xport')
    df_hdl_2013 = pd.read_sas(HDL_FILE_2013, format='xport')
    df_tgl_2013 = pd.read_sas(TGL_FILE_2013, format='xport')
    df_tch_2013 = pd.read_sas(TCH_FILE_2013, format='xport')
    df_glu_2013 = pd.read_sas(GLU_FILE_2013, format='xport')
    df_ghb_2013 = pd.read_sas(GHB_FILE_2013, format='xport')
    df_bpx_2013 = pd.read_sas(BPX_FILE_2013, format='xport')
    df_bmx_2013 = pd.read_sas(BMX_FILE_2013, format='xport')
    df_mcq_2013 = pd.read_sas(MCQ_FILE_2013, format='xport')

    df_user = pd.concat([df_user_2013, df_user_2015, df_user_2017], ignore_index=True)
    df_combined = df_user[['SEQN', 'RIAGENDR', 'RIDAGEYR', 'RIDRETH3']]

    df_diq = pd.concat([df_diq_2013, df_diq_2015, df_diq_2017], ignore_index=True)
    df_combined = pd.merge(df_combined, df_diq[['SEQN', 'DIQ010', 'DIQ160', 'DIQ050', 'DIQ070']], on='SEQN', how='left')

    df_bpq = pd.concat([df_bpq_2013, df_bpq_2015, df_bpq_2017], ignore_index=True)
    df_combined = pd.merge(df_combined, df_bpq[['SEQN', 'BPQ090D', 'BPQ100D', 'BPQ040A', 'BPQ020']], on='SEQN', how='left')

    df_hdl = pd.concat([df_hdl_2013, df_hdl_2015, df_hdl_2017], ignore_index=True)
    df_combined = pd.merge(df_combined, df_hdl[['SEQN', 'LBDHDD']], on='SEQN', how='left')

    df_tgl = pd.concat([df_tgl_2013, df_tgl_2015, df_tgl_2017], ignore_index=True)
    df_combined = pd.merge(df_combined, df_tgl[['SEQN', 'LBXTR', 'LBDLDL']], on='SEQN', how='left')

    df_tch = pd.concat([df_tch_2013, df_tch_2015, df_tch_2017], ignore_index=True)
    df_combined = pd.merge(df_combined, df_tch[['SEQN', 'LBXTC']], on='SEQN', how='left')

    df_glu = pd.concat([df_glu_2013, df_glu_2015, df_glu_2017], ignore_index=True)
    df_combined = pd.merge(df_combined, df_glu[['SEQN', 'LBXGLU']], on='SEQN', how='left')

    df_ghb = pd.concat([df_ghb_2013, df_ghb_2015, df_ghb_2017], ignore_index=True)
    df_combined = pd.merge(df_combined, df_ghb[['SEQN', 'LBXGH']], on='SEQN', how='left')

    df_bpx = pd.concat([df_bpx_2013, df_bpx_2015, df_bpx_2017], ignore_index=True)
    df_combined = pd.merge(df_combined, df_bpx[['SEQN', 'BPXOSY1', 'BPXODI1', 'BPXOPLS1', 'BPXOSY2', 'BPXODI2', 'BPXOPLS2', 'BPXOSY3', 'BPXODI3', 'BPXOPLS3']], on='SEQN', how='left')

    df_bmx = pd.concat([df_bmx_2013, df_bmx_2015, df_bmx_2017], ignore_index=True)
    df_combined = pd.merge(df_combined, df_bmx[['SEQN', 'BMXBMI']], on='SEQN', how='left')

    df_mcq = pd.concat([df_mcq_2013, df_mcq_2015, df_mcq_2017], ignore_index=True)
    df_combined = pd.merge(df_combined, df_mcq[['SEQN', 'MCQ160E']], on='SEQN', how='left')

    df_combined['Age_Group'] = df_combined['RIDAGEYR'].apply(map_age_to_group)
    df_combined['TotHDLRat'] =  df_combined['LBXTC'] / df_combined['LBDHDD']

    df_combined.to_csv(NHANES_FILE, index=False)

def export_new_sahc():
    df_combined = pd.read_csv(SAHC_FILE)

    # Rename the SAHC columns to match with the NHANES columns
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
    df_combined.to_csv(SAHC_RENAME_FILE, index=False)

export_nhanes_files()
# export_new_sahc()