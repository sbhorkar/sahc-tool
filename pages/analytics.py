import streamlit as st
import sqlite3
import pandas as pd
import os
from datetime import datetime

DB_PATH = "analytics.db"

# --- Password prompt ---
st.title("🔒 Analytics Export")

if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False

if not st.session_state.auth_ok:
    password = st.text_input("Enter admin password", type="password")
    if st.button("Log in"):
        if password == st.secrets.get("ADMIN_PASSWORD"):
            st.session_state.auth_ok = True
            st.rerun()
        else:
            st.error("Incorrect password.")
else:
    st.success("Authenticated")

    if os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            df = pd.read_sql_query("SELECT * FROM analytics", conn)
        finally:
            conn.close()

        today_str = datetime.now().strftime("%Y-%m-%d")
        csv_data = df.to_csv(index=False)

        st.download_button(
            label="📄 Download analytics as CSV",
            data=csv_data,
            file_name=f"analytics_{today_str}.csv",
            mime="text/csv",
        )
    else:
        st.warning("analytics.db not found.")