import streamlit as st
import pandas as pd

@st.cache_data
def load_data():
    # Reading from dropbox
    # file_url = 'https://www.dropbox.com/scl/fi/nunlhwxpypl1bcrnqxawq/brfss_cleaned.csv?rlkey=dlgo87ror308ao18hp9rg937k&st=3anc9tif&dl=1'
    # data = pd.read_csv(file_url)

    data = pd.read_parquet('./data/brfss_diabete_df.parquet')
    return data