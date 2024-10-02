
import streamlit as st
import pandas as pd
from views import eda, home, modelling, chat
import utils as utl
import data_loader as data_loader

st.set_page_config(
    page_title="HEART - Health Exploration and Analytical Research Tool"
)

# Button for uploading your own dataset
# if st.button("Upload your own dataset"):
#     uploaded_files = st.file_uploader(
#         "Upload your dataset", accept_multiple_files=True
#     )
#     if uploaded_files:
#         # Process the uploaded files here
#         data = data_loader.load_uploaded_data(uploaded_files)

data = data_loader.load_data()

utl.inject_custom_css()
utl.navbar_component()

def navigation():
    route = utl.get_current_route()
    print(route)
    if route == None or route == "home":
        home.load_view()
    elif route == "eda":
        eda.load_view(data)
    elif route == "modelling":
        modelling.load_view(data)
    elif route == "chat":
        chat.load_view(data)

navigation()