import streamlit as st
import pandas as pd

def load_view():
    sections = {
        "Welcome": welcome,
        "About This App": about_this_app,
        "The Behavioral Risk Factor Surveillance System (BRFSS)": brfss,
    }
    section = st.sidebar.radio("Sections:", tuple(sections.keys()))
    sections[section]()
    
def welcome():
    st.title("Welcome to Health Exploration and Analytical Research Tool (HEART) for BFRSS 2022.")
    st.markdown("""
    <p>Dive into the extensive 2022 Behavioral Risk Factor Surveillance System (BFRSS) survey data 
                through our interactive app. This powerful tool offers unprecedented access to a 
                wealth of health-related data across all 50 states, the District of Columbia, and 
                U.S. territories. Utilize dynamic visualizations and custom analysis features to 
                uncover insights into public health trends, risk behaviors, and preventive measures 
                that shape the well-being of the U.S. adult population.</p>

    <p>Start your journey by navigating through the app using the top left menu. Discover patterns, 
                generate reports, and contribute to impactful health research and policy-making 
                efforts today!</p>
                """, unsafe_allow_html=True)

def about_this_app():
    st.header("About This App")

    st.markdown("""
    <p>HEART utilizes <a href=https://www.cdc.gov/brfss/annual_data/annual_2022.html>BFRSS 2022 Survey Data</a>, 
    it incorporates Exploratory Data Analysis, Machine Learning Modeling, and AI Exploratory Data Analysis Assistant functions. These functions enable users to conduct clinical data science exploration without coding themselves. 
    </p>""", unsafe_allow_html=True)

def brfss():
    st.header("The Behavioral Risk Factor Surveillance System (BRFSS)")

    st.markdown("""
    <p>The Behavioral Risk Factor Surveillance System (BRFSS) is a collaborative health survey 
                initiative between all U.S. states, territories, and the CDC. Starting in 1984, 
                the BRFSS aims to gather uniform state-specific data on various health-related 
                aspects including risk behaviors, chronic diseases, healthcare access, and 
                preventive services. This data helps identify health patterns and supports public 
                health decisions and legislation.</p>

    <p>Annually, the BRFSS conducts health-related telephone surveys targeting noninstitutionalized 
                adults in the U.S. and its territories, using both landline and cellular phones. 
                The survey includes core questions on health status, chronic conditions, tobacco 
                use, and more, along with optional modules on topics like social determinants of 
                health and sexual orientation/gender identity.</p>

    <p>State health departments manage the field operations of the BRFSS with CDC's support, 
                ensuring adherence to protocols. Data collected is sent to the CDC for processing 
                and analysis, and each state receives an edited and weighted data file for use in 
                health programs and policymaking. This extensive data collection and analysis by 
                the BRFSS are crucial for tackling health disparities and enhancing community health 
                outcomes across the states.</p>
    """, unsafe_allow_html=True)