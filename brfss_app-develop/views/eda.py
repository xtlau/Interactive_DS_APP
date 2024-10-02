
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from ydata_profiling import ProfileReport
from streamlit_ydata_profiling import st_profile_report
import plotly.express as px
import sweetviz as sv

def load_view(data):
    sections = {
        "Exploratory Data Analysis": data_analysis,
        "For the missing data": missing_data,
        "About data weighting": data_weighting,
    }
    section = st.sidebar.radio("Sections:", tuple(sections.keys()))
    if section == "Exploratory Data Analysis":
        sections[section](data)
    else:
        sections[section]()

def data_analysis(data):
    st.title("Exploratory Data Analysis")

    st.markdown("""
    <p>There are four tabs within the exploratory data analysis: 'Data Overview', 
                'Data Visualizations', 'Profiling Report', and 'Sweetviz Report'. 
                Please select a method of exploratory data analysis to delve into the data.</p>
                """, unsafe_allow_html=True)
    eda(data)

def missing_data():
    st.header("For the missing data")

    st.markdown("""
    <p>In the BRFSS 2022 Survey Data, missing values are encoded as -1 for simplicity. 
                Non-responses are categorized distinctly with codes 9, 99, or 999, 
                indicating either refusal or missing responses due to incomplete 
                interviews or skipped questions. Users must carefully differentiate 
                these codes to accurately distinguish between instances where questions 
                were not presented to respondents and those where questions were refused 
                or unanswered after presentation. This coding system requires careful 
                interpretation to ensure the reliability and accuracy of data analysis. 
                For more information, please find the 2022 BRFSS Overview CDC on 
                <a href=https://www.cdc.gov/brfss/annual_data/annual_2022.html>2022 BRFSS Survey Data and Documentation</a></p>""", unsafe_allow_html=True)

def data_weighting():
    st.header("About data weighting")

    st.markdown("""
    <p>The BRFSS employs a weighting methodology to ensure its survey sample accurately represents 
                the U.S. adult population. This methodology includes design weights and demographic 
                adjustments through raking. Design weights adjust for the probability of selection, 
                nonresponse bias, and coverage errors, using variables like geographic stratum weight 
                (_STRWT), number of phones in a household (NUMPHON3), and number of adults (NUMADULT). 
                Stratum weight (_STRWT) is calculated based on the number of records in geographic 
                and density strata, accounting for differences in selection probabilities among these 
                strata, which are defined by area codes, prefixes, and regions like counties or census 
                tracts. For more information, please find the Calculated Variables in Data Files CDC 
                on <a href=https://www.cdc.gov/brfss/annual_data/annual_2022.html>2022 BRFSS Survey Data and Documentation</a></p>""", unsafe_allow_html=True)

def eda(data):
    select_options = ["Please select an option...","Data Overview", "Data Visualization", "Profiling Report", "Sweetviz Report"]
    selected_option = st.selectbox("Choose a way for data exploratory", select_options)

    if selected_option == "Please select an option...":
        st.write("Now let's begin our exploratory data analysis!🌍")
    elif selected_option == "Data Overview":
        st.header("Data Overview")
        st.write("First 20 rows of the data:")
        st.write(data.head(20))
        st.write("Data Description:")
        st.write(data.describe())
    elif selected_option == "Data Visualization":
        st.header("Data Visualizations")
        plot_options = ["Please select a type of data visualization", "Bar plot", "Scatter plot", "Histogram", "Box plot"]
        selected_plot = st.selectbox("Please select a plot type", plot_options)
        options = ['Select a column'] + list(data.columns)
        if selected_plot == "Bar plot":
            st.subheader("Bar Plot Configuration")
            x_axis = st.selectbox("Select x-axis", options, index=0)
            y_axis = st.selectbox("Select y-axis", options, index=0)
            if x_axis != 'Select a column' and y_axis != 'Select a column':
                bar_fig = px.bar(data, x=x_axis, y=y_axis)
                st.plotly_chart(bar_fig)
            else:
                st.write("Please select both x and y axes to generate a bar plot.")
        elif selected_plot == "Scatter plot":
            st.subheader("Scatter Plot Configuration")
            x_axis = st.selectbox("Select x-axis", options, index=0)
            y_axis = st.selectbox("Select y-axis", options, index=0)
            if x_axis != 'Select a column' and y_axis != 'Select a column':
                scatter_fig = px.scatter(data, x=x_axis, y=y_axis)
                scatter_fig.update_traces(marker=dict(size=5))
                st.plotly_chart(scatter_fig)
            else:
                st.write("Please select both x and y axes to generate a scatter plot.")
        elif selected_plot == "Histogram":
            st.subheader("Histogram Configuration")
            column = st.selectbox("Select a column", options, index=0)
            if column != 'Select a column':
                bins = st.slider("Number of bins", 5, 100, 20)
                fig, ax = plt.subplots()
                sns.histplot(data[column], bins=bins, ax=ax)
                ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))
                ax.set_xticklabels([label.get_text() for label in ax.get_xticklabels()], rotation=90, ha="right")
                st.pyplot(fig)
            else:
                st.write("Please select a column to generate a histogram.")
        elif selected_plot == "Box plot":
            st.subheader("Box Plot Configuration")
            column = st.selectbox("Select a column", options, index=0)
            if column != 'Select a column':
                fig, ax = plt.subplots()
                sns.boxplot(x=data[column], ax=ax)
                ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right") 
                st.pyplot(fig)
            else:
                st.write("Please select a column to generate a box plot.")
    elif selected_option == "Profiling Report":
        selected_columns = st.multiselect("Select columns for profiling report", data.columns)
        if selected_columns:
            pr = ProfileReport(data[selected_columns], explorative=True)
            st.header('**Profiling Report**')
            st_profile_report(pr)
            # pr.to_file('profile_report.html')
        else:
            st.warning("Please select at least one column for the profiling report.")
    elif selected_option == "Sweetviz Report":
        selected_columns = st.multiselect("Select columns for Sweetviz report", data.columns)
        if selected_columns:
            sweet_report = sv.analyze(data[selected_columns])
            st.header('**Sweetviz Report**')
            st.write(sweet_report.show_html())
        else:
            st.warning("Please select at least one column for the Sweetviz report.")