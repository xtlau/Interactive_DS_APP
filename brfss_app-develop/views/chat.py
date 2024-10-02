import pandas as pd
import os
from langchain_experimental.agents import create_pandas_dataframe_agent
from apikey import apikey 
from dotenv import load_dotenv, find_dotenv
import streamlit as st
from langchain.agents.agent_types import AgentType
from langchain_community.chat_models import ChatOpenAI
from openai.error import RateLimitError, InvalidRequestError, AuthenticationError

#OpenAIKey
os.environ['OPENAI_API_KEY'] = 'sk-proj-1nvaiwu16AmXAVnhPTBHT3BlbkFJIDSwnWD40c0qLzuSJgmR'
load_dotenv(find_dotenv())
api_key = os.getenv('OPENAI_API_KEY')

# Verify that the API key is loaded
if not api_key:
    st.error("OpenAI API key not found. Please set the API key in your environment variables.")

def load_view(data):
    st.title("Let's Chat With AI Exploratory Data Analysis Assistant")

    st.markdown(
        """

    <p> This is an AI EDA Assistant utilized OpenAI API, please feel free to ask him about any EDA questions.🤖️</p>

    <p> Step 1: Select variables you would like to explore.</p>

    <p> Step 2: Select questions from the list below or select others to ask your own questions (e.g. ask him what can you do? tell me the correlations between these variables.)</p>

    <p> Let't get started! </p>
    """,

        unsafe_allow_html=True
    )

    # Step3: Select X and Y
    def input(data):
        selected_columns = st.multiselect("Select at least one variable you would like to explore", data.columns)
        df = data[selected_columns]

        return df


    # Generate LLM response
    def generate_response(df, input_query):
        try:
            llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.2)
            # Create Pandas DataFrame Agent
            agent = create_pandas_dataframe_agent(llm, df, verbose=True, agent_type=AgentType.OPENAI_FUNCTIONS, handle_parsing_errors=True, allow_dangerous_code=True)
            # Perform Query using the Agent
            response = agent.run(input_query)
            return st.write(response)
        except RateLimitError:
            st.error("Rate limit exceeded. Please try again later.")
        except InvalidRequestError:
            st.error("Invalid request made to OpenAI API.")
        except AuthenticationError:
            st.error("Authentication failed. Check API key.")
        except Exception as e:
            st.error(f"An error occurred: {e}")


    df = input(data)

    question_list = [
    'How many rows are there?',
    'How many columns are there',
    'Other']
    query_text = st.selectbox('Select an example query:', question_list, index=None, placeholder="Select a query...")

    if len(df.columns) > 0 and query_text:
            if query_text == 'Other':
                query_text = st.text_input('Enter your query:', placeholder = 'Enter query here ...')
            st.header('Output')
            generate_response(df, query_text)



