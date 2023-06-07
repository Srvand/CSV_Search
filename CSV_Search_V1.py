import streamlit as st 
import os
import pandas as pd
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI


openai_api_key = st.secrets['OPENAI_API_KEY']

st.set_page_config(page_title='Interact with CSV file using LLM',layout="wide")
st.write("""
    <style>
        footer {visibility: hidden;}
        body {
            font-family: Arial, sans-serif;
            font-size: 14px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Interact with CSV file using LLM")

input_csv = st.file_uploader("Upload your CSV file", type=['csv'])

if input_csv is not None:
    data=pd.read_csv(input_csv)
    st.dataframe(data)
    st.info("Enter your query")
    input_text = st.text_input(label='Enter your query',label_visibility="collapsed")
    if input_text is not None:
        if st.button("Generate"):
            with st.spinner("Generating response..."):
                agent = create_pandas_dataframe_agent(OpenAI(temperature=0),data, verbose=True) #LangChain LLM
                result=agent.run(input_text)
                if result:
                    st.write(f"<h1 style='font-size: 14px; color: #6495ED; font-family: Arial, sans-serif;text-align: left;'>Below answer generated from LLM without prompt(Straightaway answer)::</h1>", unsafe_allow_html=True)
                    st.write(result)
                prompt = (
                            """"
                                For user question,reply as follows:
                                Example:

                                {"answer": "The title with the highest rating in the file is 'Jack'"}
                               
                                If column name mentioned in the user question in not in the file, reply as follows:
                                "Cant find relevant column in the file"

                                If you do not know the answer, reply as follows:
                                "Cant find relevant info in the file"

                                Return all output as a string without dict format
                                Below is the query.
                                Query: 
                                """
                            + input_text
                        )
                prompt_result=agent.run(prompt)#LangChain LLM wiht prompt
                if prompt_result:
                    st.write(f"<h1 style='font-size: 14px; color: #6495ED; font-family: Arial, sans-serif;text-align: left;'>Below answer generated from LLM with refined prompt::</h1>", unsafe_allow_html=True)
                    st.write(prompt_result)

                  
                        
