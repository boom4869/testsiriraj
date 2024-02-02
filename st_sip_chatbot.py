# -*- coding: utf-8 -*-
import openai
import streamlit as st
from llama_index import (
    KeywordTableIndex,
    SimpleDirectoryReader,
    ServiceContext,
)
from llama_index import VectorStoreIndex,download_loader
from llama_index.output_parsers import LangchainOutputParser
from llama_index.llms import OpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from pathlib import Path

openai.api_key = st.secrets.openai.api_key
st.set_page_config(page_title="Chatbot for doctor appointment", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("Chatbot for doctor appointment")
st.info("แชทบอทช่วยตอบคำถามสำหรับการนัดหมายแพทย์ที่โรงพยาบาลศิริราช ปิยมหาราชการุณย์ ดูข้อมูลแพทย์เพิ่มเติมได้ที่ (https://www.siphhospital.com/th/medical-services/find-doctor)", icon="📃")

if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "สอบถามข้อมูลในการนัดหมายแพทย์ได้เลยครับ"}
    ]
@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
        JSONReader = download_loader("JSONReader")
        loader = JSONReader()
        system_prompt = """ You are an information assistant at the siriraj hospital. Your job is to answer the questions about doctor schedule and expertise in Thai.\
        Your answers must based only on the file in the folder provided. Do not hallucinate the answer and if you don't find the answer, don't answer the wrong information """
        docs = loader.load_data(Path('./data/siriraj_doctor_details.jsonl'), is_jsonl=True)
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt= system_prompt ))
        index = VectorStoreIndex.from_documents(docs,service_context=service_context)

        return index

index = load_data()

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
    st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history
