import json
import os
import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
from functools import lru_cache

load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

with open("config.json", "r") as f:
    config = json.load(f)

with open("conversation_stages.json", "r") as f:
    conversation_stages = json.load(f)

def setup_logger(log_file):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    rotating_handler = RotatingFileHandler(log_file, maxBytes=1024*1024, backupCount=5)
    rotating_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    rotating_handler.setFormatter(formatter)
    logger.addHandler(rotating_handler)
    return logger

log_file = 'Sales_Agent_logger.log'
logger = setup_logger(log_file)

# Use a faster model and reduce temperature
llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")

stage_analyzer_prompt = PromptTemplate(template="""Determine the next conversation stage based on the following history:
{conversation_history}
Select a number from 1 to 7 representing the next stage:
{conversation_stages}
Only respond with the number, no other text.""",
input_variables=["conversation_history", "conversation_stages"])

sales_conversation_prompt = PromptTemplate(template="""You are {salesperson_name}, a {salesperson_role} at {company_name}.
Company Business: {company_business}
Company Values: {company_values}
Your Purpose is to {conversation_purpose}
Current conversation stage: {conversation_stage}

Conversation History: {conversation_history}

Response as {salesperson_name}: """,
input_variables=["salesperson_name", "salesperson_role", "company_name",
                 "company_business", "company_values",
                 "conversation_purpose", "conversation_stage", "conversation_history"])

stage_analyzer_chain = LLMChain(llm=llm, prompt=stage_analyzer_prompt)
sales_conversation_chain = LLMChain(llm=llm, prompt=sales_conversation_prompt)

@lru_cache(maxsize=128)
def determine_conversation_stage(conversation_history_str):
    result = stage_analyzer_chain.run(conversation_history=conversation_history_str,
                                      conversation_stages=json.dumps(conversation_stages))
    logger.info(f"Function determine_conversation_stage : {result}")
    return conversation_stages.get(result, conversation_stages['1'])

@lru_cache(maxsize=128)
def generate_response(conversation_history_str, current_stage):
    return sales_conversation_chain.run(**config, conversation_stage=current_stage,
                                        conversation_history=conversation_history_str)

def main():
    st.set_page_config(page_title="Chat with Sales Agent", page_icon=":robots:")
    st.title(f"EduConnect Corp. Sales Agent")

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
        st.session_state.current_stage = conversation_stages['1']
        initial_message = f"Hi there, I am a Sales agent. How can I help you with our product today?"
        st.session_state.conversation_history.append({"role": "assistant", "content": initial_message})

    with st.sidebar:
        st.subheader(f"Application Description")
        st.write("This application is a Streamlit-based chat interface that simulates a conversation with a sales agent. It uses the LangChain library to generate responses based on the conversation history and current stage of the sales process.")
        st.write("EduConnect is a SaaS platform designed for universities, colleges, and high schools")

    for message in st.session_state.conversation_history:
        with st.chat_message(message['role']):
            st.write(message["content"])

    if prompt := st.chat_input("Type your message here..."):
        st.session_state.conversation_history.append({'role': "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
            logger.info(f"Function main Users message : {prompt}")
        
        conversation_history_str = '\n'.join([m['content'] for m in st.session_state.conversation_history])
        st.session_state.current_stage = determine_conversation_stage(conversation_history_str)

        with st.chat_message("assistant"):
            response = generate_response(conversation_history_str, st.session_state.current_stage)
            st.write(response)
            logger.error(f"Function main assistant Responses: {response}")
        st.session_state.conversation_history.append({'role': "assistant", "content": response})

if __name__ == "__main__":
    main()