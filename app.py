import json
import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
from functools import lru_cache
import warnings
from textblob import TextBlob  # For sentiment analysis
import datetime
import pandas as pd

warnings.filterwarnings('ignore')

load_dotenv()

GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

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

# Replace Ollama with Groq
llm = ChatGroq(
    temperature=0.7,
    model_name="mixtral-8x7b-32768",
    groq_api_key="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"
)

stage_analyzer_prompt = PromptTemplate(template="""Determine the next conversation stage based on the following history:
{conversation_history}
Select a number from 1 to 7 representing the next stage:
{conversation_stages}
Only respond with the number, no other text.""",
input_variables=["conversation_history", "conversation_stages"])

def analyze_sentiment(text):
    """Analyze the sentiment of given text using TextBlob with more nuanced thresholds"""
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    
    # Using more nuanced thresholds
    if polarity >= 0.1:
        return "positive"
    elif polarity <= -0.1:
        return "negative"
    else:
        return "neutral"

# Update the sales conversation prompt to include more detailed emotional context
sales_conversation_prompt = PromptTemplate(template="""You are {salesperson_name}, a {salesperson_role} at {company_name}.
Company Business: {company_business}
Company Values: {company_values}
Your Purpose is to {conversation_purpose}
Current conversation stage: {conversation_stage}

Customer's current sentiment: {customer_sentiment}
Guidelines based on sentiment:
- If positive: Build on their enthusiasm and deepen engagement
- If negative: Show empathy, address concerns, and focus on solutions
- If neutral: Focus on value proposition and building interest

Conversation History: {conversation_history}

Response as {salesperson_name}: """,
input_variables=["salesperson_name", "salesperson_role", "company_name",
                "company_business", "company_values", "conversation_purpose", 
                "conversation_stage", "conversation_history", "customer_sentiment"])

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
    # Get the last user message for sentiment analysis
    messages = conversation_history_str.split('\n')
    last_user_message = ""
    for message in reversed(messages):
        if "User:" in message:
            last_user_message = message.replace("User:", "").strip()
            break
    
    # Analyze sentiment
    sentiment = analyze_sentiment(last_user_message)
    
    return sales_conversation_chain.run(
        **config,
        conversation_stage=current_stage,
        conversation_history=conversation_history_str,
        customer_sentiment=sentiment
    )

def save_feedback(message, rating, feedback_text):
    """Save feedback to a CSV file"""
    feedback_data = {
        'timestamp': datetime.datetime.now(),
        'message': message,
        'rating': rating,
        'feedback': feedback_text
    }
    
    try:
        df = pd.read_csv('feedback_data.csv')
    except FileNotFoundError:
        df = pd.DataFrame(columns=['timestamp', 'message', 'rating', 'feedback'])
    
    df = pd.concat([df, pd.DataFrame([feedback_data])], ignore_index=True)
    df.to_csv('feedback_data.csv', index=False)
    return True

def display_feedback_ui(message, idx):
    """Display feedback UI for a message"""
    with st.expander("Rate this response"):
        col1, col2 = st.columns([3, 1])  # Create columns for better layout
        
        with col1:
            rating = st.slider(
                "How helpful was this response?",
                1, 5, 3,
                key=f"rating_slider_{idx}"
            )
            feedback = st.text_area(
                "Additional feedback (optional)",
                key=f"feedback_text_{idx}"
            )
        
        with col2:
            submit_button = st.button(
                "Submit Feedback", 
                key=f"submit_btn_{idx}"
            )
            
        if submit_button:
            save_feedback(message, rating, feedback)
            st.success("Thank you for your feedback!")
            return True
    return False

def main():
    st.set_page_config(page_title="Chat with Sales Agent", page_icon=":robots:")
    st.title(f"EduConnect Corp. Sales Agent")

    # Initialize feedback state
    if 'feedback_given' not in st.session_state:
        st.session_state.feedback_given = set()

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
        st.session_state.current_stage = conversation_stages['1']
        initial_message = f"Hi there, I am a Sales agent. How can I help you with our product today?"
        st.session_state.conversation_history.append({"role": "assistant", "content": initial_message})

    with st.sidebar:
        st.subheader(f"Conversation Insights")
        if st.session_state.conversation_history:
            # Show sentiment analysis in sidebar
            last_user_message = next((msg["content"] for msg in reversed(st.session_state.conversation_history) 
                                    if msg["role"] == "user"), None)
            if last_user_message:
                sentiment = analyze_sentiment(last_user_message)
                st.write(f"Current Customer Sentiment: **{sentiment.title()}**")
                
                # Add a sentiment indicator
                if sentiment == "positive":
                    st.success("😊 Customer appears positive")
                elif sentiment == "negative":
                    st.error("😟 Customer appears concerned")
                else:
                    st.info("😐 Customer appears neutral")

        # Add feedback statistics in sidebar
        try:
            df = pd.read_csv('feedback_data.csv')
            st.subheader("Feedback Statistics")
            avg_rating = df['rating'].mean()
            st.metric("Average Response Rating", f"{avg_rating:.2f}/5.0")
            total_feedback = len(df)
            st.metric("Total Feedback Received", total_feedback)
        except FileNotFoundError:
            pass

        st.subheader(f"Application Description")
        st.write("This application is a Streamlit-based chat interface that simulates a conversation with a sales agent. It uses the LangChain library to generate responses based on the conversation history and current stage of the sales process.")
        st.write("EduConnect is a SaaS platform designed for universities, colleges, and high schools")

    # Display conversation history with feedback options
    for idx, message in enumerate(st.session_state.conversation_history):
        with st.chat_message(message['role']):
            st.write(message["content"])
            # Only show feedback option for assistant messages
            if message['role'] == "assistant" and idx not in st.session_state.feedback_given:
                if display_feedback_ui(message["content"], idx):
                    st.session_state.feedback_given.add(idx)

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
