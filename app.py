# app.py
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
from datetime import datetime
import os
from openai import OpenAI 
import json

# Load secrets
try:
    GEMINI_BASE_URL = st.secrets["GEMINI_BASE_URL"]
    SHEET_ID = st.secrets["SHEET_KEY"]
    SERVICE_ACCOUNT_FILE = st.secrets["SERVICE_ACCOUNT_FILE"]
except KeyError as e:
    st.error(f"Missing required secret: {e}. Please check your .streamlit/secrets.toml file.")
    st.stop()
except Exception as e:
    st.error(f"Error loading secrets: {e}")
    st.stop()

try:
    client = OpenAI(
        api_key="EMPTY",
        base_url=GEMINI_BASE_URL
    )
    GEMINI_MODEL_NAME = "gemini-1.5-flash"
except Exception as e:
    st.error(f"Error initializing OpenAI client for Gemini endpoint: {e}")
    st.stop()

try:
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    if not os.path.exists(SERVICE_ACCOUNT_FILE):
        st.error(f"Service account file not found at path: {SERVICE_ACCOUNT_FILE}")
        st.info("Ensure 'credentials.json' is in the same directory as app.py or update SERVICE_ACCOUNT_FILE in secrets.toml")
        st.stop()

    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=scopes)
    gc = gspread.authorize(creds)
    worksheet = gc.open_by_key(SHEET_ID).sheet1
except FileNotFoundError:
     st.error(f"Credentials file not found at {SERVICE_ACCOUNT_FILE}. Make sure it's in the correct path.")
     st.stop()
except gspread.exceptions.SpreadsheetNotFound:
     st.error(f"Spreadsheet not found. Check your SHEET_KEY in secrets.toml and ensure the service account has access.")
     st.stop()
except Exception as e:
    st.error(f"Error connecting to Google Sheets: {e}")
    st.stop()

# --- Load FAQ Data ---
FAQ_FILE = "faq.json"
try:
    with open(FAQ_FILE, 'r', encoding='utf-8') as f:
        faq_data = json.load(f)["faq_database"]
    faq_topic_descriptions = [item["topic_description"] for item in faq_data] + ["General Inquiry / Other"]
except FileNotFoundError:
    st.error(f"Error: The FAQ file '{FAQ_FILE}' was not found.")
    faq_data = []
    faq_topic_descriptions = ["General Inquiry / Other"] 
except (json.JSONDecodeError, KeyError) as e:
    st.error(f"Error reading or parsing {FAQ_FILE}: {e}")
    faq_data = []
    faq_topic_descriptions = ["General Inquiry / Other"] 
except Exception as e:
    st.error(f"An unexpected error occurred loading FAQ data: {e}")
    faq_data = []
    faq_topic_descriptions = ["General Inquiry / Other"] 



CATEGORIES = ["Support Request", "Product Feedback", "Account Issue", "Pricing Inquiry", "Website Update", "Other"]

# --- Helper Functions ---

def categorize_message_gemini(message):
    """Uses Gemini (via OpenAI client) to categorize the message."""
    system_prompt = f"""
    You are an assistant that categorizes messages from child care providers.
    Analyze the following message and categorize it into ONE of the following categories ONLY:
    {', '.join(CATEGORIES)}
    Output ONLY the category name.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message}
    ]
    try:
        response = client.chat.completions.create(
            model=GEMINI_MODEL_NAME,
            messages=messages,
            temperature=0.1, # Low temperature for deterministic categorization
            max_tokens=50,  # Limit response length
            n=1,
            stop=None,
            stream=False
        )
        category = response.choices[0].message.content.strip()

        # Validate category - handle potential hallucinations / unexpected output
        if category not in CATEGORIES:
            st.warning(f"LLM provided invalid category '{category}'. Defaulting to 'Other'.")
            return "Other"
        return category
    except Exception as e:
        st.error(f"Error during categorization call to Gemini endpoint: {e}")
        return "Error - Categorization Failed"

def auto_answer(message):
    """
    Uses LLM to match user message against predefined FAQ topics loaded from JSON
    and returns the corresponding answer if a relevant match is found.
    """
    if not faq_data: 
        st.warning("FAQ data is unavailable. Cannot provide automatic answers.")
        return None

    # Create the list of topics dynamically for the prompt
    topics_list_string = "\n - ".join(faq_topic_descriptions)

    system_prompt = f"""
    Analyze the user's message below. Determine which ONE of the following topics it matches best:
     - {topics_list_string}

    Respond with ONLY the exact text of the best matching topic description from the list above.
    If the message doesn't clearly match any topic other than "General Inquiry / Other", respond with "General Inquiry / Other".
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message}
    ]

    try:
        response = client.chat.completions.create(
            model=GEMINI_MODEL_NAME,
            messages=messages,
            temperature=0.0, 
            max_tokens=100, 
            n=1,
            stop=None,
            stream=False 
        )
        matched_topic = response.choices[0].message.content.strip()

        # Check if the matched topic is valid and not the general one
        if matched_topic in faq_topic_descriptions and matched_topic != "General Inquiry / Other":
            # Find the corresponding answer in our loaded data
            for item in faq_data:
                if item["topic_description"] == matched_topic:
                    print(f"FAQ Match Found: {matched_topic}")
                    return item["answer"] 
            # This part should ideally not be reached if matched_topic is valid
            st.warning(f"Matched topic '{matched_topic}' not found in FAQ data source after LLM match.")
            return None
        else:
            # No specific FAQ match found by the LLM
            print(f"No specific FAQ match. LLM returned: {matched_topic}") 
            return None

    except Exception as e:
        st.error(f"Error during FAQ matching call to Gemini endpoint: {e}")
        print(f"Detailed Error (FAQ Match): {e}")
        return None


def store_request(timestamp, message, category, auto_response):
    """Appends the request details to the Google Sheet."""

    try:
        row = [timestamp.strftime("%Y-%m-%d %H:%M:%S"), message, category, auto_response or "N/A"]
        worksheet.append_row(row)
        print(f"Stored request: {row}") 
        return True
    
    except Exception as e:
        st.error(f"Error storing request in Google Sheet: {e}")
        print(f"Error storing request: {e}")
        return False

# --- Streamlit App UI ---

st.set_page_config(page_title="Provider Support")
st.title("Wonderschool Provider Messaging")
st.write("Hello Provider! How can we help you today? Send us a message below.")


with st.sidebar:
    st.title("Options")
    def clear_chat_history():
        st.session_state.messages = [{"role": "assistant", "content": "Welcome! How can I assist you?"}]
    st.button('Clear Chat History', on_click=clear_chat_history)
    st.info("Messages are categorized and logged for internal review.")

# Initialize or load chat messages
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome! How can I assist you?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input via Streamlit's chat input
if prompt := st.chat_input("Your message here..."):
    # Append and display user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process the message and generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty() 
        with st.spinner("Processing your request..."):
            timestamp = datetime.now()

            # 1. Check for simple answers (Bonus Point)
            auto_response = auto_answer(prompt)

            # 2. Categorize the message using Gemini
            category = categorize_message_gemini(prompt)

            # 3. Store the request in Google Sheets
            stored_successfully = store_request(timestamp, prompt, category, auto_response)

            # 4. Determine and display assistant's response
            response_message = ""
            if auto_response:
                # Respond with the simple answer
                response_message = auto_response
                if not stored_successfully:
                     response_message += "\n\n*(Note: We answered your question, but there was an issue saving the request details. We'll still review it.)*"
            else:
                # Send a generic acknowledgment
                response_message = f"Thanks for reaching out! We've received your message (categorized as: *{category}*) and will get back to you soon."
                if not stored_successfully:
                    response_message = "Thanks for reaching out! We received your message, but had an issue logging the details. We'll review it manually."

            # Update the placeholder with the final response
            message_placeholder.markdown(response_message)

    # Add assistant response to session state *after* displaying it
    st.session_state.messages.append({"role": "assistant", "content": response_message})


