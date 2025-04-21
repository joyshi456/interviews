# app.py
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
from datetime import datetime
import os
from openai import OpenAI
import json 

# --- Load Base Secrets (URL, Sheet Key, Local File Path) ---
try:
    GEMINI_BASE_URL = st.secrets["GEMINI_BASE_URL"]
    SHEET_ID = st.secrets["SHEET_KEY"]
except KeyError as e:
    st.error(f"Missing required base secret (GEMINI_BASE_URL or SHEET_KEY): {e}. Please check secrets.")
    st.stop()
except Exception as e:
    st.error(f"Error loading base secrets: {e}")
    st.stop()

# --- Initialize OpenAI Client ---
try:
    client = OpenAI(
        api_key="EMPTY", 
        base_url=GEMINI_BASE_URL
    )
    GEMINI_MODEL_NAME = "gemini-1.5-flash" 
except Exception as e:
    st.error(f"Error initializing OpenAI client: {e}")
    st.stop()

worksheet = None # Initialize worksheet variable
try:
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = None

    if "PROJECT_ID" in st.secrets:
        print("Attempting to load credentials from Streamlit secrets.") # Debug log
        try:
           
            private_key_from_secrets = st.secrets["PRIVATE_KEY"].replace('\\n', '\n')

            credentials_dict = {
                "type": "service_account", 
                "project_id": st.secrets["PROJECT_ID"],
                "private_key_id": st.secrets["PRIVATE_KEY_ID"],
                "private_key": private_key_from_secrets,
                "client_email": st.secrets["CLIENT_EMAIL"],
                "client_id": st.secrets["CLIENT_ID"],
                "auth_uri": st.secrets.get("AUTH_URI", "https://accounts.google.com/o/oauth2/auth"),
                "token_uri": st.secrets.get("TOKEN_URI", "https://oauth2.googleapis.com/token"),
                "auth_provider_x509_cert_url": st.secrets.get("AUTH_PROVIDER", "https://www.googleapis.com/oauth2/v1/certs"), 
                "client_x509_cert_url": st.secrets["CLIENT_CERT_URL"], 
                "universe_domain": st.secrets.get("UNIVERSE_DOMAIN", "googleapis.com") 
            }
            creds = Credentials.from_service_account_info(credentials_dict, scopes=scopes)

        except KeyError as e:
             st.error(f"Missing required Google credential secret key on Streamlit Cloud: {e}. Check app settings & provided keys.")
             st.stop() 
        except Exception as e:
             st.error(f"Error loading credentials from Streamlit secrets: {e}")
             st.stop() 

    if creds is None:
        st.error("Fatal: Failed to establish Google credentials using secrets or local file.")
        st.stop()

    gc = gspread.authorize(creds)
    worksheet = gc.open_by_key(SHEET_ID).sheet1 # SHEET_ID was loaded earlier
    print("Successfully connected to Google Sheets.") # Debug log

except gspread.exceptions.SpreadsheetNotFound:
    st.error(f"Spreadsheet not found with key {SHEET_ID}. Check SHEET_KEY secret and ensure service account has Editor access.")
    st.stop()
except gspread.exceptions.APIError as e:
    st.error(f"Google Sheets API error: {e}. Check service account permissions or Sheets API enablement.")
    st.stop()
# Catch any other exceptions during the setup
except Exception as e:
    st.error(f"Unexpected error setting up Google Sheets connection: {e}")
    st.stop()
# --- End of Revised Authentication Block ---


# --- Load FAQ Data ---
FAQ_FILE = "wonderschool/faq.json" 
try:
    with open(FAQ_FILE, 'r', encoding='utf-8') as f:
        faq_data = json.load(f)["faq_database"]
    faq_topic_descriptions = [item["topic_description"] for item in faq_data] + ["General Inquiry / Other"]
except FileNotFoundError:
    st.error(f"Error: The FAQ file '{FAQ_FILE}' was not found in the 'wonderschool' directory.")
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


# --- Define Categories ---
# Using the simpler list based on previous code state
CATEGORIES = ["Support Request", "Product Feedback", "Account Issue", "Pricing Inquiry", "Website Update", "Other"]


# --- Helper Functions ---
def categorize_message_gemini(message):
    """Uses Gemini (via OpenAI client) to categorize the message."""
    if not CATEGORIES:
        st.error("Categories list is empty. Cannot perform categorization.")
        return "Error - Configuration Issue"

    system_prompt = f"""
    You are an assistant that categorizes messages from child care providers.
    Analyze the following message and categorize it into ONE of the following categories ONLY:
    {', '.join(CATEGORIES)}
    Output ONLY the category name. If none apply precisely, use 'Other'.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message}
    ]
    try:
        response = client.chat.completions.create(
            model=GEMINI_MODEL_NAME,
            messages=messages,
            temperature=0.1,
            max_tokens=50,
            n=1,
            stop=None,
            stream=False
        )
        category = response.choices[0].message.content.strip()
        if category not in CATEGORIES:
            st.warning(f"LLM provided invalid category '{category}'. Defaulting to 'Other'.")
            return "Other"
        return category
    except Exception as e:
        st.error(f"Error during categorization call: {e}")
        print(f"Detailed Error (Categorization): {e}")
        return "Error - Categorization Failed"

def auto_answer(message):
    """
    Uses LLM to match user message against predefined FAQ topics loaded from JSON
    and returns the corresponding answer if a relevant match is found.
    """
    if not faq_data:
         return None

    if not faq_topic_descriptions:
         st.error("FAQ Topic Descriptions list is empty.")
         return None

    topics_list_string = "\n - ".join(faq_topic_descriptions)
    system_prompt = f"""
    Analyze the user's message below. Determine which ONE of the following topics it matches best:
     - {topics_list_string}
    Respond with ONLY the exact text of the best matching topic description from the list above.
    If the message doesn't clearly match any topic other than "General Inquiry / Other", respond with "General Inquiry / Other".
    """
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": message}]
    try:
        response = client.chat.completions.create(
            model=GEMINI_MODEL_NAME, messages=messages, temperature=0.0,
            max_tokens=100, n=1, stop=None, stream=False
        )
        matched_topic_desc = response.choices[0].message.content.strip()

        if matched_topic_desc != "General Inquiry / Other":
            for item in faq_data:
                if item["topic_description"] == matched_topic_desc:
                    print(f"FAQ Match Found: {matched_topic_desc}")
                    # Ensure item has 'answer' key before returning
                    return item.get("answer", "Answer not found for matched topic.")
            st.warning(f"LLM matched '{matched_topic_desc}' but not found in source JSON.")
            return None
        else:
            print(f"No specific FAQ match. LLM returned: {matched_topic_desc}")
            return None
    except Exception as e:
        st.error(f"Error during FAQ matching call: {e}")
        print(f"Detailed Error (FAQ Match): {e}")
        return None

# Added generate_general_response based on previous state
def generate_general_response(user_message):
    """Generates a general conversational response using the LLM."""
    system_prompt = """
    You are a friendly and helpful assistant for the Wonderschool platform, interacting with a child care provider.
    Keep your response concise and conversational. If the user is just greeting you, greet them back.
    If the user asks a question you don't have specific information for, politely state you'll pass it on or ask for clarification.
    Avoid making up specific procedures or information.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    try:
        response = client.chat.completions.create(
            model=GEMINI_MODEL_NAME,
            messages=messages,
            temperature=0.1, #prevent hallucinations
            max_tokens=150,
            n=1,
            stop=None,
            stream=False
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error during general response generation: {e}")
        print(f"Detailed Error (General Response): {e}")
        return None

def store_request(timestamp, message, category, auto_response_content):
    """Appends the request details to the Google Sheet."""
    # Ensure worksheet is valid before trying to use it
    if worksheet is None:
        st.error("Cannot store request: Google Sheet connection not established.")
        return False
    try:
        category_str = str(category) if category else "Error - Category Undefined"
        if "Error" in category_str:
             st.warning(f"Attempting to log with category: {category_str}")

        row = [
            timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            message,
            category_str,
            auto_response_content or "N/A"
        ]
        worksheet.append_row(row)
        print(f"Stored request: {row}")
        return True
    except Exception as e:
        st.error(f"Error storing request in Google Sheet: {e}")
        print(f"Error storing request: {e}")
        return False


# --- Streamlit App UI ---
# (st.set_page_config, st.title, st.write, sidebar, message init/display) ...
st.set_page_config(page_title="Provider Support")
st.title("Wonderschool Provider Messaging")
st.write("Hello Provider! How can we help you today? Send us a message below.")

with st.sidebar:
    st.title("Options")
    def clear_chat_history():
        st.session_state.messages = [{"role": "assistant", "content": "Welcome! How can I assist you?"}]
    st.button('Clear Chat History', on_click=clear_chat_history)
    st.info("Messages are categorized and logged for internal review.")
    st.info("""
    **Voice Input:** For this exercise, please use the text input. Adding direct voice input would require additional libraries or browser integration.
    """)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome! How can I assist you?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --- Main Chat Input Logic (Using General Response for 'Other') ---
if prompt := st.chat_input("Your message here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        response_message = ""
        category = "Error - Processing Failed" # Default category
        auto_response_content = None # Content of FAQ answer if found
        stored_successfully = False

        with st.spinner("Processing your request..."):
            timestamp = datetime.now()

            # 1. Check for specific FAQ answer
            auto_response_content = auto_answer(prompt)

            # 2. Categorize the message (Needed if no FAQ, or always for logging?)
            # Let's always categorize for consistent logging.
            category = categorize_message_gemini(prompt)
            if "Error" in category:
                 st.warning("Failed to categorize message, proceeding.")
                 category = "Other" # Fallback category if categorization fails

            # 3. Store the request (using category determined above)
            stored_successfully = store_request(
                timestamp,
                prompt,
                category,
                auto_response_content # Log FAQ answer text if it exists
            )

            # 4. Determine final response message
            if auto_response_content:
                # Use the matched FAQ answer
                response_message = auto_response_content
                if not stored_successfully:
                     response_message += "\n\n*(Note: Answered question, but error logging request.)*"

            # If no FAQ answer, decide based on category
            elif category == "Other":
                # Generate a general conversational response for "Other" category
                print("Category is Other, attempting general response...") # Debug log
                general_response = generate_general_response(prompt)
                if general_response:
                    response_message = general_response
                else: # Fallback if general response fails
                    response_message = "Thanks for reaching out! We've received your message and will get back to you soon."
                # Optional: Add storage note if needed and storage failed
                if not stored_successfully:
                     response_message += "\n\n*(Note: Error logging request details.)*"
            else:
                # Specific Category (e.g., Support Request) but No FAQ Match
                response_message = f"Thanks for reaching out! We've received your message regarding '{category}' and will get back to you soon."
                if not stored_successfully:
                     response_message = "Thanks for reaching out! Error logging request details, but we'll review it manually." # Overwrite if storage failed

        # Display the final response message
        message_placeholder.markdown(response_message)

    # Add assistant response to session state
    st.session_state.messages.append({"role": "assistant", "content": response_message})