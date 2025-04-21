# Wonderschool Provider Messaging Tool

This application is a simple messaging tool for childcare providers, created for the Wonderschool interview exercise using Streamlit and Python.

Providers can type in requests or questions through the chat interface. The system uses a Gemini LLM, accessed via a custom endpoint compatible with the OpenAI library, to process the input. It first checks if the query closely matches common questions stored in `faq.json`. This matching uses a dedicated LLM call with a very low temperature (0.0) for reliable identification against predefined topic descriptions. If a relevant FAQ is found, its stored answer is displayed.

If the message doesn't match an FAQ, a second, separate LLM call categorizes the message into general types like 'Support Request', 'Account Issue', or 'Other', again using a low temperature (0.1) for consistency. I chose this two-call approach because using specialized prompts for each task (FAQ matching vs. categorization) generally yields more reliable results than one complex prompt attempting both. The system also validates the category output against a predefined list and defaults to 'Other' if the LLM response is unexpected, serving as a basic way to handle potential hallucinations.

All incoming requests, along with their timestamp and determined category, are logged as individual rows in a specified Google Sheet using service account credentials for internal review.

The voice input capability mentioned in the exercise specification was deferred to focus on implementing the core text processing, LLM integration, and backend storage features within the suggested timeframe. Adding voice would be a potential next step, likely involving a Speech-to-Text service.

