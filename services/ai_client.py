import os
import streamlit as st
from google import genai
from dotenv import load_dotenv

# Load local environment variables (.env) if present.
# On Streamlit Cloud, secrets are injected differently,
# but load_dotenv() won't hurt.
load_dotenv()


def _create_client() -> genai.Client:
    """
    Creates and returns a configured Gemini client.

    Reads the API key from environment variables or Streamlit secrets.
    If no key is found, shows an error and stops the app.

    Returns:
        genai.Client instance.
    """
  load_dotenv()


# Retrieve Gemini API key from Streamlit Secrets. 
  GEMINI_API_KEY = st.secrets["gemapikey"]

# Prevent app from running without an API key
  if not GEMINI_API_KEY:
      st.error("GEMINI_API_KEY not found. Please set it as an environment variable or Streamlit secret.")
      st.stop()

# Initialize Gemini client with the API key
  client = genai.Client(api_key=GEMINI_API_KEY)


# Global shared Gemini client for the whole app.
client: genai.Client = _create_client()
