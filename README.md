# Gemini-Document-Summarizer

# 📄 **AI Document Summarizer — Streamlit + Google Gemini**

A simple, fast, and free-to-deploy **AI-powered document summarization app** built using:

* **Google Gemini API** (2.5 Flash model)
* **Streamlit** (free deployment on Streamlit Community Cloud)
* **Google Colab** (optional prototyping environment)

This app lets users paste text or upload PDF/TXT files and receive clean, structured summaries.

---

## 🚀 **Live Demo**

Example:
[Streamlit App Link](https://gemini-document-summarizer.streamlit.app/)

---

## ✨ **Features**

### 📥 Input Options

* Paste text directly
* Upload **PDF** or **TXT** files
* Preview and **edit extracted text** before summarizing

### 🧠 Summarization Features

* Uses **Google Gemini 2.5 Flash** (fast, powerful, cheap)
* Choose:

  * **Summary style**: Bullet points, Paragraph, Executive brief
  * **Summary length**: Very short → Detailed
* High-quality, structured outputs

### 📊 Document Stats

Shows word and character count instantly.

### 📤 Export

* Download summaries as **TXT files**

### ☁️ Free Deployment

* 100% free on **Streamlit Community Cloud**
* Gemini API has a free tier suitable for light use

---

## 🏗️ **Architecture Overview**

```
[User]  
   ↓  
[Streamlit App UI]  
   ↓  
[Text / PDF Processor]  
   ↓  
[Gemini API - Summarize]  
   ↓  
[Streamlit Output + Download]
```

The backend is stateless. All summarization happens via Gemini API.

---

## 📦 **Project Structure**

```
document-summarizer/
│
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .streamlit/
│     └── secrets.toml  # (auto-created in Streamlit Cloud)
├── .gitignore
└── README.md
```

---

## 🤖 **Getting a Gemini API Key**

1. Go to **Google AI Studio**: [https://aistudio.google.com](https://aistudio.google.com)
2. Sign in with your Google account
3. Go to **API Keys**
4. Create an API key and copy it
5. Use it in `.env` (local) or Secrets (Streamlit Cloud)

---

## ☁️ **Deploy for Free on Streamlit Cloud**

### 1️⃣ Upload Your Repo to GitHub

Include:

* `app.py`
* `requirements.txt`
* `README.md`
* NO `.env` file!

### 2️⃣ Go to Streamlit Cloud

[https://share.streamlit.io](https://share.streamlit.io)

Click:

**New app → Connect GitHub repo**

### 3️⃣ Add Your API Key to Streamlit Secrets

In:

```
Advanced Settings → Secrets
```

Paste:

```toml
API_KEY_NAME = "your_actual_api_key"
```

### 4️⃣ Deploy 🎉

Streamlit installs dependencies & launches automatically.

---

---

## 🧪 **Prototyping in Google Colab**

If you want to test summarization logic before modifying the app:

```python
from google import genai

client = genai.Client(api_key="YOUR_API_KEY")

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Summarize this text: ...",
)

print(response.text)
```

Colab is great for experimenting with prompts and chunking logic.

---

Would you like me to **auto-generate this entire GitHub repo** for you (including folder structure + all files)?
