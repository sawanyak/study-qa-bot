# 📚 Study Notes Q&A Bot

A RAG (Retrieval-Augmented Generation) chatbot that lets you upload your study notes as PDFs and ask natural-language questions about them. Built with Streamlit, ChromaDB, SentenceTransformers, and Groq's LLaMA 3.3 70B.

🔗 **Live demo:** [study-app-bot.streamlit.app](https://study-app-bot-evwdfx3efdeibipbjadt8y.streamlit.app/)

💻 **Repo:** [github.com/sawanyak/study-qa-bot](https://github.com/sawanyak/study-qa-bot)

---

## ✨ Features

- 💬 **Multi-chat interface** — keep separate conversations for different subjects, each with its own documents and history
- 📎 **Per-chat PDF uploads** — drop in one or more PDFs and start asking questions
- 🧠 **Conversational memory** — the bot remembers earlier turns so you can ask natural follow-up questions
- 🔍 **Clean answers with toggleable sources** — answers read naturally without inline citations; click **Show sources** to see the exact retrieved chunks and page numbers
- ⚙️ **Adjustable retrieval** — tune how many source chunks are pulled per question
- 🎨 **Modern dark UI** — black / white / purple theme inspired by Claude and ChatGPT

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Frontend | Streamlit |
| Vector DB | ChromaDB (local, persistent) |
| Embeddings | SentenceTransformers (`all-MiniLM-L6-v2`) |
| PDF parsing | LangChain + `pypdf` |
| Text chunking | `RecursiveCharacterTextSplitter` (500 chars, 50 overlap) |
| LLM | Groq API — LLaMA 3.3 70B Versatile |
| Deployment | Streamlit Community Cloud |

---

## 📁 Project Structure

```
study-qa-bot/
├── app.py            # Streamlit UI + chat orchestration
├── ingest.py         # PDF loading + chunking
├── embeddings.py     # ChromaDB collection management
├── retriever.py      # Similarity search over chunks
├── requirements.txt  # Pinned dependencies
├── runtime.txt       # Python 3.11 pin for Streamlit Cloud
└── README.md
```

---

## 🚀 Running Locally

### 1. Clone and install

```bash
git clone https://github.com/sawanyak/study-qa-bot.git
cd study-qa-bot
pip install -r requirements.txt
```

### 2. Get a free Groq API key

Sign up at [console.groq.com](https://console.groq.com) and create an API key.

### 3. Add it to a `.env` file

```bash
echo "GROQ_API_KEY=your_key_here" > .env
```

### 4. Run

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---
