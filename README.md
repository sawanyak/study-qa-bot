# Study Notes Q&A

A RAG-powered chatbot that lets you upload PDF documents and ask questions about them. Built with Streamlit, ChromaDB, SentenceTransformers, and Groq.

## Features

- Multi-chat support with conversation history
- Per-chat PDF upload and isolated knowledge bases
- Source citations with expandable preview
- Local embeddings via SentenceTransformers (no API key needed)
- Fast LLM inference via Groq

## Tech Stack

- **Frontend**: Streamlit
- **Vector DB**: ChromaDB
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2)
- **LLM**: Llama 3.3 70B via Groq
- **Document loading**: LangChain + PyPDF

## Setup

1. Clone the repo
   \`\`\`bash
   git clone https://github.com/YOUR_USERNAME/study-qa-bot.git
   cd study-qa-bot
   \`\`\`

2. Install dependencies
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

3. Add your Groq API key to a `.env` file
   \`\`\`
   GROQ_API_KEY=your_key_here
   \`\`\`
   Get a free key at [console.groq.com](https://console.groq.com).

4. Run the app
   \`\`\`bash
   streamlit run app.py
   \`\`\`

## How it works

1. Upload one or more PDFs to a chat
2. The app chunks them and stores embeddings in a per-chat ChromaDB collection
3. When you ask a question, it retrieves the most relevant chunks and passes them to the LLM as context
4. The LLM generates an answer grounded in your documents with citations

## Architecture

- `app.py` — Streamlit UI and chat management
- `ingest.py` — PDF loading and chunking
- `embeddings.py` — ChromaDB collection management
- `retriever.py` — Semantic search