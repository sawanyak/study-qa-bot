import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_uploaded_pdf(uploaded_file, chunk_size: int = 500, chunk_overlap: int = 50):
    """Take a Streamlit UploadedFile, save it temporarily, load and chunk it."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

        for page in pages:
            page.metadata["source"] = uploaded_file.name

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " "],
        )
        chunks = splitter.split_documents(pages)
        return chunks
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)