from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader  # swap per file type

def load_and_chunk(filepath: str, chunk_size: int = 800, chunk_overlap: int = 100):
    loader = PyPDFLoader(filepath)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,      # smaller than your vanilla RAG chunk size on purpose
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    print(f"{filepath}: {len(docs)} pages -> {len(chunks)} extraction-ready chunks")
    return chunks