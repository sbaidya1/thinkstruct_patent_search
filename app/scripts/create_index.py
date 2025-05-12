"""
create_index.py

This script loads patent claim data from JSON files in the 'data' folder,
processes them into LangChain Document objects with metadata,
and builds a Chroma vector index using OpenAI embeddings.

Functions:
- load_patent_claims(): Loads and structures the raw patent data.
- build_index(): Embeds the documents and saves the index to disk.

Run this script directly to rebuild the Chroma index:
    python create_index.py
"""

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
import os, json, shutil

load_dotenv()

def load_patent_claims():
  """
    Loads and processes patent claims from JSON files in the ../data directory.
    Each claim is stored as a LangChain Document with relevant metadata (of the
    patent application it is from).

    Returns:
      List[Document]: List of processed claim documents with metadata.
  """
  docs = []
  folder = os.path.join(os.path.dirname(__file__), '..', 'data')
  folder = os.path.abspath(folder)

  for file in os.listdir(folder):
    if file.endswith(".json"):
      with open(os.path.join(folder, file)) as f:
        data = json.load(f)
        for patent in data:
          # excludes any patent with no claims
          if patent.get("claims"):
            claims = patent.get("claims", [])
            for i, claim in enumerate(claims):
              # split each patent into multiple Documents, one per claim, for logical indexing and retrieval
              docs.append(Document(
                page_content=claim.strip(),
                metadata={
                  "doc_number": patent.get("doc_number", ""),
                  "title": patent.get("title", ""),
                  "classification": patent.get("classification", ""),
                  "abstract": patent.get("abstract", ""),
                  "description": " ".join(patent.get("detailed_description", [])) 
                }
              ))
  return docs

def build_index():
  """
    Deletes any existing Chroma index directory, generates embeddings
    for patent claim documents, and creates a new persistent Chroma index.
  """
  index_dir = "chroma_index"
  if os.path.exists(index_dir):
    shutil.rmtree(index_dir)
    
  docs = load_patent_claims()

  embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
  Chroma.from_documents(
      documents=docs,
      embedding=embeddings,
      persist_directory="chroma_index"
  )

if __name__ == "__main__":
  build_index()