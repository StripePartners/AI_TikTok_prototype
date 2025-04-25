import os
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, UnstructuredFileLoader  # for loading the pdf
# from langchain.embeddings import OpenAIEmbeddings # for creating embeddings
# from langchain.vectorstores import  Pinecone # for the vectorization part
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import pickle
# from google import genai
import re

##### Set up keys #####
# Read the Gemini key from a txt file
# gemini_key_path = './gemini_key.txt'
# with open(gemini_key_path, 'r') as file:
#     gemini_key = file.read().strip()
# print("Gemini key read successfully.")

##### Load theb PDFs - Letters #####
# add letters directory
letters_dir = './assets/letters'
print("Contents of 'letters' directory:", os.listdir(letters_dir))

# Identify the various PDF files
pdfs = [file for file in os.listdir(letters_dir) if file.endswith('.pdf')]

# Debugging: Print the identified PDF files
print("Identified PDF files:", pdfs)

# Loops through each PDF in the letters directory
# and loads the content using langchain's PyPDFLoader
page_list = []
for pdf in pdfs:
    pdf_path = os.path.join(letters_dir, pdf)
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    # Update metadata for each page
    for page in pages:
        new_doc_name = 'Berkshire Hathaway letters to shareholders - ' + re.findall(r'\d+', pdf)[0]
        page.metadata["document_name"] = new_doc_name
        page.metadata["document_type"] = 'letters'
    page_list.append(pages)

flat_list = [item for sublist in page_list for item in sublist]

# index_name = "buffett"
text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(flat_list)

# Extract text content and metadata from Document objects
# Metadata includes creator, total pages, and page label
text_documents = [Document(page_content=doc.page_content, 
                           metadata={"document_name": doc.metadata.get("document_name", "Unknown"), 
                                     "total_pages": doc.metadata.get("total_pages", "Unknown"),
                                     "page_label": doc.metadata.get("page_label", "Unknown"),
                                     "document_type": doc.metadata.get("document_type", "Unknown")}) for doc in texts]


##### Load theb PDFs - Books #####
# add letters directory
books_dir = './assets/books'
print("Contents of 'books' directory:", os.listdir(books_dir))

# Identify the various PDF files
pdfs = [file for file in os.listdir(books_dir) if file.endswith('.pdf')]
print("Identified PDF files:", pdfs)

# Loops through each PDF in the letters directory
# and loads the content using langchain's PyPDFLoader
page_list = []
for pdf in pdfs:
    pdf_path = os.path.join(books_dir, pdf)
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    # Update metadata for each page
    for page in pages:
        new_doc_name = pdf
        page.metadata["document_name"] = new_doc_name
        page.metadata["document_type"] = 'books'
    page_list.append(pages)

flat_list = [item for sublist in page_list for item in sublist]
text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(flat_list)

# Extract text content and metadata from Document objects
# Metadata includes creator, total pages, and page label
text_documents_book = [Document(page_content=doc.page_content, 
                            metadata={"document_name": doc.metadata.get("document_name", "Unknown"), 
                                     "total_pages": doc.metadata.get("total_pages", "Unknown"),
                                     "page_label": doc.metadata.get("page_label", "Unknown"),
                                     "document_type": doc.metadata.get("document_type", "Unknown")}) for doc in texts]

combined_knowledge = text_documents + text_documents_book

##### Load embedding models #####
# Initialize embeddings model (using a free local model)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Convert text into FAISS vector store
db = FAISS.from_documents(combined_knowledge, embedding_model)

# Save FAISS index
vector_dbs_path = './vector_dbs/'
faiss_path = vector_dbs_path + "faiss_index"
db.save_local(faiss_path)

# Save metadata separately
metadata_list = [{"text": doc.page_content, 
                  "document_name": doc.metadata["document_name"],
                  "total_pages": doc.metadata["total_pages"],
                  "page_label": doc.metadata["page_label"],
                  "document_type": doc.metadata["document_type"]} for doc in combined_knowledge]
with open(faiss_path+"/faiss_metadata.pkl", "wb") as f:
    pickle.dump(metadata_list, f)

print("Vector database stored successfully with metadata.")

# # Initialize embeddings model (using Google API)
# client = genai.Client(api_key=gemini_key)
# def get_google_embeddings(texts):
#     return [client.models.embed_content(model="gemini-embedding-exp-03-07", contents=text).embeddings for text in texts]


# embeddings = get_google_embeddings([doc.page_content for doc in text_documents])
# db = FAISS.from_embeddings(embeddings, text_documents)

# # Save FAISS index
# vector_dbs_path = '/Users/zoeliou/Documents/GitHub/AI_TikTok_prototype/vector_dbs/'
# faiss_path = vector_dbs_path + "faiss_index"
# db.save_local(faiss_path)

# # Save metadata separately
# metadata_list = [{"text": doc.page_content, "creator": doc.metadata["creator"]} for doc in text_documents]
# with open(faiss_path+"/faiss_metadata.pkl", "wb") as f:
#     pickle.dump(metadata_list, f)

# print("Vector database stored successfully with metadata.")
