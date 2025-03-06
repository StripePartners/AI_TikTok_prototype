import os
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, UnstructuredFileLoader  # for loading the pdf
from langchain.embeddings import OpenAIEmbeddings # for creating embeddings
from langchain.vectorstores import  Pinecone # for the vectorization part
from langchain.text_splitter import TokenTextSplitter


# add letters directory
letters_dir = '/Users/zoeliou/Documents/GitHub/AI_TikTok_prototype/letters'
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
    page_list.append(pages)

flat_list = [item for sublist in page_list for item in sublist]

# index_name = "buffett"

text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(flat_list)
