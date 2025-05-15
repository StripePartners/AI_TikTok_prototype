import os
import re
import nltk
import pandas as pd
import xlsxwriter
#nltk.download('stopwords')

from Questgen import main
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, UnstructuredFileLoader  # for loading the pdf
from langchain.text_splitter import TokenTextSplitter
from tqdm import tqdm
qg = main.QGen()


letters_dir = './assets/letters'
books_dir = './assets/books'

# Identify the various PDF files
pdfs = [os.path.join(letters_dir,file) for file in os.listdir(letters_dir) if file.endswith('.pdf')] + [os.path.join(books_dir,file) for file in os.listdir(books_dir) if file.endswith('.pdf')]

# Loops through each PDF in the letters directory and loads the content using langchain's PyPDFLoader
page_list = []
for pdf_path in pdfs:
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    pdf = pdf_path.split("/")[-1]
    
    # Update metadata for each page
    for page in pages:
        if letters_dir in pdf_path:
            new_doc_name = 'Berkshire Hathaway letters to shareholders - ' + re.findall(r'\d+', pdf)[0]
            page.metadata["document_name"] = new_doc_name
            page.metadata["document_type"] = 'letters'
        
        else:
            new_doc_name = pdf
            page.metadata["document_name"] = new_doc_name
            page.metadata["document_type"] = 'books'

    page_list.append(pages)

flat_list = [item for sublist in page_list for item in sublist]
text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=0)
texts = text_splitter.split_documents(flat_list)
print(texts[0],texts[0].metadata["document_name"] + '_page' + texts[0].metadata["page_label"],len(texts),len(flat_list))

# Generate questions for each page in the documents in the knowledge base
all_questions = []
all_answers = []
all_contexts = []
all_doc_types = []
all_docs = []


for text in tqdm(texts):
    payload = {"input_text":text.page_content,"max_questions":2}
    output = qg.predict_shortq(payload)
    if "questions" in output.keys():
        if len(output["questions"])>0:
            all_questions.extend([x["Question"] for x in output["questions"]])
            all_answers.extend([x["Answer"] for x in output["questions"]])
            all_contexts.extend([x["context"] for x in output["questions"]])

            all_doc_types.extend([text.metadata["document_type"]]*len(output["questions"]))
            all_docs.extend([text.metadata["document_name"] + '_page' + text.metadata["page_label"]]*len(output["questions"]))
            print(text.metadata["document_type"])
            print(text.metadata["document_name"] + '_page' + text.metadata["page_label"])



# Save generated question-answer pairs
df = pd.DataFrame()
df["question"] = all_questions
df["answer"] = all_answers
df["context"] = all_contexts
df["doc"] = all_docs
df["doc_type"] = all_doc_types

df.to_excel("/Users/imanbilal/Desktop/StripePartners/Finance_TikTok_prototype/AI_TikTok_prototype-iman_test/generated_questions.xlsx",engine='xlsxwriter')
