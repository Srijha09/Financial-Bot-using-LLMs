import glob
import os
import json

from __future__ import annotations
from langchain.document_loaders import BSHTMLLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain import PromptTemplate, SagemakerEndpoint
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from transformers import AutoTokenizer



# Get Env Variables
AWS_REGION='us-east-1'
EMBEDDING_MODEL='sentence-transformers/all-MiniLM-L6-v2'
LLAMA2_ENDPOINT='jumpstart-dft-meta-textgeneration-l-20240510-040141'
INFERENCE_COMPONENT = 'meta-textgeneration-llama-2-7b-f-20240510-040141'
MAX_HISTORY_LENGTH=10


# Step1: Define a sentence transformer model that will be used
#        to convert the documents into vector embeddings
embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
loader = PyPDFLoader(r"data/tsla-20240331-gen.pdf")

# load your data
print('Loading the financial report corpus ...')
data = loader.load()
# Text splitter
print('Instantiating Text Splitter...')
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=300)
all_splits = text_splitter.split_documents(data)

    
# Step3: Create & save a vector database with the vector embeddings
#        of the documents
print('Preparing Vector Embeddings...')
db = FAISS.from_documents(all_splits, embeddings)
db.save_local("faiss_index")
print("Done")