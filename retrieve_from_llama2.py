from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from langchain import PromptTemplate, SagemakerEndpoint
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer



# Get Env Variables
AWS_REGION='us-east-1'
EMBEDDING_MODEL='sentence-transformers/all-MiniLM-L6-v2'
LLAMA2_ENDPOINT='jumpstart-dft-meta-textgeneration-l-20240510-040141'
INFERENCE_COMPONENT = 'meta-textgeneration-llama-2-7b-f-20240510-040141'
MAX_HISTORY_LENGTH=10
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)



def extract_document_content(document):
    # Extract the text within the <document_content> tags
    start_tag = "<document_content>"
    end_tag = "</document_content>"
    start_index = document.find(start_tag) + len(start_tag)
    end_index = document.find(end_tag)
    return document[start_index:end_index].strip()
    
def build_chain():
    print('Preparing chain...')
    # Sentence transformer
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

    # Laod Faiss index
    db = FAISS.load_local("faiss_index", embeddings)

    # Define the Prompt for the 
    system_prompt = """You are an assistant for question-answering tasks for Retrieval Augmented Generation system for the financial reports such as 10Q and 10K.
                        Use the following pieces of retrieved context to answer the question. 
                        If the answer is directly available in the context, provide the precise answer.
                        If the answer is not directly available or cannot be inferred from the context, say that the information is not available to answer the question.
                        Use two sentences maximum and keep the answer concise.
                        Question: {question} 
                        Context: {context} 
                        Answer:"""

    # Custom ContentHandler to handle input and output to the SageMaker Endpoint
    class ContentHandler(LLMContentHandler):
        content_type = "application/json"
        accepts = "application/json"

        def transform_input(self, prompt: str, model_kwargs: dict) -> bytes:
            user_content = prompt.split("user_content: ")[-1].strip()
            payload = {
                "inputs": json.dumps([
                    [
                        {
                            "role": "system", "content": system_prompt,
                        },
                        {"role": "user", "content": user_content},
                    ],
                ]),
                "parameters": {"max_new_tokens": 1000, "top_p": 0.9, "temperature": 0.6},
            }
            input_str = json.dumps(payload)
            return input_str.encode("utf-8")

        def transform_output(self, output: bytes) -> str:
            try:
                response_str = output.read().decode("utf-8")
                response_json = json.loads(response_str)
                #print(f"Response JSON: {response_json}")  # Debug print

                if isinstance(response_json, list) and len(response_json) > 0 and 'generated_text' in response_json[0]:
                    content = response_json[0]['generated_text']
                    return content.split("Answer:")[-1].strip()
                else:
                    return "The information is not available to answer the question."
            except (json.JSONDecodeError, KeyError, IndexError, ValueError) as e:
                print(f"Error parsing response: {e}")
                return "The information is not available to answer the question."


    # Langchain chain for invoking SageMaker Endpoint
    llm = SagemakerEndpoint(
        endpoint_name=LLAMA2_ENDPOINT,
        region_name=AWS_REGION,
        content_handler=ContentHandler(),
        # credentials_profile_name="credentials-profile-name", # AWS Credentials profile name 
        # callbacks=[StreamingStdOutCallbackHandler()],
        endpoint_kwargs={"CustomAttributes": "accept_eula=true",
                        "InferenceComponentName": INFERENCE_COMPONENT},
    )

    def get_chat_history(inputs) -> str:
        res = []

        for _i in inputs:
            if len(_i) == 2:
                role, content = _i
                if role == "user":
                    user_content = content
                elif role == "assistant":
                    assistant_content = content
                    res.append(f"user:{user_content}\nassistant:{assistant_content}")
        return "\n".join(res)

    #Setting up RAG using ConversationalRetrieval Chain
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        get_chat_history=get_chat_history,
        # verbose=True,
    )

    return qa


# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)

def truncate_context(context, max_tokens):
    tokens = tokenizer.tokenize(context)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return tokenizer.convert_tokens_to_string(tokens)


def run_chain(chain, prompt: str, history=[], document=""):
    # Extract the document content if any
    document_content = extract_document_content(document)
    #print(f"Document Content: '{document_content}'")  

    max_input_tokens = 3000  # Reserve some tokens for the response
    truncated_context = truncate_context(document_content, max_input_tokens)
    # Prepare the input for the chain
    input_data = {
        "question": prompt,
        "chat_history": history,
        "context": truncated_context
    }
    #print(f"Input Data: {input_data}")  # Debug print
    
    # Run the chain
    response = chain(input_data)
    #print(f"Chain Response: {response}")  # Debug print
    
    # Extract the answer from the response
    answer = response.get("answer", "The information is not available to answer the question.")
    #print(f"Answer: {answer}")  # Debug print
    
    return answer
