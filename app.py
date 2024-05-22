import os
import pinecone
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_pinecone import PineconeVectorStore 
from langchain import HuggingFaceHub
from flask import Flask, request

app = Flask(__name__)

PINECONE_API_KEY = "49d330f0-a127-4534-83e4-be396251e67b"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index_name = "sarthibot"
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY

docSearch = PineconeVectorStore.from_existing_index(index_name, embeddings)

prompt_template = """
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Context:{context}
Question:{question}
Only return the helpful answer below and nothing else.
Helpful answer:
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt":PROMPT}

llm = HuggingFaceHub(huggingfacehub_api_token="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", repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.8, "max_length":512})
chain = load_qa_chain(llm, chain_type="stuff")

def solve(queryText):
    docsResult = docSearch.similarity_search(queryText, k=2)
    return(chain.run(input_documents=docsResult, question=queryText))

@app.route("/")
def index():
    return "Hello World!"

@app.route("/chat", methods=["POST", "GET"])
def chat():
    output = request.get_json()
    if(not len(output)):
        return {"status" : "Bad Response"}
    else:
        
        input = output["prompt"]
        result = solve(input)
        return {"answer" : result}
    
if __name__ == "__main__":
    app.run(debug=True)