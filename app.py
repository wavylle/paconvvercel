# A very simple Flask Hello World app for you to get started with...

from flask import Flask, jsonify
from flask import request
import json
import time
import requests
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Pinecone
from langchain.chains import LLMChain
from langchain import VectorDBQA, OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
import pinecone
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
# import nltk
# nltk.download('punkt')
# from nltk.tokenize import word_tokenize
from collections import namedtuple
import os
from dotenv import load_dotenv
import sys
# Load environment variables from .env file
load_dotenv()

# Define the named tuple structure
Document = namedtuple('Document', ['page_content', 'metadata'])

pinecone.init(
    api_key=os.environ.get("PINECONE_API_KEY"),
    environment=os.environ.get("PINECONE_ENVIRONMENT"),
)

INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")


def read_text_file(url):
    response = requests.get(url)
    if response.status_code == 200:
        # Extract the content as a single string
        content = response.text.strip()

        content = content.replace('\r', '')

        # Create the Document namedtuple and populate the array
        documents = [Document(page_content=content, metadata={'source': 'data.txt'})]
        return documents
    else:
        print("Error: Failed to fetch the text file.")
        return []


app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello from Flask!'


@app.route("/testpost", methods=['POST'])
def testPost():
    data = request.get_json()

    # Retrieve values
    question = data.get("question")
    file_url = data.get("file_url")

    json_response = {"question": question, "file_url": file_url, "message": "POST request received"}
    time.sleep(35)
    return jsonify(json_response)


@app.route('/get-data', methods=['POST'])
def paconv():
    # Access the JSON payload
    data = request.get_json()

    print("API KEY: ", os.environ.get("OPENAI_API_KEY"))

    # Retrieve values
    question = data.get("question")
    file_url = data.get("file_url")
    pinecone_api_key = data.get("pinecone_api_key")
    pinecone_environment = data.get("pinecone_environment")

    pinecone.init(
        api_key=pinecone_api_key,
        environment=pinecone_environment,
    )

    print("FILE URL: ", file_url)

    document = read_text_file(file_url)

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    docsearch = Pinecone.from_documents(
        texts, embeddings, index_name="pythonllm-embeddings2"
    )

    qa = VectorDBQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", vectorstore=docsearch, return_source_documents=True
    )

    query = question
    result = qa({"query": query})

    if "result" in result:
        json_response = {"question": question, "file_url": file_url, "message": result["result"]}
    else:
        json_response = {"question": question, "file_url": file_url,
                         "message": "There's an error from our end. We are getting on it."}

    print("RESULT: ", result)
    print("RESULT TYPE: ", type(result))

    return jsonify(json_response)


@app.route('/answer', methods=['POST'])
def getanswer():
    data = request.get_json()

    print("API KEY: ", os.environ.get("OPENAI_API_KEY"))

    # Retrieve values
    question = data.get("question")
    file_url = data.get("file_url")
    pinecone_api_key = data.get("pinecone_api_key")
    pinecone_environment = data.get("pinecone_environment")

    try:

        llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"))

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=0
        )

        document = read_text_file(file_url)

        docs = text_splitter.split_documents(document)
        chain = load_qa_chain(llm, chain_type="map_rerank", verbose=True, return_intermediate_steps=True)

        query = question

        result = chain({"input_documents": docs, "question": query}, return_only_outputs=True)

        if "output_text" in result:
            json_response = {"question": question, "file_url": file_url, "message": result["output_text"]}
        else:
            json_response = {"question": question, "file_url": file_url,
                             "message": "There's an error from our end. We are getting on it."}
    except:
        json_response = {"question": question, "file_url": file_url,
                         "message": "There's an error from our end. We are getting on it."}

    return jsonify(json_response)


@app.route("/productassiststoreembeddings", methods=['POST'])
def storeEmbeddigsPinecone():
    data = request.get_json()

    openai_api_key = os.environ.get("OPENAI_API_KEY")

    openai_api_key_rec = data.get('openai_api_key')

    if "openai_api_key_rec" is not None:
        openai_api_key = openai_api_key_rec

    print("API KEY: ", openai_api_key)

    # Retrieve values
    file_url = data.get("file_url")
    embedding_map = data.get("embedding_map")

    try:
        document = read_text_file(file_url)
        response = requests.get(file_url)
        # Extract the content as a single string
        content = response.text.strip()
    
        content = content.replace('\r', '')
        if "\n\n" in content:
            print("Here...")
            print(sys.getrecursionlimit())
            text_splitter = RecursiveCharacterTextSplitter(chunk_overlap=0, separators=["\n\n"])
            texts = text_splitter.split_documents(document)
            print("Splitting text block 1...")
        else:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            texts = text_splitter.split_documents(document)
        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        # print(text_splitter)
        # texts = text_splitter.split_documents(document)
    
        embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    
        doc_store = Pinecone.from_texts([d.page_content for d in texts], embeddings, index_name=INDEX_NAME,
                                        namespace=embedding_map)
    
        json_message = {"status": "success"}
    except Exception as e:
        json_message = {"status": "failed", "error": str(e)}

    return jsonify(json_message)


@app.route("/productassistgetdata", methods=['POST'])
def responseFromPinecone():
    data = request.get_json()

    openai_api_key = os.environ.get("OPENAI_API_KEY")

    openai_api_key_rec = data.get('openai_api_key')

    if "openai_api_key_rec" is not None:
        openai_api_key = openai_api_key_rec

    print("API KEY: ", openai_api_key)

    # Retrieve values
    question = data.get("question")
    file_url = data.get("file_url")
    embedding_map = data.get("embedding_map")

    try:
        # document = read_text_file(file_url)
        # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        # texts = text_splitter.split_documents(document)

        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

        doc_store = Pinecone.from_existing_index(index_name=INDEX_NAME, embedding=embeddings, namespace=embedding_map)

        query = question

        # llm = OpenAI(temperature=0, openai_api_key=os.environ.get("OPENAI_API_KEY"))
        # qa_chain = load_qa_chain(llm, chain_type="stuff")
        docs = doc_store.similarity_search(query)
        combined_text = '\n\n'.join(doc.page_content for doc in docs)

        prompt_template = """Given this information: {information}
                            Assist me with this: {question}"""

        prompt = PromptTemplate(
        input_variables=["information", "question"],
        template=prompt_template,
        )

        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)
        chain = LLMChain(llm=llm, prompt=prompt)
        result1 = chain.run(information=combined_text, question=query)
        # result1 = ""
        result = ""
        # result = qa_chain.run(input_documents=docs, question=query)

        doc_dict = []

        print("Before the loop")

        for e, x in enumerate(docs):
            doc_dict.append({"page_content": x.page_content, "metadata": x.metadata})

        print("Doc Dict: ", doc_dict)

        json_message = {"status": "success", "result": result, "docs": doc_dict, "result1": result1}

    except Exception as e:
        print("There's an error!!")
        json_message = {"status": "failed", "error": str(e)}

    return jsonify(json_message)


if __name__ == "__main__":
    app.run(debug=True)
