from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers, Bedrock
from langchain.chains import LLMChain, RetrievalQA
from langchain_experimental.llms.anthropic_functions import AnthropicFunctions
from pymilvus import utility
from langchain.vectorstores import Milvus
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from django.conf import settings
from pymilvus import connections
import boto3
import os

class LanguageModelManager:
    def __init__(self, model_path, temperature, context_length, gpu_layers):
        self.model_path = model_path
        self.temperature = temperature
        self.context_length = context_length
        self.gpu_layers = gpu_layers

    def load_model(self):
        raise NotImplementedError("Subclasses must implement the load_model method.")

class CTransformersManager(LanguageModelManager):
    def load_model(self):
        return CTransformers(
            model=self.model_path,
            config={
                'max_new_tokens': int(512),
                'temperature': self.temperature,
                'context_length': self.context_length,
                'reset': True,
                'gpu_layers': self.gpu_layers
            }
        )


class MilvusManager:
    def __init__(self, host, port, user, password):
        self.host = host
        self.port = port
        self.user = user
        self.password = password

    def create_connection(self):
        connections.connect(
            alias="default",
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port
        )

    def get_query_result(self, query, collection_name, embed_model):
        try:
            self.create_connection()
            collections = utility.list_collections()
            if collection_name in collections:
                vector_db = Milvus(embed_model, collection_name)
                docs = vector_db.similarity_search(query, k=5)
                output = ""
                for doc in docs:
                    output += doc.page_content + "\n" + str(doc.metadata) + "\n" + "\n" + "\n"
                return output
        except Exception as e:
            print(e)
            return "", ""

class EmbeddingModelManager:
    def embed_model(self, model_name, model_kwargs, encode_kwargs):
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

class RetrieverManager:
    @staticmethod
    def get_retriever_text(llm, vector_store, query):
        vector_db = Milvus(llm, vector_store)
        document_content_description = "Brief summary of a topic"
        retriever = SelfQueryRetriever.from_llm(
            llm, vector_db, document_content_description, verbose=True
        )
        response = retriever.get_relevant_documents(query)
        print('response: ', response)
        return response

    @staticmethod
    def get_qa_retriever_text(llm, model, vector_store, query):
        try:
            vector_db = Milvus(model, vector_store)
            qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_db.as_retriever(),
                                             return_source_documents=True)
            result = qa({"query": query})
            response = result['result']
        except Exception as e:
            print(e)
            response = f"ERROR- {str(e)}"
        return response

class FileReader:
    @staticmethod
    def check_extension(file_path):
        if file_path.endswith('.txt'):
            print("File extension is .txt")
            return "txt"
        elif file_path.endswith('.pdf'):
            print("File extension is .pdf")
            return "pdf"
        else:
            raise ValueError("File extension is not .txt, .pdf")

    @staticmethod
    def file_reader(file_path, chunk_size, chunk_overlap):
        extension = FileReader.check_extension(file_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        if extension == "txt":
            loader = TextLoader(file_path)
            documents = loader.load()
            docs = text_splitter.split_documents(documents)
            return docs

        elif extension == "pdf":
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            docs = text_splitter.split_documents(documents)
            return docs



