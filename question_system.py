from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter


def setup_qa_system(file_path):

    loader = PyPDFLoader(file_path)
    docs = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()

    vector_store = FAISS.from_documents(chunks, embeddings)

    retriever = vector_store.as_retriever()

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

    return qa_chain



