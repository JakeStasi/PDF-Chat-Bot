from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter


def setup_qa_system(file_path):
    # Reads the text from the pdf
    loader = PyPDFLoader(file_path)
    docs = loader.load_and_split()

    # Spilts the texts into chunks containing 1000 characters and neighboring chunk overlapping with 200 characters
    # Overlapping helps to not lose important information or cut off important context
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    # Converts chunks into embeddings which are numerical representations of the words so the computer can work with them
    embeddings = OpenAIEmbeddings()

    #Store the different chunk embeddings into a vector database
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Searches the vector store to find chunks that are similar to the question
    retriever = vector_store.as_retriever()

    # Creates the Open AI LLM 
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    # This creates the RAG Chain which retrieves the relevant chunks and prompts the LLM to answer the user's question with the chunks as context so it can better answer the question 
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

    return qa_chain



