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

    # Converts chunks into embeddings which are numerical representations of the words, where each chunk "basketball is played 10 players on the court..."
    # is converted to a embedding [1,.4,.6,.8,1.2] where the numbers represent what the words mean
    embeddings = OpenAIEmbeddings()

    # Stores the different chunk embeddings into a vector database called FAISS which can store these embeddings and quickly look up semantically similar embeddings
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Searches the vector database FAISS to find chunks that are similar to the question
    # and then later it is given to the LLM to use as context so it can better answer the user's question
    retriever = vector_store.as_retriever()

    # Creates the Open AI LLM 
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    # This creates the Retrieval-Augmented Generation Chain which retrieves the relevant chunks and prompts the LLM to answer the user's question with the chunks as context so it can better answer the question 
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

    return qa_chain



