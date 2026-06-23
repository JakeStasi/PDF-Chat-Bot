I wanted to create a PDF chatbot that can read in the documents I give it and the help summarize the document and then give specific answers about the document when I ask it questions. I found myself using an online website that does this and I wanted to build my own.
I used a basic html stlying, ChatGPT API for my llm, and LangChain for this project. 

I load the PDF in and extract the text from the file. I split the large text into smaller chunks. This step is necessary because you can't send huge documents directly to the model. 

I convert each chunk into a numerical vector that represents the semantic meaning of the text, so chunks with similar meaning will have similar numerical values/vector representations. This makes it easier for the model to find similar meaning texts. 

I store the vectors in FAISS which acts as my vector database. When the user asks a question, it is also converted to a vector representation and FAISS finds similar chunk vector representations with the question and retrieves them by comparing the chunk vector to the question vector.
This finds the most semantically similar chunk to the question.

Those relevant chunks are then passed into the language model through a retrieval QA chain. The LLM uses the retrieved document context to generate an answer. So instead of the model only relying on its general training data,
it answers based on the specific PDF content that was retrieved.

I created a basic html website that shows the PDF on the left side and then has conversation box with the LLM on right side.
