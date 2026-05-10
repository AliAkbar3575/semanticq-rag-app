import os
from dotenv import load_dotenv
import streamlit as st

from langchain_groq import ChatGroq
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


#--------------------------------initiate LLM--------------------------------

def initiate_llm():
    load_dotenv()  # Load environment variables from .env file

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        api_key=os.getenv("GROQ_API_KEY")
    )
    return llm

#--------------------------------loading data--------------------------------

def load_data():
    loader = CSVLoader("faqs.csv", source_column="prompt")
    data = loader.load()
    return data

#--------------------------------embedding and vector store--------------------------------

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "my-rag-index"

def embedding_and_vectorstore():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Create index if it does not exist
    existing_indexes = [index.name for index in pc.list_indexes()]

    if INDEX_NAME not in existing_indexes:

        pc.create_index(
            name=INDEX_NAME,
            dimension=384,   # all-MiniLM-L6-v2 => 384 dimensions
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    # Connect to index
    index = pc.Index(INDEX_NAME)

    # Create vector store
    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings
    )

    # Check whether vectors already exist
    stats = index.describe_index_stats()

    total_vectors = stats.get("total_vector_count", 0)

    # Upload documents only if empty
    if total_vectors == 0:

        data = load_data()

        vectorstore.add_documents(data)

        print("Documents uploaded to Pinecone.")

    else:
        print("Pinecone index already contains vectors.")

    # Create retriever
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    return retriever


#-------------------------------prompt template and chain--------------------------------

def create_chain(llm, retriever):
      
    prompt = ChatPromptTemplate.from_template(
            """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question} """
        )

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

#--------------------main function to run the chain----------------------

if __name__ == "__main__":


    st.title("SemanticQ 🌱")
    # st.caption("Ask questions about our courses and get accurate answers based on the course content!")

    st.markdown("<h6>Ask questions about our courses and get accurate answers based on the course content! 😀</h6>", unsafe_allow_html=True)


    llm = initiate_llm()
    retriever = embedding_and_vectorstore()
    rag_chain = create_chain(llm, retriever)


    # question = st.text_input("Question: ")

    st.header("Ask your question")
    question = st.text_input("")

    if question:

        # st.spinner("Generating answer...")

        answer = rag_chain.invoke(question)

        st.header("Answer")
        st.write(answer)

        # rdocs = retriever.invoke(question)


        # st.header("Retrieved Context...")

        # for doc in rdocs:
        #     st.write(doc.page_content)
