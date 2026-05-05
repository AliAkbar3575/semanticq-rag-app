import logging
import config
from data_loader import create_vector_database

from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

# ---------------------------- formatting docs -----------------------------

def format_docs(docs):

    try:
        formatted = "\n\n".join([doc.page_content for doc in docs])
        logger.debug(f"Formatted {len(docs)} documents into context")
        return formatted
        
    except Exception as e:
        logger.error(f"Error formatting documents: {str(e)}")
        raise

# ---------------------------- creating RAG chain -----------------------------

def create_rag_chain():

    try:
        logger.info("🔃 Creating RAG chain...")
        logger.info("🔃 Initializing large language model...")

        llm = ChatGroq(
            model=config.GROQ_MODEL,
            temperature=config.TEMPERATURE,
            api_key=config.GROQ_API_KEY
        )
        logger.info("✅ LLM initialized successfully")

        logger.info("🔃 Setting up retriever...")
        retriever = create_vector_database()
        logger.info("✅ Retriever obtained successfully")

        logger.info("🔃 Creating RAG chain using prompt template...")
        prompt = ChatPromptTemplate.from_template("""
            Given the following context and a question, generate an answer based on this context only.
            In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
            If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

            CONTEXT: {context}

            QUESTION: {question}
        """)

        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        
        logger.info("✅ RAG chain created successfully")
        return rag_chain


    except Exception as e:
        logger.error(f"Error creating RAG chain: {str(e)}")
        raise