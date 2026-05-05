import os
import sys
import logging
import config
from datetime import datetime
import streamlit as st
from rag_chain import create_rag_chain


def setup_logging():

    os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.LOG_FILE),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Log startup
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("SemanticQ Application Starting...")
    logger.info(f"Log file: {config.LOG_FILE}")
    logger.info("=" * 50)

setup_logging()
logger = logging.getLogger(__name__)


@st.cache_resource
def get_chain():

    try:
        logger.info("🔃 Initializing RAG chain from cache...")
        rag_chain = create_rag_chain()
        logger.info("✅ RAG chain initialized successfully from cache.\n")
        return rag_chain
    
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG chain: {str(e)}")
        st.error(f"Failed to initialize the system: {str(e)}")
        st.stop()

def main():
        
    try:

        st.title("SemanticQ 🌱")
        st.markdown("<h6>Ask questions about our courses and get accurate answers based on the course content! 😀</h6>", unsafe_allow_html=True)

        chain = get_chain()

        st.header("Ask your question")
        question = st.text_input("Enter here...")

        if question:

            logger.info(f"🔃 User question: {question}")

            answer = chain.invoke(question)

            st.header("Answer")
            st.write(answer)

    except Exception as e:
        logger.error(f"❌ An error occurred in the main application: {str(e)}")
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()