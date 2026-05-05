import logging
import os
import config

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)

# ---------------------------- loading documents -----------------------------

def load_documents():
        
    try:
        logger.info("🔃 Loading documents from CSV file...")
        loader = CSVLoader(file_path=config.DATA_PATH)
        documents = loader.load()
        logger.info(f"✅ Successfully loaded {len(documents)} documents.\n")

        return documents
    
    except FileNotFoundError:
        logger.error(f"❌ File not found at {config.DATA_PATH}")
        raise

    except Exception as e:
        logger.error(f"❌ Error loading documents: {str(e)}")
        raise

# ---------------------------- creating embeddings -----------------------------

def create_embeddings():

    try:
        logger.info("🔃 Creating embeddings for documents...")
        embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
        logger.info("✅ Successfully created embeddings.\n")

        return embeddings
    
    except Exception as e:
        logger.error(f"❌ Error creating embeddings: {str(e)}")
        raise

# ---------------------------- creating vector database -----------------------------   

def create_vector_database():

    try:
        embeddings = create_embeddings()

        if os.path.exists(config.VECTOR_DB_PATH):
            logger.info(f"🔃 Loading existing vector store from {config.VECTOR_DB_PATH}...")
            
            vector_db = FAISS.load_local(
                config.VECTOR_DB_PATH,
                embeddings,
                allow_dangerous_deserialization=True)
            
            logger.info("✅ Successfully loaded existing vector store.\n")
            
        else:
            logger.info(f"⚠️ No existing vector store found at {config.VECTOR_DB_PATH}, creating a new one...")
            documents = load_documents()
            vector_db = FAISS.from_documents(documents, embeddings)
            vector_db.save_local(config.VECTOR_DB_PATH)
            logger.info(f"✅ Successfully created and saved vector store at {config.VECTOR_DB_PATH}.\n")        
            
        return vector_db.as_retriever(search_kwargs={"k": config.SEARCH_K})
    
    except Exception as e:
        logger.error(f"❌ Error creating vector store: {str(e)}")
        raise

