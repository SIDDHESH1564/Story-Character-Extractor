from langchain.embeddings import MistralAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from typing import List, Optional, Union
import os

class VectorDB:
    """Vector database manager using FAISS and MistralAI embeddings"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize the vector database with MistralAI embeddings.
        
        Args:
            api_key (str, optional): MistralAI API key. If not provided, will look for MISTRAL_API_KEY in env.
        """
        self.db_path = "vector_store.faiss"
        # Initialize MistralAI embeddings as required in the assignment
        self.embeddings = MistralAIEmbeddings(api_key=api_key)
        self.db = None

    def _ensure_documents(self, documents: List[Union[str, Document]]) -> List[Document]:
        """
        Ensure all documents are in Document format.
        
        Args:
            documents: List of strings or Document objects
            
        Returns:
            List of Document objects
        """
        return [
            doc if isinstance(doc, Document) else Document(page_content=doc)
            for doc in documents
        ]

    def create_vector_db(self, documents: List[Union[str, Document]]) -> bool:
        """
        Create a new FAISS vector database from documents.
        
        Args:
            documents: List of documents to embed
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not documents:
                raise ValueError("No documents provided for initialization")
            
            docs = self._ensure_documents(documents)
            self.db = FAISS.from_documents(docs, self.embeddings)
            return self.save_vector_db()
        except Exception as e:
            print(f"Error creating vector database: {str(e)}")
            return False

    def save_vector_db(self) -> bool:
        """
        Save the FAISS vector database to disk.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.db is None:
                raise ValueError("No database to save")
            
            self.db.save_local(self.db_path)
            print(f"Successfully saved vector database to {self.db_path}")
            return True
        except Exception as e:
            print(f"Error saving vector database: {str(e)}")
            return False

    def load_vector_db(self) -> bool:
        """
        Load the FAISS vector database from disk.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not os.path.exists(self.db_path):
                raise FileNotFoundError(f"No vector database found at {self.db_path}")
            
            self.db = FAISS.load_local(
                self.db_path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"Successfully loaded vector database from {self.db_path}")
            return True
        except Exception as e:
            print(f"Error loading vector database: {str(e)}")
            return False

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Perform similarity search in the vector database.
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of similar documents
            
        Raises:
            ValueError: If database is not initialized
        """
        if self.db is None:
            raise ValueError("Vector database not initialized. Please create or load it first.")
        
        return self.db.similarity_search(query, k=k)

    def clear_vector_db(self) -> bool:
        """
        Delete the vector database file and reset the instance.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            self.db = None
            return True
        except Exception as e:
            print(f"Error clearing vector database: {str(e)}")
            return False