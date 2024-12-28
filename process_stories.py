from langchain.llms import MistralAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from typing import List, Dict, Any
import json
from vector_db import VectorDB
class StoryProcessor:
    """Handles story processing and character information extraction"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize the story processor.
        
        Args:
            api_key (str, optional): MistralAI API key. If not provided, will look for MISTRAL_API_KEY in env.
        """
        self.vector_db = VectorDB(api_key=api_key)
        self.llm = MistralAI(api_key=api_key)
        self._setup_prompt_template()

    def _setup_prompt_template(self):
        """Set up the prompt template for character information extraction"""
        self.prompt_template = PromptTemplate(
            template="""
            Extract structured details about the character "{character}" from the following story context.
            Focus only on information that is explicitly mentioned in the story.
            
            Required details:
            1. Character's full name
            2. Title of the story they appear in
            3. A concise summary of their role and actions in the story
            4. Their relationships with other characters
            5. Their character type (protagonist, antagonist, or supporting character)
            
            Respond only with valid JSON in this exact format:
            {{
                "name": "Character's name",
                "storyTitle": "Story title",
                "summary": "Brief character summary",
                "relations": [
                    {{"name": "Related character name", "relation": "Relationship description"}}
                ],
                "characterType": "Character type"
            }}
            
            If the character is not found in the story, respond with:
            {{"error": "Character not found in the stories"}}
            
            Context: {context}
            Character to analyze: {character}
            """,
            input_variables=["context", "character"]
        )

    def compute_embeddings(self, files: List[Any]) -> str:
        """
        Process story files and compute their embeddings.
        
        Args:
            files: List of file objects containing story text
            
        Returns:
            str: Success or error message
            
        Raises:
            ValueError: If no valid files are provided
        """
        try:
            if not files:
                raise ValueError("No files provided")

            documents = []
            for file in files:
                try:
                    content = file.read().decode("utf-8")
                    title = file.name.replace(".txt", "")
                    documents.append(Document(
                        page_content=content,
                        metadata={"title": title}
                    ))
                except Exception as e:
                    print(f"Error processing file {file.name}: {str(e)}")
                    continue

            if not documents:
                raise ValueError("No valid documents found in provided files")

            # Split documents into chunks
            text_splitter = CharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separator="\n"
            )
            texts = text_splitter.split_documents(documents)

            # Create and save vector database
            success = self.vector_db.create_vector_db(texts)
            
            if success:
                return "Embeddings computed and stored successfully"
            else:
                return "Error storing embeddings"

        except Exception as e:
            return f"Error computing embeddings: {str(e)}"

    def get_character_info(self, character_name: str) -> Dict[str, Any]:
        """
        Extract information about a character from the processed stories.
        
        Args:
            character_name: Name of the character to search for
            
        Returns:
            dict: Character information in JSON format
            
        Raises:
            ValueError: If vector database is not initialized
        """
        try:
            # Load vector database if not already loaded
            if self.vector_db.db is None:
                success = self.vector_db.load_vector_db()
                if not success:
                    raise ValueError("Failed to load vector database")

            # Create retrieval chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_db.db.as_retriever(
                    search_kwargs={"k": 4}
                ),
                chain_type_kwargs={
                    "prompt": self.prompt_template
                }
            )

            # Get response
            response = qa_chain.run(character_name)

            # Parse response to ensure valid JSON
            try:
                result = json.loads(response)
                return result
            except json.JSONDecodeError:
                return {"error": "Failed to parse character information"}

        except Exception as e:
            return {"error": f"Error getting character information: {str(e)}"}

    def clean_up(self) -> bool:
        """
        Clean up resources and temporary files.
        
        Returns:
            bool: True if cleanup successful, False otherwise
        """
        return self.vector_db.clear_vector_db()