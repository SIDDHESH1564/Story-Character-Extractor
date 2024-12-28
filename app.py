import streamlit as st
import os
from process_stories import StoryProcessor
from typing import Optional

def initialize_processor() -> Optional[StoryProcessor]:
    """Initialize the StoryProcessor with API key."""
    try:
        # Get API key from environment or Streamlit secrets
        api_key = os.getenv('MISTRAL_API_KEY') or st.secrets.get('MISTRAL_API_KEY')
        if not api_key:
            st.error("MistralAI API key not found. Please set MISTRAL_API_KEY in environment or Streamlit secrets.")
            return None
        
        return StoryProcessor(api_key=api_key)
    except Exception as e:
        st.error(f"Error initializing processor: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="Story Character Extractor",
        page_icon="📚",
        layout="wide"
    )

    st.title("📚 Story Character Extractor")
    st.markdown("""
    This application helps you extract structured information about characters from stories.
    Upload your story files and search for character details.
    """)

    # Initialize processor
    processor = initialize_processor()
    if not processor:
        st.stop()

    # Create two columns for layout
    col1, col2 = st.columns(2)

    with col1:
        st.header("📤 Upload Stories")
        uploaded_files = st.file_uploader(
            "Upload .txt files containing stories",
            accept_multiple_files=True,
            type="txt",
            help="Select one or more .txt files containing story text"
        )

        if uploaded_files:
            st.write("📋 Uploaded files:")
            for file in uploaded_files:
                st.write(f"- {file.name}")

            if st.button("🔄 Process Stories", help="Compute embeddings for the uploaded stories"):
                with st.spinner("Processing stories..."):
                    try:
                        message = processor.compute_embeddings(uploaded_files)
                        if "successfully" in message.lower():
                            st.success(message)
                        else:
                            st.error(message)
                    except Exception as e:
                        st.error(f"Error processing stories: {str(e)}")

    with col2:
        st.header("🔍 Search Character Information")
        character_name = st.text_input(
            "Enter character name",
            help="Type the name of a character from the uploaded stories"
        )

        if st.button("🔎 Get Character Info", disabled=not character_name):
            if character_name:
                with st.spinner("Searching for character information..."):
                    try:
                        result = processor.get_character_info(character_name)
                        
                        if "error" in result:
                            st.error(result["error"])
                        else:
                            # Display character information in a structured way
                            st.subheader(f"📖 {result['name']}")
                            st.write(f"**Story:** {result['storyTitle']}")
                            st.write(f"**Character Type:** {result['characterType']}")
                            
                            st.write("**Summary:**")
                            st.info(result['summary'])
                            
                            st.write("**Relationships:**")
                            if result['relations']:
                                for relation in result['relations']:
                                    st.write(f"- {relation['name']}: {relation['relation']}")
                            else:
                                st.write("No relationships found")
                            
                            # Show raw JSON in expander
                            with st.expander("View Raw JSON"):
                                st.json(result)
                    except Exception as e:
                        st.error(f"Error retrieving character information: {str(e)}")

    # Add footer with cleanup option
    st.divider()
    if st.button("🗑️ Clear All Data", help="Remove all processed stories and embeddings"):
        try:
            if processor.clean_up():
                st.success("All data cleared successfully")
            else:
                st.error("Error clearing data")
        except Exception as e:
            st.error(f"Error during cleanup: {str(e)}")

if __name__ == "__main__":
    main()