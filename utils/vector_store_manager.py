"""
Vector Store Manager for the Jain Learning Ecosystem
"""
import logging
from pathlib import Path
from utils.vector_store import create_vector_store, load_vector_store

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Manager class for vector store operations"""

    def __init__(self, config):
        self.config = config
        self.vector_stores = {}  # Store vector stores by sect and language
        logger.info("VectorStoreManager initialized")

    def _get_store_key(self, sect, language):
        """Generate a key for storing/retrieving vector stores"""
        return f"{sect}_{language}"

    def _get_persist_directory(self, sect, language):
        """Get the persistence directory for a specific sect and language"""
        base_dir = self.config.get('vector_db', {}).get('persist_directory', './data/output')
        prefix = self.config.get('vector_db', {}).get('collection_prefix', 'jain_learning')
        return f"{base_dir}/{prefix}_{sect}_{language}"

    def add_documents(self, documents, metadatas, ids, sect, language):
        """Add documents to the vector store"""
        try:
            store_key = self._get_store_key(sect, language)
            persist_dir = self._get_persist_directory(sect, language)

            # Create directory if it doesn't exist
            Path(persist_dir).mkdir(parents=True, exist_ok=True)

            # Try to load existing vector store or create new one
            if store_key in self.vector_stores:
                vector_store = self.vector_stores[store_key]
                # Append to existing store
                logger.info(f"Appending {len(documents)} documents to existing vector store")
                vector_store = create_vector_store(
                    chunks=documents,
                    config=self.config,
                    pdf_paths=[m.get('file_path', 'unknown.pdf') for m in metadatas],
                    output_dir=persist_dir,
                    author=metadatas[0].get('author', 'Unknown') if metadatas else 'Unknown',
                    append=True
                )
            else:
                # Create new vector store
                logger.info(f"Creating new vector store with {len(documents)} documents")
                vector_store = create_vector_store(
                    chunks=documents,
                    config=self.config,
                    pdf_paths=[m.get('file_path', 'unknown.pdf') for m in metadatas],
                    output_dir=persist_dir,
                    author=metadatas[0].get('author', 'Unknown') if metadatas else 'Unknown',
                    append=False
                )

            if vector_store:
                self.vector_stores[store_key] = vector_store
                logger.info(f"Successfully added documents to vector store for {sect}/{language}")
                return True
            else:
                logger.error("Failed to create or update vector store")
                return False

        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            return False

    def get_vector_store(self, sect, language):
        """Get the vector store for a specific sect and language"""
        try:
            store_key = self._get_store_key(sect, language)
            logger.info(f"Attempting to get vector store for {sect}/{language} (key: {store_key})")

            # Return cached store if available
            if store_key in self.vector_stores:
                logger.info(f"Found cached vector store for {sect}/{language}")
                return self.vector_stores[store_key]

            # Try to load from disk
            persist_dir = self._get_persist_directory(sect, language)
            logger.info(f"Loading vector store from disk: {persist_dir}")
            vector_store = load_vector_store(persist_dir)

            if vector_store:
                self.vector_stores[store_key] = vector_store
                logger.info(f"Successfully loaded and cached vector store for {sect}/{language}")
                return vector_store

            logger.warning(f"No vector store found for {sect}/{language} at {persist_dir}")
            return None

        except Exception as e:
            logger.error(f"Error getting vector store: {str(e)}")
            return None

    def get_collection_stats(self, sect, language, religion='jainism'):
        """Get statistics about the vector store collection"""
        try:
            vector_store = self.get_vector_store(sect, language)
            if not vector_store:
                return {
                    'document_count': 0,
                    'has_content': False
                }

            # Get all documents
            try:
                data = vector_store.get()
                doc_count = len(data.get('ids', [])) if data else 0

                return {
                    'document_count': doc_count,
                    'has_content': doc_count > 0,
                    'sect': sect,
                    'language': language
                }
            except Exception as e:
                logger.error(f"Error getting collection data: {str(e)}")
                return {
                    'document_count': 0,
                    'has_content': False
                }

        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {
                'document_count': 0,
                'has_content': False
            }

    def get_unique_metadata_values(self, sect, language, religion, metadata_key):
        """Get unique values for a metadata field"""
        try:
            vector_store = self.get_vector_store(sect, language)
            if not vector_store:
                return []

            try:
                data = vector_store.get()
                if not data or not data.get('metadatas'):
                    return []

                # Extract unique values
                values = set()
                for metadata in data['metadatas']:
                    if metadata_key in metadata:
                        values.add(metadata[metadata_key])

                return list(values)
            except Exception as e:
                logger.error(f"Error getting unique metadata values: {str(e)}")
                return []

        except Exception as e:
            logger.error(f"Error in get_unique_metadata_values: {str(e)}")
            return []

    def query_vector_store(self, query, sect, language, religion='jainism', k=5):
        """Query the vector store for similar documents"""
        try:
            vector_store = self.get_vector_store(sect, language)
            if not vector_store:
                return []

            # Import the query function from vector_store module
            from utils.vector_store import query_vector_store as vs_query
            results = vs_query(query, vector_store, k=k)

            # Format results with relevance scores
            formatted_results = []
            for doc in results:
                formatted_results.append({
                    'content': doc.page_content if hasattr(doc, 'page_content') else str(doc),
                    'metadata': doc.metadata if hasattr(doc, 'metadata') else {},
                    'relevance_score': 0.8  # Default score
                })

            return formatted_results

        except Exception as e:
            logger.error(f"Error querying vector store: {str(e)}")
            return []
