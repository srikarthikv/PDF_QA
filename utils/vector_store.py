from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import logging
import os
import shutil
import traceback
from PyPDF2 import PdfReader
from typing import List, Dict, Optional

# Disable ChromaDB telemetry to avoid telemetry errors
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

logger = logging.getLogger(__name__)

def extract_page_number(metadata, default_index, pdf_path):
    """Extract page number from metadata or PDF, falling back to default index if unavailable."""
    if metadata and 'page' in metadata:
        return str(metadata['page'])
    try:
        if pdf_path and pdf_path != "unknown.pdf":
            pdf_reader = PdfReader(pdf_path)
            page_count = len(pdf_reader.pages)
            return str(default_index % page_count + 1)  # 1-based index within PDF page count
    except Exception as e:
        logger.warning(f"Failed to extract page number from {pdf_path}: {str(e)}")
    return str(default_index + 1)  # Fallback to sequential numbering

def create_vector_store(chunks, config, pdf_paths, output_dir, author="Unknown Author", append=False):
    """
    Create or append to a vector store with the given chunks and metadata.

    Args:
        chunks (list): List of text chunks to embed.
        config (dict): Configuration dictionary containing model settings.
        pdf_paths (list): List of PDF file paths associated with the chunks.
        output_dir (str): Directory to persist the vector store.
        author (str): Author of the PDFs (default: "Unknown Author").
        append (bool): If True, append new chunks to an existing vector store; if False, create a new one.

    Returns:
        Chroma: The vector store instance, or None if creation fails.
    """
    try:
        if not chunks:
            logger.error("No chunks provided to create vector store")
            return None

        logger.info(f"Creating vector store with {len(chunks)} chunks at {output_dir}")

        # Initialize embeddings with intfloat/multilingual-e5-base
        model_name = "intfloat/multilingual-e5-base"
        logger.info(f"Load pretrained SentenceTransformer: {model_name}")
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        # Determine dimension by encoding a sample text
        sample_text = "Sample text to determine embedding dimension"
        sample_embedding = embeddings.embed_query(sample_text)
        dimension = len(sample_embedding)
        logger.info(f"Embeddings initialized successfully with dimension {dimension}")

        # Create metadatas list with proper filenames
        metadatas = []
        for i, chunk in enumerate(chunks):
            pdf_path = pdf_paths[i % len(pdf_paths)] if pdf_paths and len(pdf_paths) > 0 else "unknown.pdf"
            pdf_filename = os.path.basename(pdf_path)
            page_num = extract_page_number({"page": i}, i, pdf_path)
            metadata = {
                "source": pdf_filename,  # Store original filename
                "pdf_path": str(pdf_path),
                "author": str(author),
                "page_number": page_num,
                "chunk_index": i
            }
            metadatas.append(metadata)

        # Check and handle existing collection dimensionality
        os.makedirs(output_dir, exist_ok=True)

        if append and os.path.exists(os.path.join(output_dir, "chroma-collections")):
            try:
                vector_store = Chroma(persist_directory=output_dir, embedding_function=embeddings)
                existing_dims = vector_store._collection.get_dimension() if vector_store.get()['ids'] else None
                if existing_dims and existing_dims != dimension:
                    logger.warning(f"Dimensionality mismatch: Existing {existing_dims} vs new {dimension}. Recreating collection.")
                    shutil.rmtree(output_dir, ignore_errors=True)
                    os.makedirs(output_dir, exist_ok=True)
                    vector_store = Chroma.from_texts(
                        texts=chunks,
                        embedding=embeddings,
                        metadatas=metadatas,
                        persist_directory=output_dir
                    )
                else:
                    existing_ids = vector_store.get()['ids']
                    new_chunk_index = len(existing_ids) if existing_ids else 0
                    # Update metadata with correct page numbers
                    updated_metadatas = []
                    for i, m in enumerate(metadatas):
                        new_metadata = m.copy()
                        new_metadata["page_number"] = extract_page_number(m, new_chunk_index + i, m["pdf_path"])
                        updated_metadatas.append(new_metadata)
                    vector_store.add_texts(texts=chunks, metadatas=updated_metadatas)
                    logger.info(f"Appended {len(chunks)} new chunks to vector store at {output_dir}")
            except Exception as e:
                logger.error(f"Failed to append to existing vector store: {str(e)}, Traceback: {traceback.format_exc()}")
                shutil.rmtree(output_dir, ignore_errors=True)
                os.makedirs(output_dir, exist_ok=True)
                vector_store = Chroma.from_texts(
                    texts=chunks,
                    embedding=embeddings,
                    metadatas=metadatas,
                    persist_directory=output_dir
                )
        else:
            logger.info(f"Creating new vector store at {output_dir} with permissions check")
            test_file = os.path.join(output_dir, "test_write.txt")
            try:
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
            except PermissionError as e:
                logger.error(f"Permission denied when writing to {output_dir}: {str(e)}")
                return None
            except Exception as e:
                logger.error(f"File system error when testing write to {output_dir}: {str(e)}")
                return None

            try:
                vector_store = Chroma.from_texts(
                    texts=chunks,
                    embedding=embeddings,
                    metadatas=metadatas,
                    persist_directory=output_dir
                )
                logger.info(f"Created new vector store at {output_dir}")
            except Exception as e:
                logger.error(f"Failed to create vector store due to index error: {str(e)}, Traceback: {traceback.format_exc()}")
                shutil.rmtree(output_dir, ignore_errors=True)
                return None

        logger.info(f"Vector store automatically persisted to {output_dir}")
        return vector_store

    except Exception as e:
        logger.error(f"Error creating vector_store: {str(e)}, Traceback: {traceback.format_exc()}")
        return None

def load_vector_store(persist_directory):
    """
    Load an existing vector store from the specified directory.

    Args:
        persist_directory (str): Directory where the vector store is persisted.

    Returns:
        Chroma: The loaded vector store instance, or None if loading fails.
    """
    try:
        logger.info(f"Loading pretrained SentenceTransformer for vector store")
        embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-base",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        sample_text = "Sample text to determine embedding dimension"
        sample_embedding = embeddings.embed_query(sample_text)
        dimension = len(sample_embedding)
        logger.info(f"Attempting to load vector store from {persist_directory} with dimension {dimension}")

        # Check if persist directory exists and has content
        if not os.path.exists(persist_directory):
            logger.warning(f"Persist directory does not exist: {persist_directory}")
            return None

        # Check for any of the possible Chroma database files
        chroma_files = ["chroma.sqlite3", "chroma-collections", "chroma-embeddings"]
        has_chroma_data = any(os.path.exists(os.path.join(persist_directory, f)) for f in chroma_files)

        if not has_chroma_data:
            # Check if directory has any files
            dir_contents = os.listdir(persist_directory) if os.path.exists(persist_directory) else []
            logger.warning(f"No Chroma database files found in {persist_directory}. Contents: {dir_contents}")
            return None

        # Try to load the vector store
        try:
            vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

            # Check if the vector store has any data
            try:
                data = vector_store.get()
                if data and data.get('ids') and len(data['ids']) > 0:
                    # Check dimensionality if possible
                    try:
                        store_dimension = vector_store._collection.get_dimension() if hasattr(vector_store, '_collection') else dimension
                        if store_dimension != dimension:
                            logger.warning(f"Dimensionality mismatch: expected {dimension}, got {store_dimension}")
                            return None
                    except:
                        # If we can't check dimensionality, assume it's okay
                        pass

                    logger.info(f"Successfully loaded vector store from {persist_directory} with {len(data['ids'])} documents")
                    return vector_store
                else:
                    logger.warning(f"Loaded vector store from {persist_directory} is empty")
                    return None
            except Exception as e:
                logger.error(f"Error checking vector store data: {str(e)}")
                return None
        except Exception as e:
            logger.error(f"Error loading Chroma instance: {str(e)}")
            return None

    except Exception as e:
        logger.error(f"Error loading vector_store: {str(e)}, Traceback: {traceback.format_exc()}")
        return None

def query_vector_store(query_text, vector_store, k=50):
    """
    Query the vector store for similar documents.
    Optimized for large-scale book collections (10-15 books with ~2000 pages each).

    Args:
        query_text (str): The query text to search for.
        vector_store (Chroma): The vector store instance to query.
        k (int): Number of similar documents to return (default 50 for comprehensive retrieval).

    Returns:
        list: List of similar documents with relevance scores.
    """
    if not vector_store:
        logger.error("Vector store not initialized")
        return []

    try:
        # Use similarity_search_with_relevance_scores for better ranking
        results_with_scores = vector_store.similarity_search_with_relevance_scores(query_text, k=k)

        # Filter by relevance threshold (0.5 = 50% similarity)
        filtered_results = [
            doc for doc, score in results_with_scores
            if score >= 0.5
        ]

        # If we got too few results, lower the threshold
        if len(filtered_results) < 10 and len(results_with_scores) > 0:
            logger.info(f"Only {len(filtered_results)} results above 0.5 threshold, using top {min(k, len(results_with_scores))} results")
            filtered_results = [doc for doc, score in results_with_scores[:min(k, len(results_with_scores))]]

        logger.info(f"Found {len(filtered_results)} relevant documents from {len(results_with_scores)} total matches")
        return filtered_results
    except Exception as e:
        # Fallback to basic similarity search if relevance scores not available
        logger.warning(f"Relevance score search failed, using basic similarity search: {str(e)}")
        try:
            results = vector_store.similarity_search(query_text, k=k)
            logger.info(f"Found {len(results)} similar documents")
            return results
        except Exception as e2:
            logger.error(f"Error querying vector_store: {str(e2)}, Traceback: {traceback.format_exc()}")
            return []

def create_vector_store_batch(book_chunks: List[Dict], config: dict, output_dir: str, batch_size: int = 1000):
    """
    Create vector store for large-scale books with batch processing.
    Optimized for 10-15 books with ~2000 pages each.

    Args:
        book_chunks: List of dictionaries containing chunks from multiple books
                     Each dict should have: {'chunks': [...], 'pdf_path': '...', 'author': '...'}
        config: Configuration dictionary
        output_dir: Directory to persist the vector store
        batch_size: Number of chunks to process at once (default 1000)

    Returns:
        Chroma: The vector store instance, or None if creation fails
    """
    try:
        logger.info(f"Creating vector store for {len(book_chunks)} books with batch processing")

        # Initialize embeddings
        model_name = "intfloat/multilingual-e5-base"
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        # Prepare all chunks and metadata
        all_chunks = []
        all_metadatas = []

        for book_idx, book_data in enumerate(book_chunks):
            chunks = book_data.get('chunks', [])
            pdf_path = book_data.get('pdf_path', 'unknown.pdf')
            author = book_data.get('author', 'Unknown Author')
            pdf_filename = os.path.basename(pdf_path)

            logger.info(f"Processing book {book_idx + 1}/{len(book_chunks)}: {pdf_filename} ({len(chunks)} chunks)")

            for i, chunk in enumerate(chunks):
                page_num = extract_page_number({"page": i}, i, pdf_path)
                metadata = {
                    "source": pdf_filename,
                    "pdf_path": str(pdf_path),
                    "author": str(author),
                    "page_number": page_num,
                    "chunk_index": i,
                    "book_index": book_idx
                }
                all_chunks.append(chunk)
                all_metadatas.append(metadata)

        total_chunks = len(all_chunks)
        logger.info(f"Total chunks to process: {total_chunks}")

        # Create vector store directory
        os.makedirs(output_dir, exist_ok=True)

        # Process in batches to avoid memory issues
        vector_store = None
        for batch_start in range(0, total_chunks, batch_size):
            batch_end = min(batch_start + batch_size, total_chunks)
            batch_chunks = all_chunks[batch_start:batch_end]
            batch_metadatas = all_metadatas[batch_start:batch_end]

            logger.info(f"Processing batch {batch_start // batch_size + 1}: chunks {batch_start}-{batch_end}")

            if vector_store is None:
                # Create initial vector store
                vector_store = Chroma.from_texts(
                    texts=batch_chunks,
                    embedding=embeddings,
                    metadatas=batch_metadatas,
                    persist_directory=output_dir
                )
            else:
                # Append to existing vector store
                vector_store.add_texts(texts=batch_chunks, metadatas=batch_metadatas)

        logger.info(f"Successfully created vector store with {total_chunks} chunks from {len(book_chunks)} books")
        return vector_store

    except Exception as e:
        logger.error(f"Error creating batch vector store: {str(e)}, Traceback: {traceback.format_exc()}")
        return None

def get_vector_store_stats(vector_store) -> Optional[Dict]:
    """
    Get statistics about the vector store for large book collections.

    Args:
        vector_store: The Chroma vector store instance

    Returns:
        Dictionary with statistics about books, chunks, and pages
    """
    try:
        if not vector_store:
            return None

        # Get all documents
        all_docs = vector_store.get()

        if not all_docs or not all_docs.get('metadatas'):
            return {
                'total_chunks': 0,
                'total_books': 0,
                'books': []
            }

        # Analyze metadata
        books = {}
        for metadata in all_docs['metadatas']:
            source = metadata.get('source', 'Unknown')
            if source not in books:
                books[source] = {
                    'name': source,
                    'author': metadata.get('author', 'Unknown'),
                    'chunks': 0,
                    'pages': set()
                }
            books[source]['chunks'] += 1
            page_num = metadata.get('page_number', 'Unknown')
            if page_num != 'Unknown':
                books[source]['pages'].add(str(page_num))

        # Format book stats
        book_stats = []
        for source, info in books.items():
            book_stats.append({
                'name': info['name'],
                'author': info['author'],
                'total_chunks': info['chunks'],
                'total_pages': len(info['pages']),
                'page_range': f"{min(info['pages'])}-{max(info['pages'])}" if info['pages'] else "N/A"
            })

        return {
            'total_chunks': len(all_docs['metadatas']),
            'total_books': len(books),
            'books': book_stats
        }

    except Exception as e:
        logger.error(f"Error getting vector store stats: {str(e)}")
        return None
