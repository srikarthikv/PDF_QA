"""
PDF Processor Wrapper for the Jain Learning Ecosystem
"""
import logging
import re
from pathlib import Path
from utils.pdf_processor import extract_text_from_pdf, split_text_into_chunks

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Wrapper class for PDF processing functionality"""

    def __init__(self, config):
        self.config = config
        logger.info("PDFProcessor initialized")

    def extract_page_numbers_from_chunk(self, chunk_text):
        """Extract all page numbers from a chunk of text that contains page markers"""
        # Find all page markers in the format "=== Page X ===" or "=== Page X (method) ==="
        page_pattern = r'===\s*Page\s+(\d+)\s*(?:\([^)]+\))?\s*==='
        page_numbers = re.findall(page_pattern, chunk_text)

        if page_numbers:
            # Convert to integers and return the range
            page_nums = [int(p) for p in page_numbers]
            return page_nums

        return []

    def process_pdf(self, file_path, sect, language, source_name):
        """Process a PDF file and return chunks with metadata"""
        try:
            # Extract text from PDF
            output_dir = self.config.get('data', {}).get('output_dir', 'data/output')
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            text = extract_text_from_pdf(file_path, output_dir)

            if not text or not text.strip():
                return {
                    'success': False,
                    'error': 'No text could be extracted from the PDF'
                }

            # Split text into chunks
            chunks_list = split_text_into_chunks(text, self.config)

            if not chunks_list:
                return {
                    'success': False,
                    'error': 'Could not split text into chunks'
                }

            # Create chunk objects with metadata
            chunks = []
            for i, chunk_text in enumerate(chunks_list):
                # Extract page numbers from this chunk
                page_numbers = self.extract_page_numbers_from_chunk(chunk_text)

                # Determine the primary page number for this chunk
                # Use the first page number found, or use chunk index if no page markers
                if page_numbers:
                    # Use the first page number in the chunk as the primary page
                    primary_page = page_numbers[0]
                else:
                    # If no page markers found, use chunk index as fallback
                    primary_page = i + 1

                chunk = {
                    'id': f"{source_name}_{sect}_{language}_{i}",
                    'content': chunk_text,
                    'metadata': {
                        'source': source_name,
                        'sect': sect,
                        'language': language,
                        'chunk_index': i,
                        'file_path': file_path,
                        'page_number': primary_page,
                        'page_numbers': page_numbers if page_numbers else [primary_page]
                    }
                }
                chunks.append(chunk)
                logger.debug(f"Chunk {i}: extracted page numbers {page_numbers}, primary page: {primary_page}")

            return {
                'success': True,
                'chunks': chunks,
                'total_chunks': len(chunks),
                'total_characters': len(text)
            }

        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
