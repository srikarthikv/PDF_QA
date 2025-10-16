import os
import logging
import sys
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import cv2
import numpy as np
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Custom StreamHandler to force UTF-8 encoding
class UTF8StreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_system.log', encoding='utf-8'),
        UTF8StreamHandler(sys.stdout)
    ]
)
# Set the stdout encoding to UTF-8
sys.stdout.reconfigure(encoding='utf-8')
logger = logging.getLogger(__name__)

class HindiPDFProcessor:
    def __init__(self):
        self.setup_tesseract()
        self.validate_tesseract_languages()

    def setup_tesseract(self):
        """Configure Tesseract paths and settings"""
        possible_tesseract_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
        ]

        for path in possible_tesseract_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                break
        else:
            logger.error("Tesseract OCR not found. Please install it and add to PATH")
            raise FileNotFoundError("Tesseract OCR not found")

        # Set TESSDATA_PREFIX environment variable
        tessdata_dirs = [
            r'C:\Program Files\Tesseract-OCR\tessdata',
            r'C:\Program Files (x86)\Tesseract-OCR\tessdata',
            os.path.join(os.path.dirname(pytesseract.pytesseract.tesseract_cmd), 'tessdata')
        ]

        for dir_path in tessdata_dirs:
            if os.path.exists(dir_path):
                os.environ['TESSDATA_PREFIX'] = dir_path
                break

    def validate_tesseract_languages(self):
        """Check if required language data files are available"""
        required_langs = ['hin', 'eng']
        missing_langs = []

        try:
            available_langs = pytesseract.get_languages(config='')
        except:
            available_langs = []

        for lang in required_langs:
            if lang not in available_langs:
                missing_langs.append(lang)

        if missing_langs:
            logger.error(f"Missing language data files: {', '.join(missing_langs)}")
            logger.info("Please download the required .traineddata files from:")
            logger.info("https://github.com/tesseract-ocr/tessdata")
            logger.info(f"And place them in: {os.environ.get('TESSDATA_PREFIX', 'tessdata directory')}")
            raise ValueError("Missing language data files")

    def preprocess_image_for_hindi_ocr(self, img):
        """Enhanced image preprocessing specifically for Hindi text"""
        try:
            # Convert PIL Image to OpenCV format
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            # Convert to grayscale
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )

            # Apply dilation and erosion to remove noise
            kernel = np.ones((1, 1), np.uint8)
            img_processed = cv2.dilate(thresh, kernel, iterations=1)
            img_processed = cv2.erode(img_processed, kernel, iterations=1)

            # Apply Gaussian blur to smooth the image
            img_processed = cv2.GaussianBlur(img_processed, (3, 3), 0)

            # Enhance contrast (helps with light text)
            img_processed = cv2.convertScaleAbs(img_processed, alpha=1.5, beta=0)

            return Image.fromarray(img_processed)
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            return img  # Return original if processing fails

    def postprocess_hindi_text(self, text):
        """Clean up OCR output for Hindi text with advanced corrections"""
        if not text:
            return ""

        # Standardize line breaks and remove empty lines
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)

        # Standardize Hindi punctuation
        text = re.sub(r'[॥|]', '।', text)
        text = re.sub(r'[\|\_\~]', '', text)

        # Fix common Hindi OCR errors (expanded list)
        replacements = {
            'िे': 'िए',    # Common error for "ie" matra
            '्र्': '्र',    # Remove extra halant
            'क्क': 'क्ष',   # Common confusion with "ksha"
            'प्प': 'प्य',   # Common confusion with "pya"
            'त्त': 'त्र',   # Common confusion with "tra"
            'ट्ट': 'ट्ठ',    # Common error for "ttha"
            'ड्ड': 'ड्ड़',   # Common error for "ddha"
            'अा': 'आ',      # Split "aa" vowel
            'इि': 'ई',      # Split "ee" vowel
            'उु': 'ऊ',      # Split "oo" vowel
            'एे': 'ऐ',      # Common error for "ai" vowel
            'ओो': 'औ',      # Common error for "au" vowel
            'ंं': 'ं',      # Double anusvara
            'ःः': 'ः',      # Double visarga
            ''': "'",       # Standardize quotes
            ''': "'",
            '"': '"',
            '"': '"',
            'क़': 'क',      # Common confusion with Arabic qaf
            'ख़': 'ख',
            'ग़': 'ग',
            'ज़': 'ज',
            'ड़': 'ज',
            'ढ़': 'ढ',
            'फ़': 'फ',
            'य़': 'य'
        }

        for wrong, right in replacements.items():
            text = text.replace(wrong, right)

        # Fix spacing issues (matras and halants)
        text = re.sub(r'(?<=\S)([ािीुूृेैोौंः])(?=\S)', r'\1', text)
        text = re.sub(r'(?<=\S)(्)(?=\S)', r'\1', text)

        # Remove hyphenated word breaks (common in PDFs)
        text = re.sub(r'(\S)-\s+(\S)', r'\1\2', text)

        # Standardize spaces around punctuation
        text = re.sub(r'\s+([।,\.;:!?\'])', r'\1', text)
        text = re.sub(r'([\(\[{])\s+', r'\1', text)
        text = re.sub(r'\s+([\)\]}])', r'\1', text)

        return text.strip()

    def is_valid_hindi_text(self, text):
        """Check if text contains valid Hindi characters"""
        hindi_pattern = re.compile(
            r'[\u0900-\u097F।॥,.?!\s\'"-]|'
            r'[0-9]|'
            r'[a-zA-Z]'  # Allow some English for mixed documents
        )

        valid_chars = sum(1 for char in text if hindi_pattern.fullmatch(char))
        ratio = valid_chars / max(1, len(text))

        return ratio > 0.6

    def process_pdf_page(self, page, page_num, total_pages, pdf_filename):
        """Process a single PDF page with fallback mechanisms"""
        result = {
            'page_num': page_num,
            'text': '',
            'method': 'direct',
            'error': None,
            'pdf_filename': pdf_filename
        }

        try:
            # First try direct text extraction
            page_text = page.get_text()

            # Check if extracted text contains valid Hindi characters
            if page_text.strip() and self.is_valid_hindi_text(page_text):
                result['text'] = page_text
                return result

            # If direct extraction fails, use OCR
            result['method'] = 'ocr'

            # Get high-resolution image (400 DPI for better OCR)
            zoom = 4.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)

            # Convert to PIL Image
            img = Image.open(io.BytesIO(pix.tobytes()))

            # Preprocess image for better Hindi OCR
            processed_img = self.preprocess_image_for_hindi_ocr(img)

            # OCR configuration for Hindi with fallback to English
            custom_config = r'--oem 3 --psm 6 -l hin+eng'

            # Perform OCR with timeout
            page_text = pytesseract.image_to_string(
                processed_img,
                config=custom_config,
                timeout=60
            )

            # Post-process the extracted text
            page_text = self.postprocess_hindi_text(page_text)

            result['text'] = page_text

        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Error processing page {page_num + 1} in {pdf_filename}: {str(e)}")

        return result

def extract_text_from_pdf(pdf_path, output_dir):
    """Extract text from PDF using advanced Hindi processing"""
    try:
        pdf_filename = os.path.basename(pdf_path)
        logger.info(f"Extracting text from {pdf_filename}")

        # First try PyPDF2 for text-based PDFs
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text += page_text

            if text.strip():
                logger.info(f"Successfully extracted text via PyPDF2 from {pdf_filename}")
                # Save extracted text to file
                output_file = os.path.join(output_dir, pdf_filename.replace('.pdf', '_extracted.txt'))
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(text)
                return text
        except Exception as pdf_error:
            logger.warning(f"PyPDF2 extraction failed for {pdf_filename}, trying advanced methods: {str(pdf_error)}")

        # For scanned PDFs or when PyPDF2 fails, use advanced processing
        processor = HindiPDFProcessor()
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        extracted_text = []

        logger.info(f"Processing {total_pages} pages with advanced Hindi extraction")

        # Process pages in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []

            for page_num, page in enumerate(doc):
                futures.append(
                    executor.submit(
                        processor.process_pdf_page,
                        page,
                        page_num,
                        total_pages,
                        pdf_filename
                    )
                )

            # Process results as they complete
            for future in tqdm(as_completed(futures), total=total_pages, desc="Processing pages"):
                result = future.result()

                if result['error']:
                    logger.warning(f"Page {result['page_num'] + 1} error in {pdf_filename}: {result['error']}")
                    extracted_text.append(f"\n[Error processing page {result['page_num'] + 1}]\n")
                else:
                    extracted_text.append(
                        f"\n=== Page {result['page_num'] + 1} ({result['method']}) ===\n"
                        f"{result['text']}\n"
                    )

        full_text = ''.join(extracted_text)

        if not full_text.strip():
            logger.error(f"No usable text extracted from {pdf_filename} via any method")
            return ""

        # Save extracted text to file
        output_file = os.path.join(output_dir, pdf_filename.replace('.pdf', '_extracted.txt'))
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_text)
        logger.info(f"Extracted text saved to {output_file}")
        return full_text

    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
        return ""

def extract_text_from_pdfs(pdf_paths, output_dir):
    """Process multiple PDF files with Hindi text extraction"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        all_text = ""
        for pdf_path in pdf_paths:
            text = extract_text_from_pdf(pdf_path, output_dir)
            all_text += f"\n--- {os.path.basename(pdf_path)} ---\n{text}"

        # Save combined text
        combined_file = os.path.join(output_dir, 'extracted_text.txt')
        with open(combined_file, 'w', encoding='utf-8') as f:
            f.write(all_text)
        logger.info(f"Combined extracted text saved to {combined_file}")
        return all_text
    except Exception as e:
        logger.error(f"Error processing multiple PDFs: {str(e)}")
        return ""

def split_text_into_chunks(text, config):
    """Split the extracted text into chunks using langchain's RecursiveCharacterTextSplitter"""
    try:
        if not text.strip():
            logger.warning("No text to split into chunks")
            return []

        chunk_size = config['vector_stores']['chunking']['chunk_size']
        chunk_overlap = config['vector_stores']['chunking']['chunk_overlap']
        logger.info(f"Chunking text with size={chunk_size}, overlap={chunk_overlap}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)
        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks
    except KeyError as e:
        logger.error(f"Configuration error for chunking: {str(e)}. Using defaults (chunk_size=1000, chunk_overlap=200)")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)
        logger.info(f"Split text into {len(chunks)} chunks with default settings")
        return chunks
    except Exception as e:
        logger.error(f"Error splitting text into chunks: {str(e)}")
        return []
