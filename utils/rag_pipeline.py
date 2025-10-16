from langchain_core.prompts import PromptTemplate
from langchain_google_vertexai import VertexAI
from utils.vector_store import query_vector_store
import logging
import os
import hashlib
import re
import yaml
from pathlib import Path

# Initialize logging
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, config=None, llm=None):
        try:
            # Load config if not provided or if a Path object is provided
            if config is None or isinstance(config, (str, Path)):
                if isinstance(config, str):
                    config_path = Path(config)
                elif isinstance(config, Path):
                    config_path = config
                else:
                    config_path = Path(__file__).parent.parent / "config" / "config.yaml"

                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)

            self.config = config

            # Initialize LLM if not provided
            if llm is None:
                logger.info("Initializing Vertex AI LLM")
                # Set up Google credentials from config
                credentials_path = config.get('google_credentials', {}).get('path')
                if credentials_path and os.path.exists(credentials_path):
                    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path

                llm = VertexAI(
                    model_name=config.get('google_credentials', {}).get('model_name', 'gemini-2.0-flash-exp'),
                    project=config.get('google_credentials', {}).get('project_id'),
                    location=config.get('google_credentials', {}).get('location', 'us-central1'),
                    temperature=config.get('ai', {}).get('temperature', 0.7),
                    max_output_tokens=config.get('ai', {}).get('max_tokens', 4096)
                )

            self.llm = llm
            logger.info("Initializing RAG pipeline with Vertex AI")
            logger.info("RAG pipeline initialized successfully with Vertex AI")
        except Exception as e:
            logger.error(f"Error initializing RAG pipeline: {str(e)}")
            raise

    def get_prompt_template(self, language):
        # Define language-specific instructions with much stronger emphasis
        if language.lower() == "hindi":
            language_instruction = """
            🚨 CRITICAL: आपको हिंदी में ही उत्तर देना है। अंग्रेजी में बिल्कुल न लिखें।
            🚨 MANDATORY: Your entire response MUST be in Hindi (हिंदी) language ONLY. Do NOT use English.
            🚨 आवश्यक: सभी शब्द, वाक्य, और अनुभाग शीर्षक हिंदी में होने चाहिए।
            """
            section_headers = """
            संरचना:
            [संक्षिप्त परिचय 2-3 वाक्यों में]

            **मुख्य सिद्धांत और दर्शन** 🕊️
            [3-5 मुख्य सिद्धांतों की सूची]

            **ऐतिहासिक संदर्भ और शास्त्र** 📜
            [ऐतिहासिक पृष्ठभूमि और महत्वपूर्ण ग्रंथ]

            **जिज्ञासु हैं? इन मुख्य क्षेत्रों का अन्वेषण करें:** 💫
            [4-5 संबंधित विषय]

            स्रोत:
            [फ़ाइल नाम और पेज नंबर]
            """
        elif language.lower() == "gujarati":
            language_instruction = """
            🚨 CRITICAL: તમારે ગુજરાતીમાં જ જવાબ આપવાનો છે। અંગ્રેજીમાં બિલકુલ ન લખો।
            🚨 MANDATORY: Your entire response MUST be in Gujarati (ગુજરાતી) language ONLY. Do NOT use English.
            🚨 આવશ્યક: તમામ શબ્દો, વાક્યો અને વિભાગ શીર્ષકો ગુજરાતીમાં હોવા જોઈએ।
            """
            section_headers = """
            માળખું:
            [સંક્ષિપ્ત પરિચય 2-3 વાક્યોમાં]

            **મુખ્ય સિદ્ધાંતો અને ફિલસૂફી** 🕊️
            [3-5 મુખ્ય સિદ્ધાંતોની યાદી]

            **ઐતિહાસિક સંદર્ભ અને શાસ્ત્રો** 📜
            [ઐતિહાસિક પૃષ્ઠભૂમિ અને મહત્વપૂર્ણ ગ્રંથો]

            **જિજ્ઞાસુ છો? આ મુખ્ય ક્ષેત્રોનું અન્વેષણ કરો:** 💫
            [4-5 સંબંધિત વિષયો]

            સ્રોત:
            [ફાઇલનું નામ અને પેજ નંબર]
            """
        else:  # English
            language_instruction = """
            🚨 CRITICAL: You must answer in English language ONLY.
            """
            section_headers = """
            Structure your answer with these sections:

            [Brief 2-3 sentence introduction with key bolded terms]

            **Core Principles and Philosophy** 🕊️
            [List 3-5 key principles with brief explanations]

            **Historical Context and Scriptures** 📜
            [Brief historical background, key figures, and important texts]

            **Curious to Learn More? Explore These Key Areas:** 💫
            [4-5 related topics as bullet points]

            **Source:**
            [Filename and page ranges]
            """

        return PromptTemplate(
            input_variables=["context", "question"],
            template=f"""{language_instruction}

            You are answering based on a large collection of Jain religious texts and books.
            Use the following context to provide a clear, concise, and engaging answer with emojis.

            IMPORTANT INSTRUCTIONS:
            1. Keep the answer concise and easy to read
            2. Use markdown formatting with ** for bold text and emojis to make it engaging
            3. Structure the answer with clear sections using emojis
            4. Cite sources at the end
            5. Focus on the core information first, then provide deeper insights

            {section_headers}

            FORMATTING RULES:
            - Use ** for bold text (e.g., **Ahimsa**, **Moksha**)
            - Add relevant emojis to section headers (🕊️, 📜, 📚, 💫, etc.)
            - Keep paragraphs short (2-3 sentences max)
            - Use bullet points for lists
            - Make it engaging and easy to scan

            Context from multiple books:
            {{context}}

            Question:
            {{question}}

            {language_instruction}"""
        )

    def generate_unique_key(self, doc):
        metadata = doc.metadata or {}
        page_num = metadata.get('page_number', 'Unknown Page')
        source = metadata.get('source', 'Unknown Source')
        content_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()[:8]
        return f"{source}_{page_num}_{content_hash}"

    def consolidate_sources(self, unique_docs):
        """Consolidate sources by PDF filename and page numbers, prioritizing most relevant sources."""
        sources_dict = {}
        for doc in unique_docs:
            if hasattr(doc, 'metadata') and doc.metadata:
                source = doc.metadata.get('source', 'Unknown Source')
                page_num = doc.metadata.get('page_number', 'Unknown Page')

                if source not in sources_dict:
                    sources_dict[source] = {'pages': set(), 'count': 0}

                if page_num != 'Unknown Page' and str(page_num).isdigit():
                    sources_dict[source]['pages'].add(int(page_num))

                sources_dict[source]['count'] += 1

        # Sort sources by relevance (count of matching chunks)
        sorted_sources = sorted(sources_dict.items(), key=lambda x: x[1]['count'], reverse=True)

        # Format the sources - prioritize the most relevant source
        formatted_sources = []
        # Only show the top 1 most relevant source to keep it clean
        for source, data in sorted_sources[:1]:
            pages = data['pages']
            if pages:
                sorted_pages = sorted(pages)
                if len(sorted_pages) > 3:
                    page_str = f"{sorted_pages[0]}, {sorted_pages[1]}, ..., {sorted_pages[-1]}"
                else:
                    page_str = ", ".join(map(str, sorted_pages))
                formatted_sources.append(f"{source} ({page_str})")
            else:
                formatted_sources.append(f"{source}")

        return formatted_sources

    def extract_related_topics(self, answer):
        """Extract related topics from the answer text"""
        # Try multiple patterns to extract topics
        topics_patterns = [
            r'Curious to Learn More\? Explore These Key Areas:\s*\*\*\s*(.*?)(?=\n\n|\*\*Source:|\*\*Here are|$)',
            r'Curious to Learn More\? Explore These Key Areas:\s*(.*?)(?=\n\n\*\*|\*\*Source:|Source:|$)',
            r'जिज्ञासु हैं\? इन मुख्य क्षेत्रों का अन्वेषण करें:\s*💫\s*(.*?)(?=\n\n|$)',
        ]

        for pattern in topics_patterns:
            topics_section = re.search(pattern, answer, re.DOTALL)
            if topics_section:
                topics_text = topics_section.group(1).strip()
                # Extract topics - handle both bullet points and plain text
                topics = []
                for line in topics_text.split('\n'):
                    line = line.strip()
                    # Remove markdown bullets, asterisks, and numbering
                    line = re.sub(r'^[\*\-\•]\s*', '', line)
                    line = re.sub(r'^\d+\.\s*', '', line)
                    if line and len(line) > 3 and not line.startswith('**'):
                        topics.append(line)

                if topics:
                    return topics[:5]  # Limit to top 5 topics

        return []

    def get_context_summary(self, sect, language):
        """Get summary of available content for a specific sect and language"""
        try:
            # This is a placeholder implementation
            # In a real implementation, this would query the vector store
            return {
                'has_content': False,
                'sect': sect,
                'language': language,
                'document_count': 0,
                'chunk_count': 0
            }
        except Exception as e:
            logger.error(f"Error getting context summary: {str(e)}")
            return {
                'has_content': False,
                'sect': sect,
                'language': language,
                'error': str(e)
            }

    def answer_question(self, question, vector_store=None, language='english', pdf_paths=None, sect=None, user_context=None):
        try:
            logger.info(f"Processing question: {question} in language: {language}")

            # Check if vector store is available
            if vector_store is None:
                logger.warning("No vector store provided, cannot answer question")
                return self._no_content_response(question, language)

            # Retrieve relevant documents with metadata (no limit - get top 50 most relevant)
            # This supports large-scale book collections (10-15 books with ~2000 pages each)
            docs = query_vector_store(question, vector_store, k=50)
            logger.info(f"Retrieved {len(docs)} documents from vector store")

            # Check if we have any relevant documents
            if not docs or len(docs) == 0:
                logger.warning("No relevant documents found in vector store")
                return self._no_content_response(question, language)

            # Deduplicate documents using a unique key including content hash
            unique_docs = []
            seen = set()
            for doc in docs:
                unique_key = self.generate_unique_key(doc)
                if unique_key not in seen:
                    seen.add(unique_key)
                    unique_docs.append(doc)
            logger.info(f"After deduplication, {len(unique_docs)} unique documents remain")

            # Combine all documents into context with metadata
            # Group by source book for better organization
            context_by_book = {}
            for doc in unique_docs:
                if hasattr(doc, 'metadata') and doc.metadata:
                    source = doc.metadata.get('source', 'Unknown Source')
                    page_num = doc.metadata.get('page_number', 'Unknown Page')
                    content = doc.page_content

                    if source not in context_by_book:
                        context_by_book[source] = []

                    context_by_book[source].append({
                        'content': content,
                        'page': page_num
                    })
                    logger.info(f"Context content from {source} (page {page_num}): {content[:100]}...")

            # Build organized context from all books
            context_parts = []
            for source, chunks in context_by_book.items():
                pages = [c['page'] for c in chunks]
                content_list = [c['content'] for c in chunks]
                context_parts.append(
                    f"=== From: {source} (Pages: {', '.join(map(str, pages))}) ===\n"
                    + "\n\n".join(content_list)
                )

            context_text = "\n\n".join(context_parts) if context_parts else "No relevant context available."

            # Get the appropriate prompt template based on language
            prompt_template = self.get_prompt_template(language)
            prompt = prompt_template.format(context=context_text, question=question)

            # Generate answer using Vertex AI
            logger.info("Generating answer with Vertex AI")
            response = self.llm.invoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)

            # Process the answer to ensure proper formatting
            answer = self._clean_answer(answer)

            # Extract related topics for button generation
            related_topics = self.extract_related_topics(answer)

            # Append consolidated sources only if unique_docs exist
            sources = []
            if unique_docs:
                consolidated_sources = self.consolidate_sources(unique_docs)
                if consolidated_sources:
                    # Remove any existing source lines (English, Hindi, Gujarati)
                    answer = re.sub(r'\n\*\*Source:\*\*.*?(?=\n\n|$)', '', answer, flags=re.MULTILINE | re.DOTALL)
                    answer = re.sub(r'\nSource:.*?(?=\n\n|$)', '', answer, flags=re.MULTILINE | re.DOTALL)
                    answer = re.sub(r'\n\*\*स्रोत:\*\*.*?(?=\n\n|$)', '', answer, flags=re.MULTILINE | re.DOTALL)
                    answer = re.sub(r'\nस्रोत:.*?(?=\n\n|$)', '', answer, flags=re.MULTILINE | re.DOTALL)
                    # Add the single most relevant source at the end
                    answer = answer.strip() + f"\n\nस्रोत:\n{consolidated_sources[0]}"
                    sources = consolidated_sources

            logger.info("Answer generated successfully")
            return {
                'answer': answer,
                'key_points': [],
                'related_topics': related_topics,
                'sources': sources,
                'confidence': 0.8 if unique_docs else 0,
                'based_on_uploaded_content': bool(unique_docs)
            }
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return {
                'answer': f"Error generating answer: {str(e)}",
                'key_points': [],
                'related_topics': [],
                'sources': [],
                'confidence': 0,
                'based_on_uploaded_content': False
            }

    def _clean_answer(self, answer):
        """Clean up the answer format"""
        # Keep markdown formatting (**, emojis, etc.)
        # Only remove excessive ###
        answer = answer.replace('###', '')

        # Ensure consistent section headers
        sections = [
            'Overview', 'Historical Context', 'Content and Structure',
            'Significance', 'Key Details',
            'Curious to Learn More', 'Here are Some Related Questions', 'Source:',
            'Core Principles and Philosophy', 'Historical Context and Scriptures',
            'Sects and Key Texts'
        ]

        for section in sections:
            # Don't remove the : after section names as they may be intentional
            pass

        return answer.strip()

    def _no_content_response(self, question: str, language: str) -> dict:
        """Return a response indicating no content is available to answer the question"""
        try:
            logger.info("No content available to answer question")

            # Create language-specific "no content" messages
            no_content_messages = {
                "english": "I cannot find any information about this topic in the uploaded documents. Please upload relevant PDFs to get accurate answers.",
                "hindi": "अपलोड किए गए दस्तावेज़ों में इस विषय के बारे में कोई जानकारी नहीं मिली। सटीक उत्तर पाने के लिए कृपया प्रासंगिक पीडीएफ अपलोड करें।",
                "gujarati": "અપલોડ કરેલા દસ્તાવેજોમાં આ વિષય વિશે કોઈ માહિતી મળી નથી. ચોક્કસ જવાબો મેળવવા માટે કૃપા કરીને સંબંધિત પીડીએફ અપલોડ કરો."
            }

            answer = no_content_messages.get(language.lower(), no_content_messages["english"])

            logger.info("No content response returned")
            return {
                'answer': answer,
                'key_points': [],
                'related_topics': [],
                'sources': [],
                'confidence': 0,
                'based_on_uploaded_content': False
            }

        except Exception as e:
            logger.error(f"Error generating no content response: {str(e)}")
            return {
                'answer': "I apologize, but I cannot answer this question as no relevant content has been uploaded.",
                'key_points': [],
                'related_topics': [],
                'sources': [],
                'confidence': 0,
                'based_on_uploaded_content': False
            }
