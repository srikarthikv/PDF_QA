from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        self.models: Dict[str, Optional[AutoModelForSequenceClassification]] = {}
        self.tokenizers: Dict[str, Optional[AutoTokenizer]] = {}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Model configurations for each language
        self.model_configs = {
            'english': {
                'model_name': 'nlptown/bert-base-multilingual-uncased-sentiment',
                'sentiment_map': {
                    0: 'negative',
                    1: 'somewhat negative',
                    2: 'neutral',
                    3: 'somewhat positive',
                    4: 'positive'
                }
            },
            'hindi': {
                'model_name': 'ai4bharat/indic-bert',
                'sentiment_map': {
                    0: 'negative',
                    1: 'neutral',
                    2: 'positive'
                }
            },
            'gujarati': {
                'model_name': 'ai4bharat/indic-bert',
                'sentiment_map': {
                    0: 'negative',
                    1: 'neutral',
                    2: 'positive'
                }
            }
        }

        self._load_models()

    def _load_models(self):
        """Load all sentiment analysis models"""
        for lang, config in self.model_configs.items():
            try:
                logger.info(f"Loading sentiment model for {lang}")
                self.tokenizers[lang] = AutoTokenizer.from_pretrained(config['model_name'])
                self.models[lang] = AutoModelForSequenceClassification.from_pretrained(
                    config['model_name']
                ).to(self.device)
                logger.info(f"Successfully loaded model for {lang}")
            except Exception as e:
                logger.error(f"Failed to load model for {lang}: {str(e)}")
                self.models[lang] = None
                self.tokenizers[lang] = None

    def analyze_sentiment(self, text: str, language: str = 'english') -> str:
        """
        Analyze sentiment of the given text in specified language

        Args:
            text: Text to analyze
            language: Language of the text (english, hindi, gujarati)

        Returns:
            Sentiment label (positive, negative, neutral, etc.)
        """
        if not text or language not in self.model_configs:
            logger.warning(f"Invalid text or language {language}")
            return 'neutral'

        if self.models.get(language) is None or self.tokenizers.get(language) is None:
            logger.warning(f"No model available for {language}")
            return 'neutral'

        try:
            # Tokenize input
            inputs = self.tokenizers[language](
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)

            # Get model prediction
            with torch.no_grad():
                outputs = self.models[language](**inputs)
                logits = outputs.logits
                predicted_class = torch.argmax(logits, dim=1).item()

            # Map to sentiment label
            sentiment_map = self.model_configs[language]['sentiment_map']
            return sentiment_map.get(predicted_class, 'neutral')

        except RuntimeError as e:
            logger.error(f"Runtime error analyzing {language} text: {str(e)}")
            return 'neutral'
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return 'neutral'

    def get_supported_languages(self) -> list:
        """Return list of supported languages"""
        return list(self.model_configs.keys())

# Alias for compatibility
MultilingualSentimentAnalyzer = SentimentAnalyzer
