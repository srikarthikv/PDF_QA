"""
Content Formatter for Jain Learning System
Handles sect-specific terminology replacement and content formatting
"""

import re
import yaml
from typing import Dict, List, Tuple, Any
from pathlib import Path

class ContentFormatter:
    """Formats content with sect-specific terminology and proper structure"""

    def __init__(self, config_path: str = None):
        """Initialize with configuration"""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "config.yaml"

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # Extract sect terminology mappings
        self.sect_terms = {}
        for sect in self.config['sects']['supported']:
            self.sect_terms[sect['code']] = {
                'male_ascetic': sect['male_ascetic'],
                'female_ascetic': sect['female_ascetic'],
                'male_layperson': sect['male_layperson'],
                'female_layperson': sect['female_layperson'],
                'name': sect['name'],
                'display_name': sect['display_name']
            }

        # Generic term mappings for replacement
        self.generic_terms = {
            'sadhu': 'male_ascetic',
            'sadhvi': 'female_ascetic',
            'muni': 'male_ascetic',
            'aryika': 'female_ascetic',
            'shravak': 'male_layperson',
            'shravaka': 'male_layperson',
            'shravika': 'female_layperson',
            'monk': 'male_ascetic',
            'nun': 'female_ascetic',
            'layman': 'male_layperson',
            'laywoman': 'female_layperson'
        }

    def format_content_for_sect(self, content: str, sect: str,
                              language: str = 'en') -> str:
        """
        Replace generic terminology with sect-specific terms

        Args:
            content: Raw content text
            sect: Target sect code
            language: Target language code

        Returns:
            Formatted content with sect-specific terminology
        """
        if sect not in self.sect_terms:
            return content

        formatted_content = content
        sect_mapping = self.sect_terms[sect]

        # Replace generic terms with sect-specific ones
        for generic_term, term_type in self.generic_terms.items():
            if term_type in sect_mapping:
                specific_term = sect_mapping[term_type]

                # Case-insensitive replacement with proper case handling
                pattern = re.compile(re.escape(generic_term), re.IGNORECASE)

                def replace_func(match):
                    original = match.group()
                    replacement = specific_term

                    # Maintain original case
                    if original.isupper():
                        return replacement.upper()
                    elif original.istitle():
                        return replacement.title()
                    elif original.islower():
                        return replacement.lower()
                    else:
                        return replacement

                formatted_content = pattern.sub(replace_func, formatted_content)

        return formatted_content

    def structure_answer(self, content: str, question: str = None,
                        sect: str = None, language: str = 'en') -> Dict[str, Any]:
        """
        Structure raw content into properly formatted answer

        Args:
            content: Raw answer content
            question: Original question
            sect: Target sect
            language: Target language

        Returns:
            Structured answer dictionary
        """
        # Apply sect-specific formatting if specified
        if sect:
            content = self.format_content_for_sect(content, sect, language)

        # Extract key sections from content
        sections = self._extract_sections(content)

        # Generate related topics
        related_topics = self._extract_related_topics(content)

        # Create structured response
        structured_answer = {
            'main_answer': sections.get('main', content),
            'key_points': sections.get('points', []),
            'examples': sections.get('examples', []),
            'related_topics': related_topics,
            'sect_specific': sect is not None,
            'language': language,
            'formatted_content': self._apply_markdown_formatting(content)
        }

        return structured_answer

    def _extract_sections(self, content: str) -> Dict[str, Any]:
        """Extract different sections from content"""
        sections = {'main': content, 'points': [], 'examples': []}

        # Extract bullet points
        bullet_pattern = r'(?:^|\n)(?:[-*•]|\d+\.)\s*(.+?)(?=\n|$)'
        bullets = re.findall(bullet_pattern, content, re.MULTILINE)
        if bullets:
            sections['points'] = [point.strip() for point in bullets]

        # Extract examples
        example_pattern = r'(?:example|instance|illustration):\s*(.+?)(?=\n\n|$)'
        examples = re.findall(example_pattern, content, re.IGNORECASE | re.DOTALL)
        if examples:
            sections['examples'] = [ex.strip() for ex in examples]

        return sections

    def _extract_related_topics(self, content: str) -> List[str]:
        """Extract related topics from content"""
        # Common Jain concepts to look for
        jain_concepts = [
            'ahimsa', 'karma', 'moksha', 'dharma', 'tirthankara',
            'jina', 'kevala gyan', 'ratnatraya', 'panch mahavrat',
            'anuvratas', 'mahavratas', 'puja', 'samadhi', 'tapas',
            'dana', 'shravana', 'swadhyaya', 'tirth', 'upashraya'
        ]

        related = []
        content_lower = content.lower()

        for concept in jain_concepts:
            if concept in content_lower and concept not in related:
                related.append(concept.title())

        return related[:5]  # Limit to 5 related topics

    def _apply_markdown_formatting(self, content: str) -> str:
        """Apply markdown formatting to content"""
        # Add proper headings
        content = re.sub(r'^(.+?):$', r'## \1', content, flags=re.MULTILINE)

        # Format lists
        content = re.sub(r'^(\d+\.)\s*(.+)$', r'\1 **\2**', content, flags=re.MULTILINE)
        content = re.sub(r'^([-*])\s*(.+)$', r'\1 \2', content, flags=re.MULTILINE)

        # Emphasize important terms
        jain_terms = [
            'Tirthankara', 'Jina', 'Ahimsa', 'Karma', 'Moksha',
            'Ratnatraya', 'Kevala Gyan', 'Dharma'
        ]

        for term in jain_terms:
            pattern = rf'\b({re.escape(term)})\b'
            content = re.sub(pattern, r'**\1**', content, flags=re.IGNORECASE)

        return content

    def format_quiz_content(self, quiz_data: Dict[str, Any],
                          sect: str, language: str = 'en') -> Dict[str, Any]:
        """Format quiz content with sect-specific terminology"""
        formatted_quiz = quiz_data.copy()

        # Format questions
        if 'questions' in formatted_quiz:
            for question in formatted_quiz['questions']:
                # Format question text
                if 'question' in question:
                    question['question'] = self.format_content_for_sect(
                        question['question'], sect, language
                    )

                # Format options
                if 'options' in question:
                    question['options'] = [
                        self.format_content_for_sect(option, sect, language)
                        for option in question['options']
                    ]

                # Format explanation
                if 'explanation' in question:
                    question['explanation'] = self.format_content_for_sect(
                        question['explanation'], sect, language
                    )

        return formatted_quiz

    def get_sect_specific_greeting(self, sect: str, language: str = 'en') -> str:
        """Get sect-specific greeting message"""
        greetings = {
            'en': {
                'digambara': f"Welcome to the {self.sect_terms[sect]['display_name']} learning path. May you find wisdom in the teachings of our {self.sect_terms[sect]['male_ascetic']}s and {self.sect_terms[sect]['female_ascetic']}s.",
                'shwetambara': f"Namaste! Welcome to the {self.sect_terms[sect]['display_name']} tradition. Learn from the sacred teachings guided by our {self.sect_terms[sect]['male_ascetic']}s and {self.sect_terms[sect]['female_ascetic']}s.",
                'terapanthi': f"Welcome to the {self.sect_terms[sect]['display_name']} path of learning. Discover the reformed teachings through our {self.sect_terms[sect]['male_ascetic']}s and {self.sect_terms[sect]['female_ascetic']}s.",
                'sthanakvasi': f"Greetings! Welcome to the {self.sect_terms[sect]['display_name']} learning journey. Explore the pure teachings with guidance from our {self.sect_terms[sect]['male_ascetic']}s and {self.sect_terms[sect]['female_ascetic']}s."
            },
            'hi': {
                'digambara': f"दिगंबर शिक्षण पथ में आपका स्वागत है। हमारे {self.sect_terms[sect]['male_ascetic']} और {self.sect_terms[sect]['female_ascetic']} की शिक्षाओं में ज्ञान प्राप्त करें।",
                'shwetambara': f"नमस्ते! श्वेतांबर परंपरा में आपका स्वागत है। हमारे {self.sect_terms[sect]['male_ascetic']} और {self.sect_terms[sect]['female_ascetic']} द्वारा निर्देशित पवित्र शिक्षाओं से सीखें।",
                'terapanthi': f"तेरापंथी शिक्षण पथ में आपका स्वागत है। हमारे {self.sect_terms[sect]['male_ascetic']} और {self.sect_terms[sect]['female_ascetic']} के माध्यम से सुधारित शिक्षाओं की खोज करें।",
                'sthanakvasi': f"नमस्कार! स्थानकवासी शिक्षण यात्रा में आपका स्वागत है। हमारे {self.sect_terms[sect]['male_ascetic']} और {self.sect_terms[sect]['female_ascetic']} के मार्गदर्शन में शुद्ध शिक्षाओं का अन्वेषण करें।"
            }
        }

        if language in greetings and sect in greetings[language]:
            return greetings[language][sect]

        return f"Welcome to the {self.sect_terms.get(sect, {}).get('display_name', 'Jain')} learning path!"

    def format_learning_content(self, content: Dict[str, Any],
                              sect: str, age_group: str = 'adult',
                              language: str = 'en') -> Dict[str, Any]:
        """Format learning content based on sect, age group, and language"""
        formatted_content = content.copy()

        # Apply sect-specific formatting
        for key, value in formatted_content.items():
            if isinstance(value, str):
                formatted_content[key] = self.format_content_for_sect(value, sect, language)
            elif isinstance(value, list):
                formatted_content[key] = [
                    self.format_content_for_sect(item, sect, language) if isinstance(item, str) else item
                    for item in value
                ]

        # Age-appropriate adjustments
        if age_group == 'child':
            # Simplify language for children
            formatted_content = self._simplify_for_children(formatted_content)
        elif age_group == 'teen':
            # Add more context and examples for teenagers
            formatted_content = self._enhance_for_teenagers(formatted_content)

        return formatted_content

    def _simplify_for_children(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Simplify content for children"""
        simplified = content.copy()

        # Replace complex terms with simpler ones
        simplifications = {
            'philosophy': 'teachings',
            'principle': 'rule',
            'meditation': 'quiet thinking',
            'scripture': 'holy book',
            'doctrine': 'teaching'
        }

        for key, value in simplified.items():
            if isinstance(value, str):
                for complex_term, simple_term in simplifications.items():
                    value = re.sub(rf'\b{complex_term}\b', simple_term, value, flags=re.IGNORECASE)
                simplified[key] = value

        return simplified

    def _enhance_for_teenagers(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance content for teenagers with more context"""
        enhanced = content.copy()

        # Add contemporary relevance and examples
        if 'main_content' in enhanced:
            enhanced['contemporary_relevance'] = self._add_contemporary_context(enhanced['main_content'])

        return enhanced

    def _add_contemporary_context(self, content: str) -> str:
        """Add contemporary context to make content relatable for teenagers"""
        context_additions = {
            'ahimsa': "This principle is very relevant today in discussions about environmental protection and ethical treatment of animals.",
            'karma': "Similar to the modern concept of 'what goes around comes around' or cause and effect.",
            'meditation': "Like mindfulness practices that are popular today for mental health and stress reduction."
        }

        for term, context in context_additions.items():
            if term.lower() in content.lower():
                content += f"\n\n💡 **Modern Connection**: {context}"

        return content