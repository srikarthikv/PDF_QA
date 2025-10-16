"""
Assessment Engine for Jain Learning System
Handles assessment creation, scoring, and knowledge level detection
"""

import os
import json
import random
import logging
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
import yaml
from pathlib import Path
import re

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from .content_formatter import ContentFormatter
from .vector_store_manager import VectorStoreManager

logger = logging.getLogger(__name__)

class AssessmentEngine:
    """Creates and scores assessments based on uploaded content and user profile"""

    def __init__(self, config_path: str = None):
        """Initialize assessment engine with configuration"""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "config.yaml"

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.content_formatter = ContentFormatter(config_path)
        self.vector_store = VectorStoreManager(self.config)

        # Assessment configuration
        self.difficulty_levels = {item['code']: item for item in self.config['assessment']['difficulty_levels']}
        self.categories = {item['code']: item for item in self.config['assessment']['categories']}
        self.knowledge_levels = {item['code']: item for item in self.config['user_profiles']['knowledge_levels']}

        # Initialize Gemini if available
        self.gemini_available = GEMINI_AVAILABLE and 'GOOGLE_API_KEY' in os.environ
        if self.gemini_available:
            genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
            self.model = genai.GenerativeModel('gemini-1.5-pro')

    def create_assessment(self, sect: str, language: str = 'en',
                         religion: str = 'jainism', difficulty: str = 'medium',
                         question_count: int = 10, categories: List[str] = None) -> Dict[str, Any]:
        """
        Create assessment based on uploaded content

        Args:
            sect: Jain sect code
            language: Language code
            religion: Religion identifier
            difficulty: Difficulty level
            question_count: Number of questions
            categories: Topic categories to focus on

        Returns:
            Generated assessment with questions and metadata
        """
        try:
            # Check if content is available
            content_available = self._check_content_availability(sect, language, religion)
            if not content_available:
                return self._create_fallback_assessment(sect, language, difficulty, question_count)

            # Generate questions from uploaded content
            questions = self._generate_questions_from_content(
                sect, language, religion, difficulty, question_count, categories
            )

            if not questions:
                return self._create_fallback_assessment(sect, language, difficulty, question_count)

            # Create assessment metadata
            assessment_id = f"assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{sect}_{language}"

            assessment = {
                'id': assessment_id,
                'sect': sect,
                'language': language,
                'religion': religion,
                'difficulty': difficulty,
                'question_count': len(questions),
                'categories': categories or list(self.categories.keys()),
                'created_timestamp': datetime.now().isoformat(),
                'time_limit_minutes': self._calculate_time_limit(len(questions), difficulty),
                'questions': questions,
                'scoring': self._create_scoring_rubric(questions),
                'based_on_uploaded_content': True
            }

            return assessment

        except Exception as e:
            logger.error(f"Error creating assessment: {e}")
            return {
                'error': str(e),
                'based_on_uploaded_content': False
            }

    def _check_content_availability(self, sect: str, language: str, religion: str) -> bool:
        """Check if sufficient content is available for assessment creation"""
        try:
            stats = self.vector_store.get_collection_stats(sect, language, religion)
            return stats and stats.get('document_count', 0) > 0
        except:
            return False

    def _generate_questions_from_content(self, sect: str, language: str, religion: str,
                                       difficulty: str, question_count: int,
                                       categories: List[str] = None) -> List[Dict[str, Any]]:
        """Generate questions based on uploaded content"""
        if not self.gemini_available:
            return self._generate_basic_questions_from_content(sect, language, religion, difficulty, question_count)

        try:
            # Get sample content from vector store
            sample_queries = [
                "principles of Jainism",
                "Tirthankara teachings",
                "ahimsa non-violence",
                "karma and moksha",
                "Jain practices rituals"
            ]

            content_samples = []
            for query in sample_queries:
                docs = self.vector_store.query_vector_store(
                    query=query,
                    sect=sect,
                    language=language,
                    religion=religion,
                    k=3
                )
                content_samples.extend([doc['content'] for doc in docs])

            if not content_samples:
                return []

            # Combine content samples
            combined_content = '\n\n'.join(content_samples[:10])  # Limit to prevent token overflow

            # Get sect-specific terminology
            sect_info = self.content_formatter.sect_terms.get(sect, {})

            # Generate questions using AI
            prompt = self._create_question_generation_prompt(
                combined_content, sect, language, difficulty, question_count, sect_info, categories
            )

            response = self.model.generate_content(prompt)
            questions_data = self._parse_generated_questions(response.text, sect, language)

            return questions_data

        except Exception as e:
            logger.error(f"Error generating questions with AI: {e}")
            return self._generate_basic_questions_from_content(sect, language, religion, difficulty, question_count)

    def _create_question_generation_prompt(self, content: str, sect: str, language: str,
                                         difficulty: str, question_count: int,
                                         sect_info: Dict[str, Any], categories: List[str] = None) -> str:
        """Create prompt for AI question generation"""
        return f"""
Based on the following Jain religious content, create {question_count} assessment questions for the {sect_info.get('display_name', sect)} tradition.

CONTENT TO BASE QUESTIONS ON:
{content}

REQUIREMENTS:
1. Create questions at {difficulty} difficulty level
2. Use sect-specific terminology:
   - Male ascetics: {sect_info.get('male_ascetic', 'Sadhu')}
   - Female ascetics: {sect_info.get('female_ascetic', 'Sadhvi')}
   - Male laypeople: {sect_info.get('male_layperson', 'Shravaka')}
   - Female laypeople: {sect_info.get('female_layperson', 'Shravika')}
3. Questions should be answerable ONLY from the provided content
4. Include multiple choice, true/false, and short answer questions
5. Provide explanations based on the content
6. Use {language} language for questions

QUESTION CATEGORIES TO FOCUS ON:
{', '.join(categories) if categories else 'principles, history, practices, philosophy'}

FORMAT RESPONSE AS JSON:
{{
  "questions": [
    {{
      "id": "q1",
      "type": "multiple_choice",
      "category": "principles",
      "difficulty": "{difficulty}",
      "question": "Question text here",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correct_answer": "Option A",
      "explanation": "Explanation based on provided content",
      "points": 2
    }}
  ]
}}

Generate diverse, content-based questions now:
"""

    def _parse_generated_questions(self, ai_response: str, sect: str, language: str) -> List[Dict[str, Any]]:
        """Parse AI-generated questions from JSON response"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in response")

            questions_data = json.loads(json_match.group())
            questions = questions_data.get('questions', [])

            # Validate and enhance questions
            validated_questions = []
            for i, q in enumerate(questions):
                validated_q = self._validate_and_enhance_question(q, i, sect, language)
                if validated_q:
                    validated_questions.append(validated_q)

            return validated_questions

        except Exception as e:
            logger.error(f"Error parsing generated questions: {e}")
            return []

    def _validate_and_enhance_question(self, question: Dict[str, Any], index: int,
                                     sect: str, language: str) -> Optional[Dict[str, Any]]:
        """Validate and enhance a single question"""
        try:
            # Required fields
            if not all(key in question for key in ['question', 'type']):
                return None

            # Generate ID if missing
            if 'id' not in question:
                question['id'] = f"q{index + 1}_{sect}_{datetime.now().strftime('%H%M%S')}"

            # Set default values
            question.setdefault('category', 'principles')
            question.setdefault('difficulty', 'medium')
            question.setdefault('points', self._calculate_question_points(question))
            question.setdefault('language', language)
            question.setdefault('sect', sect)

            # Validate question type specific fields
            if question['type'] == 'multiple_choice':
                if 'options' not in question or 'correct_answer' not in question:
                    return None
                if len(question['options']) < 2:
                    return None

            elif question['type'] == 'true_false':
                if 'correct_answer' not in question:
                    return None
                question['options'] = ['True', 'False']

            elif question['type'] == 'short_answer':
                if 'correct_answer' not in question:
                    question['correct_answer'] = "Answer based on uploaded content"

            # Apply sect-specific formatting
            question['question'] = self.content_formatter.format_content_for_sect(
                question['question'], sect, language
            )

            if 'explanation' in question:
                question['explanation'] = self.content_formatter.format_content_for_sect(
                    question['explanation'], sect, language
                )

            return question

        except Exception as e:
            logger.error(f"Error validating question: {e}")
            return None

    def _calculate_question_points(self, question: Dict[str, Any]) -> int:
        """Calculate points for a question based on type and difficulty"""
        base_points = {
            'multiple_choice': 1,
            'true_false': 1,
            'short_answer': 2,
            'scenario_based': 3
        }

        difficulty_multiplier = {
            'easy': 1,
            'medium': 1.5,
            'hard': 2
        }

        base = base_points.get(question.get('type', 'multiple_choice'), 1)
        multiplier = difficulty_multiplier.get(question.get('difficulty', 'medium'), 1)

        return int(base * multiplier)

    def _generate_basic_questions_from_content(self, sect: str, language: str, religion: str,
                                             difficulty: str, question_count: int) -> List[Dict[str, Any]]:
        """Fallback method to generate basic questions without AI"""
        questions = []

        # Get some content samples
        sample_queries = ["Jainism", "Tirthankara", "ahimsa", "karma"]
        content_found = False

        for query in sample_queries:
            docs = self.vector_store.query_vector_store(
                query=query,
                sect=sect,
                language=language,
                religion=religion,
                k=2
            )

            if docs:
                content_found = True
                # Create simple questions from content
                for i, doc in enumerate(docs):
                    if len(questions) >= question_count:
                        break

                    question = self._create_basic_question_from_text(
                        doc['content'], sect, language, difficulty, len(questions) + 1
                    )
                    if question:
                        questions.append(question)

            if len(questions) >= question_count:
                break

        return questions[:question_count] if content_found else []

    def _create_basic_question_from_text(self, text: str, sect: str, language: str,
                                       difficulty: str, question_num: int) -> Optional[Dict[str, Any]]:
        """Create a basic question from text content"""
        try:
            # Extract key concepts from text
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            if not sentences:
                return None

            # Pick a sentence with good content
            selected_sentence = None
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in ['jain', 'ahimsa', 'karma', 'dharma', 'moksha']):
                    selected_sentence = sentence
                    break

            if not selected_sentence:
                selected_sentence = sentences[0]

            # Create multiple choice question
            question_text = f"According to the uploaded content, what does the text say about {self._extract_key_concept(selected_sentence)}?"

            question = {
                'id': f"basic_q{question_num}_{sect}",
                'type': 'multiple_choice',
                'category': 'principles',
                'difficulty': difficulty,
                'language': language,
                'sect': sect,
                'question': question_text,
                'options': [
                    selected_sentence,
                    "This information is not provided in the uploaded content",
                    "The content discusses a different concept",
                    "This is beyond the scope of the uploaded materials"
                ],
                'correct_answer': selected_sentence,
                'explanation': f"This information is directly stated in the uploaded content: {selected_sentence}",
                'points': self._calculate_question_points({'type': 'multiple_choice', 'difficulty': difficulty}),
                'based_on_content': True
            }

            # Apply sect formatting
            question = self.content_formatter.format_quiz_content({'questions': [question]}, sect, language)['questions'][0]

            return question

        except Exception as e:
            logger.error(f"Error creating basic question: {e}")
            return None

    def _extract_key_concept(self, text: str) -> str:
        """Extract key Jain concept from text"""
        concepts = ['ahimsa', 'karma', 'dharma', 'moksha', 'tirthankara', 'jina', 'ratnatraya']
        text_lower = text.lower()

        for concept in concepts:
            if concept in text_lower:
                return concept

        # Fallback: extract first meaningful word
        words = text.split()
        for word in words:
            if len(word) > 4 and word.lower() not in ['this', 'that', 'which', 'with', 'from']:
                return word.lower()

        return "this concept"

    def _create_fallback_assessment(self, sect: str, language: str,
                                   difficulty: str, question_count: int) -> Dict[str, Any]:
        """Create fallback assessment when no content is available"""
        # Basic Jain knowledge questions (not content-dependent)
        fallback_questions = self._get_fallback_questions(sect, language, difficulty)

        assessment_id = f"fallback_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{sect}"

        return {
            'id': assessment_id,
            'sect': sect,
            'language': language,
            'difficulty': difficulty,
            'question_count': min(len(fallback_questions), question_count),
            'created_timestamp': datetime.now().isoformat(),
            'time_limit_minutes': self._calculate_time_limit(question_count, difficulty),
            'questions': fallback_questions[:question_count],
            'scoring': self._create_scoring_rubric(fallback_questions[:question_count]),
            'based_on_uploaded_content': False,
            'warning': 'This assessment uses general knowledge as no uploaded content was found.'
        }

    def _get_fallback_questions(self, sect: str, language: str, difficulty: str) -> List[Dict[str, Any]]:
        """Get basic fallback questions for when no content is available"""
        sect_info = self.content_formatter.sect_terms.get(sect, {})

        questions = [
            {
                'id': f'fallback_1_{sect}',
                'type': 'multiple_choice',
                'category': 'principles',
                'difficulty': difficulty,
                'language': language,
                'sect': sect,
                'question': f'What is the primary principle emphasized by {sect_info.get("display_name", sect)} tradition?',
                'options': ['Ahimsa (Non-violence)', 'Wealth accumulation', 'Political power', 'Social status'],
                'correct_answer': 'Ahimsa (Non-violence)',
                'explanation': 'Ahimsa or non-violence is the fundamental principle of all Jain traditions.',
                'points': 2
            },
            {
                'id': f'fallback_2_{sect}',
                'type': 'true_false',
                'category': 'history',
                'difficulty': difficulty,
                'language': language,
                'sect': sect,
                'question': f'The {sect_info.get("display_name", sect)} tradition follows the teachings of Tirthankaras.',
                'options': ['True', 'False'],
                'correct_answer': 'True',
                'explanation': 'All Jain traditions follow the teachings of the 24 Tirthankaras.',
                'points': 1
            }
        ]

        # Apply sect-specific formatting
        formatted_questions = []
        for question in questions:
            formatted_q = self.content_formatter.format_quiz_content({'questions': [question]}, sect, language)['questions'][0]
            formatted_questions.append(formatted_q)

        return formatted_questions

    def score_assessment(self, assessment: Dict[str, Any],
                        answers: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score assessment and provide detailed feedback

        Args:
            assessment: Assessment data
            answers: User's answers {question_id: answer}

        Returns:
            Scoring results with feedback
        """
        try:
            total_points = 0
            earned_points = 0
            question_results = []
            category_scores = {}

            for question in assessment['questions']:
                question_id = question['id']
                user_answer = answers.get(question_id, '')
                correct_answer = question.get('correct_answer', '')

                # Score individual question
                is_correct = self._score_question(question, user_answer)
                question_points = question.get('points', 1)
                earned = question_points if is_correct else 0

                total_points += question_points
                earned_points += earned

                # Track category performance
                category = question.get('category', 'general')
                if category not in category_scores:
                    category_scores[category] = {'earned': 0, 'total': 0, 'questions': 0}

                category_scores[category]['earned'] += earned
                category_scores[category]['total'] += question_points
                category_scores[category]['questions'] += 1

                question_results.append({
                    'question_id': question_id,
                    'question': question['question'],
                    'user_answer': user_answer,
                    'correct_answer': correct_answer,
                    'is_correct': is_correct,
                    'points_earned': earned,
                    'points_possible': question_points,
                    'explanation': question.get('explanation', ''),
                    'category': category
                })

            # Calculate overall percentage
            percentage = (earned_points / total_points * 100) if total_points > 0 else 0

            # Determine knowledge level
            knowledge_level = self._determine_knowledge_level(percentage)

            # Generate feedback
            feedback = self._generate_assessment_feedback(
                percentage, category_scores, knowledge_level, assessment['sect']
            )

            return {
                'assessment_id': assessment['id'],
                'total_questions': len(assessment['questions']),
                'correct_answers': sum(1 for r in question_results if r['is_correct']),
                'total_points': total_points,
                'earned_points': earned_points,
                'percentage': round(percentage, 1),
                'knowledge_level': knowledge_level,
                'category_scores': category_scores,
                'question_results': question_results,
                'feedback': feedback,
                'completion_timestamp': datetime.now().isoformat(),
                'based_on_uploaded_content': assessment.get('based_on_uploaded_content', False)
            }

        except Exception as e:
            logger.error(f"Error scoring assessment: {e}")
            return {
                'error': str(e),
                'percentage': 0,
                'knowledge_level': 'beginner'
            }

    def _score_question(self, question: Dict[str, Any], user_answer: str) -> bool:
        """Score individual question"""
        question_type = question.get('type', 'multiple_choice')
        correct_answer = question.get('correct_answer', '').strip()
        user_answer = user_answer.strip()

        if question_type in ['multiple_choice', 'true_false']:
            return user_answer.lower() == correct_answer.lower()

        elif question_type == 'short_answer':
            # For short answers, check for key concepts
            return self._score_short_answer(user_answer, correct_answer)

        return False

    def _score_short_answer(self, user_answer: str, correct_answer: str) -> bool:
        """Score short answer questions with partial matching"""
        if not user_answer:
            return False

        # Simple keyword matching for now
        user_words = set(user_answer.lower().split())
        correct_words = set(correct_answer.lower().split())

        # Check for significant overlap
        overlap = len(user_words.intersection(correct_words))
        return overlap >= len(correct_words) * 0.3  # 30% overlap threshold

    def _determine_knowledge_level(self, percentage: float) -> str:
        """Determine knowledge level based on assessment score"""
        for level_code, level_info in self.knowledge_levels.items():
            try:
                score_range = level_info.get('score_range', [0, 100])

                # Safely convert to int/float, handling any type
                if isinstance(score_range, (list, tuple)) and len(score_range) >= 2:
                    # Convert to int first, then compare
                    try:
                        min_score = int(score_range[0])
                        max_score = int(score_range[1])
                    except (ValueError, TypeError):
                        # Try float conversion if int fails
                        min_score = float(str(score_range[0]))
                        max_score = float(str(score_range[1]))
                else:
                    continue  # Skip invalid entries

                if min_score <= percentage <= max_score:
                    return level_code

            except (ValueError, TypeError, IndexError) as e:
                # Log and skip if conversion fails
                logger.warning(f"Invalid score_range for {level_code}: {score_range}, error: {e}")
                continue

        return 'beginner'  # Default fallback

    def _generate_assessment_feedback(self, percentage: float,
                                    category_scores: Dict[str, Any],
                                    knowledge_level: str, sect: str) -> Dict[str, Any]:
        """Generate personalized feedback based on performance"""
        feedback = {
            'overall_performance': self._get_overall_feedback(percentage, knowledge_level),
            'strengths': [],
            'improvement_areas': [],
            'recommendations': [],
            'next_steps': []
        }

        # Analyze category performance
        for category, scores in category_scores.items():
            category_percentage = (scores['earned'] / scores['total'] * 100) if scores['total'] > 0 else 0

            if category_percentage >= 75:
                feedback['strengths'].append(f"Strong understanding of {category}")
            elif category_percentage < 50:
                feedback['improvement_areas'].append(f"Need improvement in {category}")

        # Generate recommendations
        if knowledge_level == 'beginner':
            feedback['recommendations'].extend([
                "Start with basic Jain principles and core concepts",
                f"Focus on fundamental teachings of the {sect} tradition",
                "Practice with easier content before advancing"
            ])
        elif knowledge_level == 'intermediate':
            feedback['recommendations'].extend([
                "Deepen your understanding of philosophical concepts",
                "Explore advanced practices and rituals",
                "Study comparative aspects with other Jain traditions"
            ])
        else:  # advanced
            feedback['recommendations'].extend([
                "Explore scholarly texts and commentaries",
                "Consider teaching or mentoring others",
                "Engage with complex philosophical discussions"
            ])

        # Next steps based on performance
        if percentage < 60:
            feedback['next_steps'] = [
                "Review uploaded content more thoroughly",
                "Take additional assessments to track progress",
                "Focus on weak areas identified in this assessment"
            ]
        else:
            feedback['next_steps'] = [
                "Continue with advanced topics",
                "Explore practical applications",
                "Share knowledge with others"
            ]

        return feedback

    def _get_overall_feedback(self, percentage: float, knowledge_level: str) -> str:
        """Get overall performance feedback"""
        if percentage >= 90:
            return "Excellent performance! You have a strong grasp of the material."
        elif percentage >= 75:
            return "Good work! You understand most concepts well."
        elif percentage >= 60:
            return "Fair performance. There's room for improvement in some areas."
        else:
            return "Keep practicing! Focus on reviewing the basic concepts."

    def _calculate_time_limit(self, question_count: int, difficulty: str) -> int:
        """Calculate appropriate time limit for assessment"""
        base_time_per_question = {
            'easy': 1.5,
            'medium': 2,
            'hard': 3
        }

        minutes = question_count * base_time_per_question.get(difficulty, 2)
        return max(int(minutes), 5)  # Minimum 5 minutes

    def _create_scoring_rubric(self, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create scoring rubric for the assessment"""
        total_points = sum(q.get('points', 1) for q in questions)

        return {
            'total_points': total_points,
            'grading_scale': {
                'excellent': {'min': 90, 'max': 100},
                'good': {'min': 75, 'max': 89},
                'fair': {'min': 60, 'max': 74},
                'needs_improvement': {'min': 0, 'max': 59}
            },
            'point_distribution': {
                q['category']: sum(quest.get('points', 1) for quest in questions if quest.get('category') == q['category'])
                for q in questions
            }
        }