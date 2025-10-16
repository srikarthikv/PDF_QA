"""
Tutor Engine for Jain Learning System
Provides context-aware tutoring with personalized guidance
"""

import os
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import yaml
from pathlib import Path

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from .rag_pipeline import RAGPipeline
from .content_formatter import ContentFormatter
from .user_profiler import UserProfiler
from .progress_tracker import ProgressTracker
from .vector_store_manager import VectorStoreManager

logger = logging.getLogger(__name__)

class TutorEngine:
    """Context-aware tutoring engine with personalized learning support"""

    def __init__(self, config_path: str = None):
        """Initialize tutor engine with configuration"""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "config.yaml"

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # Initialize components
        self.rag_pipeline = RAGPipeline(config_path)
        self.content_formatter = ContentFormatter(config_path)
        self.user_profiler = UserProfiler(config_path)
        self.progress_tracker = ProgressTracker()
        self.vector_store_manager = VectorStoreManager(self.config)

        # Initialize Gemini if available
        self.gemini_available = GEMINI_AVAILABLE and 'GOOGLE_API_KEY' in os.environ
        if self.gemini_available:
            genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
            self.model = genai.GenerativeModel('gemini-1.5-pro')

        # Tutoring session memory (in-memory for simplicity)
        self.session_contexts = {}

    def start_tutoring_session(self, user_id: str, topic: str = None,
                             learning_objective: str = None) -> Dict[str, Any]:
        """Start a new tutoring session for user"""
        try:
            # Get user profile
            user_profile = self.user_profiler.get_user_profile(user_id)
            if not user_profile:
                user_profile = self.user_profiler.create_user_profile(user_id)

            sect = user_profile.get('sect', 'digambara')
            language = user_profile.get('language', 'en')

            # Check if content is available
            vector_store = self.vector_store_manager.get_vector_store(sect, language)
            has_content = vector_store is not None

            # Initialize session context
            session_id = f"session_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            session_context = {
                'session_id': session_id,
                'user_id': user_id,
                'user_profile': user_profile,
                'start_time': datetime.now().isoformat(),
                'topic': topic,
                'learning_objective': learning_objective,
                'interaction_history': [],
                'current_difficulty': user_profile.get('knowledge_level', 'beginner'),
                'topics_covered': [],
                'questions_asked': [],
                'misconceptions_identified': [],
                'learning_gaps': [],
                'has_content': has_content,
                'session_progress': {
                    'concepts_learned': 0,
                    'questions_answered': 0,
                    'engagement_level': 'medium'
                }
            }

            self.session_contexts[session_id] = session_context

            # Generate opening message based on content availability
            if has_content:
                opening_message = self._generate_opening_message(user_profile, topic, learning_objective)
                topic_suggestions = self._suggest_initial_topics(user_profile) if not topic else []
            else:
                opening_message = (
                    f"Welcome to the {sect.title()} learning path! "
                    f"I'm your AI tutor ready to help you explore Jain philosophy.\n\n"
                    f"⚠️ I notice you haven't uploaded any content yet. "
                    f"To provide personalized answers based on specific texts, please upload PDF documents first by going back to the dashboard.\n\n"
                    f"I can still answer general questions about Jainism, but my responses will be limited without your uploaded content."
                )
                topic_suggestions = []

            return {
                'session_id': session_id,
                'message': opening_message,
                'topic_suggestions': topic_suggestions,
                'user_level': user_profile.get('knowledge_level', 'beginner'),
                'preferred_language': user_profile.get('language', 'en'),
                'sect': sect,
                'has_content': has_content
            }

        except Exception as e:
            logger.error(f"Error starting tutoring session: {e}")
            return {
                'error': str(e),
                'message': 'I apologize, but I encountered an error starting your tutoring session. Please try again.'
            }

    def process_user_input(self, session_id: str, user_input: str,
                          input_type: str = 'question') -> Dict[str, Any]:
        """Process user input and generate appropriate response"""
        try:
            session_context = self.session_contexts.get(session_id)
            if not session_context:
                return {
                    'error': 'Session not found',
                    'message': 'Your tutoring session has expired. Please start a new session.'
                }

            user_profile = session_context['user_profile']
            sect = user_profile.get('sect', 'digambara')
            language = user_profile.get('language', 'en')

            # Record interaction
            interaction = {
                'timestamp': datetime.now().isoformat(),
                'user_input': user_input,
                'input_type': input_type,
                'response': None  # Will be filled after generating response
            }

            # Analyze user input
            input_analysis = self._analyze_user_input(user_input, session_context)

            # Generate appropriate response based on input type and analysis
            if input_type == 'question':
                response = self._handle_question(user_input, session_context, input_analysis)
            elif input_type == 'answer':
                response = self._handle_answer(user_input, session_context, input_analysis)
            elif input_type == 'topic_request':
                response = self._handle_topic_request(user_input, session_context)
            else:
                response = self._handle_general_input(user_input, session_context, input_analysis)

            # Update session context
            interaction['response'] = response
            session_context['interaction_history'].append(interaction)

            # Update session progress
            self._update_session_progress(session_context, user_input, response)

            # Check for learning opportunities
            learning_suggestions = self._identify_learning_opportunities(session_context)

            # Generate follow-up questions or suggestions
            follow_up = self._generate_follow_up(session_context, response, input_analysis)

            return {
                'session_id': session_id,
                'response': response,
                'follow_up': follow_up,
                'learning_suggestions': learning_suggestions,
                'session_progress': session_context['session_progress'],
                'topics_covered': session_context['topics_covered']
            }

        except Exception as e:
            logger.error(f"Error processing user input: {e}")
            return {
                'error': str(e),
                'response': 'I apologize, but I encountered an error processing your input. Could you please try again?'
            }

    def _generate_opening_message(self, user_profile: Dict[str, Any],
                                topic: str = None, learning_objective: str = None) -> str:
        """Generate personalized opening message for tutoring session"""
        sect = user_profile.get('sect', 'digambara')
        language = user_profile.get('language', 'en')
        knowledge_level = user_profile.get('knowledge_level', 'beginner')

        # Get sect-specific greeting
        greeting = self.content_formatter.get_sect_specific_greeting(sect, language)

        # Customize based on topic and objective
        if topic:
            greeting += f"\n\nI see you'd like to learn about {topic}. "

        if learning_objective:
            greeting += f"Your goal is to {learning_objective}. "

        # Add level-appropriate guidance
        if knowledge_level == 'beginner':
            greeting += "I'll start with the fundamentals and guide you step by step. Feel free to ask any questions!"
        elif knowledge_level == 'intermediate':
            greeting += "I'll help deepen your understanding and explore more advanced concepts. What would you like to focus on?"
        else:  # advanced
            greeting += "I'm here to discuss complex topics and support your advanced studies. What philosophical or scholarly question interests you today?"

        return greeting

    def _suggest_initial_topics(self, user_profile: Dict[str, Any]) -> List[str]:
        """Suggest initial topics based on user profile"""
        knowledge_level = user_profile.get('knowledge_level', 'beginner')
        sect = user_profile.get('sect', 'digambara')
        interests = user_profile.get('interests', [])

        # Base topics by knowledge level
        if knowledge_level == 'beginner':
            base_topics = [
                'What is Jainism?',
                'The principle of Ahimsa',
                'Who are the Tirthankaras?',
                'Basic Jain practices'
            ]
        elif knowledge_level == 'intermediate':
            base_topics = [
                'Understanding Karma theory',
                'The path to Moksha',
                'Jain meditation practices',
                'Comparative philosophy'
            ]
        else:  # advanced
            base_topics = [
                'Advanced metaphysics',
                'Scholarly interpretations',
                'Historical developments',
                'Contemporary applications'
            ]

        # Add sect-specific topics
        sect_info = self.content_formatter.sect_terms.get(sect, {})
        sect_topics = [
            f'{sect_info.get("display_name", sect)} traditions',
            f'Role of {sect_info.get("male_ascetic", "Sadhu")}s',
            f'Unique {sect} practices'
        ]

        # Combine and personalize
        all_topics = base_topics + sect_topics

        # Filter based on interests if available
        if interests:
            filtered_topics = []
            for topic in all_topics:
                if any(interest.lower() in topic.lower() for interest in interests):
                    filtered_topics.append(topic)
            if filtered_topics:
                return filtered_topics[:5]

        return all_topics[:5]

    def _analyze_user_input(self, user_input: str, session_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user input for intent, difficulty, and learning needs"""
        analysis = {
            'intent': 'question',  # question, clarification, challenge, agreement
            'complexity_level': 'medium',  # low, medium, high
            'emotional_tone': 'neutral',  # confused, frustrated, excited, neutral
            'knowledge_indicators': [],
            'misconceptions': [],
            'learning_gaps': []
        }

        input_lower = user_input.lower()

        # Detect intent
        if any(word in input_lower for word in ['what', 'how', 'why', 'when', 'where', 'who']):
            analysis['intent'] = 'question'
        elif any(word in input_lower for word in ['explain', 'clarify', 'mean', 'understand']):
            analysis['intent'] = 'clarification'
        elif any(word in input_lower for word in ['but', 'however', 'disagree', 'wrong']):
            analysis['intent'] = 'challenge'
        elif any(word in input_lower for word in ['yes', 'right', 'correct', 'understand']):
            analysis['intent'] = 'agreement'

        # Assess complexity
        complex_terms = ['metaphysics', 'epistemology', 'ontology', 'philosophy', 'doctrine']
        if any(term in input_lower for term in complex_terms):
            analysis['complexity_level'] = 'high'
        elif len(user_input.split()) > 20:
            analysis['complexity_level'] = 'high'
        elif len(user_input.split()) < 5:
            analysis['complexity_level'] = 'low'

        # Detect emotional indicators
        if any(word in input_lower for word in ['confused', 'don\'t understand', 'hard', 'difficult']):
            analysis['emotional_tone'] = 'confused'
        elif any(word in input_lower for word in ['frustrated', 'annoying', 'stupid']):
            analysis['emotional_tone'] = 'frustrated'
        elif any(word in input_lower for word in ['exciting', 'amazing', 'wonderful', 'love']):
            analysis['emotional_tone'] = 'excited'

        # Identify knowledge indicators
        jain_terms = ['ahimsa', 'karma', 'moksha', 'tirthankara', 'jina', 'ratnatraya']
        for term in jain_terms:
            if term in input_lower:
                analysis['knowledge_indicators'].append(term)

        return analysis

    def _handle_question(self, question: str, session_context: Dict[str, Any],
                        input_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Handle user questions with context-aware responses"""
        user_profile = session_context['user_profile']
        sect = user_profile.get('sect', 'digambara')
        language = user_profile.get('language', 'en')

        # Get vector store for this sect and language
        vector_store = self.vector_store_manager.get_vector_store(sect, language)

        # Use RAG pipeline to get answer from uploaded content
        rag_response = self.rag_pipeline.answer_question(
            question=question,
            vector_store=vector_store,
            sect=sect,
            language=language,
            user_context={
                'knowledge_level': session_context['current_difficulty'],
                'age_group': user_profile.get('age_group', 'adult'),
                'learning_history': session_context['topics_covered'],
                'session_context': session_context['interaction_history'][-5:]  # Last 5 interactions
            }
        )

        # Enhance response with tutoring elements
        enhanced_response = self._enhance_response_for_tutoring(
            rag_response, question, session_context, input_analysis
        )

        # Update session context
        session_context['questions_asked'].append(question)
        if rag_response.get('related_topics'):
            session_context['topics_covered'].extend(rag_response['related_topics'])

        return enhanced_response

    def _handle_answer(self, answer: str, session_context: Dict[str, Any],
                      input_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Handle user answers to tutor questions"""
        # This would typically involve checking the answer against expected responses
        # For now, provide encouragement and follow-up

        if input_analysis['intent'] == 'agreement':
            response_text = "Great! I'm glad that makes sense. "
        elif input_analysis['emotional_tone'] == 'confused':
            response_text = "I understand this might be challenging. Let me explain it differently. "
        else:
            response_text = "Thank you for sharing your thoughts. "

        # Provide feedback and next steps
        response_text += "Let's explore this concept further. "

        return {
            'main_answer': response_text,
            'type': 'feedback',
            'encouragement': self._generate_encouragement(session_context, input_analysis)
        }

    def _handle_topic_request(self, topic_request: str, session_context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle requests to learn about specific topics"""
        user_profile = session_context['user_profile']
        sect = user_profile.get('sect', 'digambara')
        language = user_profile.get('language', 'en')

        # Extract topic from request
        topic = topic_request.replace('tell me about', '').replace('explain', '').strip()

        # Get vector store for this sect and language
        vector_store = self.vector_store_manager.get_vector_store(sect, language)

        # Get comprehensive information about the topic
        topic_response = self.rag_pipeline.answer_question(
            question=f"Explain {topic} in detail",
            vector_store=vector_store,
            sect=sect,
            language=language,
            user_context={
                'knowledge_level': session_context['current_difficulty'],
                'age_group': user_profile.get('age_group', 'adult')
            }
        )

        # Structure as a learning session
        structured_response = {
            'main_answer': topic_response.get('answer', f'Let me explain {topic} for you.'),
            'key_points': topic_response.get('key_points', []),
            'examples': topic_response.get('examples', []),
            'type': 'topic_explanation',
            'topic': topic
        }

        # Add to session topics
        session_context['topics_covered'].append(topic)

        return structured_response

    def _handle_general_input(self, user_input: str, session_context: Dict[str, Any],
                             input_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general conversational input"""
        if input_analysis['emotional_tone'] == 'confused':
            response = "I can see you might need some clarification. What specific part would you like me to explain more clearly?"
        elif input_analysis['emotional_tone'] == 'frustrated':
            response = "Learning can be challenging sometimes. Let's take a step back and approach this differently. What would help you most right now?"
        elif input_analysis['emotional_tone'] == 'excited':
            response = "I'm so glad you're enthusiastic about learning! Your excitement will help you grasp these concepts quickly."
        else:
            response = "I appreciate you sharing that with me. How can I help you with your learning today?"

        return {
            'main_answer': response,
            'type': 'conversational',
            'supportive': True
        }

    def _enhance_response_for_tutoring(self, rag_response: Dict[str, Any], question: str,
                                     session_context: Dict[str, Any],
                                     input_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance RAG response with tutoring elements"""
        enhanced = rag_response.copy()

        # Add tutoring elements
        enhanced['tutoring_elements'] = {
            'difficulty_check': self._should_adjust_difficulty(input_analysis, session_context),
            'comprehension_questions': self._generate_comprehension_questions(rag_response, session_context),
            'real_world_connections': self._suggest_real_world_applications(rag_response, session_context),
            'memory_aids': self._suggest_memory_techniques(rag_response),
            'encouragement': self._generate_encouragement(session_context, input_analysis)
        }

        # Adjust response complexity if needed
        if enhanced['tutoring_elements']['difficulty_check'] == 'too_hard':
            enhanced['simplified_explanation'] = self._simplify_explanation(rag_response['answer'])
        elif enhanced['tutoring_elements']['difficulty_check'] == 'too_easy':
            enhanced['advanced_details'] = self._add_advanced_details(rag_response, session_context)

        return enhanced

    def _should_adjust_difficulty(self, input_analysis: Dict[str, Any],
                                session_context: Dict[str, Any]) -> str:
        """Determine if difficulty should be adjusted"""
        if input_analysis['emotional_tone'] == 'confused':
            return 'too_hard'
        elif input_analysis['complexity_level'] == 'high' and session_context['current_difficulty'] == 'beginner':
            return 'too_hard'
        elif input_analysis['complexity_level'] == 'low' and session_context['current_difficulty'] == 'advanced':
            return 'too_easy'
        else:
            return 'appropriate'

    def _generate_comprehension_questions(self, rag_response: Dict[str, Any],
                                        session_context: Dict[str, Any]) -> List[str]:
        """Generate questions to check comprehension"""
        questions = []

        # Extract key concepts from response
        key_points = rag_response.get('key_points', [])

        if key_points:
            # Create questions based on key points
            for point in key_points[:2]:  # Limit to 2 questions
                questions.append(f"Can you explain what '{point}' means in your own words?")

        # Add application question
        if rag_response.get('examples'):
            questions.append("How might you apply this concept in daily life?")

        return questions

    def _suggest_real_world_applications(self, rag_response: Dict[str, Any],
                                       session_context: Dict[str, Any]) -> List[str]:
        """Suggest real-world applications of concepts"""
        applications = []

        # Basic applications for common Jain concepts
        content = rag_response.get('answer', '').lower()

        if 'ahimsa' in content:
            applications.append("Practice ahimsa by choosing cruelty-free products")
            applications.append("Apply non-violence in communication and conflict resolution")

        if 'karma' in content:
            applications.append("Make mindful choices knowing they have consequences")
            applications.append("Take responsibility for your actions and their effects")

        if 'meditation' in content:
            applications.append("Use meditation techniques for stress management")
            applications.append("Practice mindfulness in daily activities")

        return applications[:3]  # Limit to 3 suggestions

    def _suggest_memory_techniques(self, rag_response: Dict[str, Any]) -> List[str]:
        """Suggest techniques to remember the content"""
        techniques = []

        key_points = rag_response.get('key_points', [])
        if len(key_points) > 1:
            techniques.append(f"Create an acronym using the first letters: {', '.join(key_points[:3])}")

        if rag_response.get('examples'):
            techniques.append("Associate the concepts with the examples provided")

        techniques.append("Write a summary in your own words")
        techniques.append("Teach this concept to someone else")

        return techniques

    def _generate_encouragement(self, session_context: Dict[str, Any],
                              input_analysis: Dict[str, Any]) -> str:
        """Generate appropriate encouragement based on context"""
        progress = session_context['session_progress']

        if input_analysis['emotional_tone'] == 'confused':
            return "Don't worry - these concepts take time to understand. You're making progress by asking questions!"

        elif input_analysis['emotional_tone'] == 'frustrated':
            return "I understand this can be challenging. Remember, every expert was once a beginner. You're doing great by persisting!"

        elif progress['concepts_learned'] > 3:
            return "Excellent progress! You've covered several important concepts in this session."

        elif progress['questions_answered'] > 5:
            return "Great engagement! Your questions show you're thinking deeply about these topics."

        else:
            return "Keep up the good work! Your dedication to learning is admirable."

    def _simplify_explanation(self, explanation: str) -> str:
        """Simplify explanation for better understanding"""
        # This is a simple implementation - could be enhanced with AI
        simple_replacements = {
            'philosophy': 'way of thinking',
            'metaphysics': 'nature of reality',
            'epistemology': 'how we know things',
            'doctrine': 'teaching',
            'principle': 'important rule'
        }

        simplified = explanation
        for complex_term, simple_term in simple_replacements.items():
            simplified = simplified.replace(complex_term, simple_term)

        return simplified

    def _add_advanced_details(self, rag_response: Dict[str, Any],
                            session_context: Dict[str, Any]) -> str:
        """Add advanced details for more experienced learners"""
        # This could query for more advanced content or add scholarly perspectives
        return "For deeper understanding, consider exploring the historical development of this concept and its various interpretations across different Jain traditions."

    def _update_session_progress(self, session_context: Dict[str, Any],
                               user_input: str, response: Dict[str, Any]):
        """Update session progress metrics"""
        progress = session_context['session_progress']

        # Count questions answered
        if '?' in user_input:
            progress['questions_answered'] += 1

        # Count concepts learned (simplified heuristic)
        if response.get('key_points'):
            progress['concepts_learned'] += len(response['key_points'])

        # Assess engagement level
        if len(user_input) > 50:  # Detailed input suggests high engagement
            progress['engagement_level'] = 'high'
        elif len(user_input) < 10:  # Short responses suggest low engagement
            progress['engagement_level'] = 'low'
        else:
            progress['engagement_level'] = 'medium'

    def _identify_learning_opportunities(self, session_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify opportunities for enhanced learning"""
        opportunities = []

        topics_covered = session_context['topics_covered']
        user_profile = session_context['user_profile']

        # Suggest related topics
        if topics_covered:
            opportunities.append({
                'type': 'related_topic',
                'title': 'Explore Related Concepts',
                'description': f"Based on your interest in {topics_covered[-1]}, you might enjoy learning about related topics."
            })

        # Suggest practice
        if session_context['session_progress']['concepts_learned'] > 2:
            opportunities.append({
                'type': 'practice',
                'title': 'Practice with Assessment',
                'description': 'Test your understanding with a quick quiz on today\'s topics.'
            })

        # Suggest deeper study
        if session_context['session_progress']['engagement_level'] == 'high':
            opportunities.append({
                'type': 'deeper_study',
                'title': 'Advanced Study',
                'description': 'You seem very engaged! Consider exploring advanced materials on this topic.'
            })

        return opportunities

    def _generate_follow_up(self, session_context: Dict[str, Any],
                          response: Dict[str, Any],
                          input_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate appropriate follow-up questions or activities"""
        follow_up = {
            'questions': [],
            'activities': [],
            'next_steps': []
        }

        # Generate questions based on response content
        if response.get('key_points'):
            follow_up['questions'].append(
                f"Which of these concepts would you like to explore further: {', '.join(response['key_points'][:3])}?"
            )

        # Suggest activities
        if response.get('examples'):
            follow_up['activities'].append("Try to think of your own example of this concept")

        # Next steps based on engagement
        if session_context['session_progress']['engagement_level'] == 'high':
            follow_up['next_steps'].append("Continue with advanced topics")
        else:
            follow_up['next_steps'].append("Review and practice current concepts")

        return follow_up

    def end_tutoring_session(self, session_id: str) -> Dict[str, Any]:
        """End tutoring session and provide summary"""
        try:
            session_context = self.session_contexts.get(session_id)
            if not session_context:
                return {'error': 'Session not found'}

            # Calculate session duration
            start_time = datetime.fromisoformat(session_context['start_time'])
            duration_minutes = int((datetime.now() - start_time).total_seconds() / 60)

            # Create session summary
            summary = {
                'session_id': session_id,
                'duration_minutes': duration_minutes,
                'topics_covered': list(set(session_context['topics_covered'])),  # Remove duplicates
                'questions_asked': len(session_context['questions_asked']),
                'concepts_learned': session_context['session_progress']['concepts_learned'],
                'engagement_level': session_context['session_progress']['engagement_level'],
                'recommendations': self._generate_session_recommendations(session_context)
            }

            # Update user progress
            user_id = session_context['user_id']
            session_data = {
                'duration_minutes': duration_minutes,
                'topics_covered': session_context['topics_covered'],
                'activities_completed': ['tutoring'],
                'satisfaction_rating': 4,  # Default good rating
                'timestamp': datetime.now().isoformat()
            }

            self.progress_tracker.record_study_session(user_id, session_data)

            # Clean up session context
            del self.session_contexts[session_id]

            return {
                'summary': summary,
                'message': self._generate_closing_message(summary)
            }

        except Exception as e:
            logger.error(f"Error ending tutoring session: {e}")
            return {'error': str(e)}

    def _generate_session_recommendations(self, session_context: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on tutoring session"""
        recommendations = []

        progress = session_context['session_progress']
        topics_covered = session_context['topics_covered']

        if progress['concepts_learned'] > 5:
            recommendations.append("You covered a lot of material! Consider taking an assessment to reinforce your learning.")

        if progress['engagement_level'] == 'high':
            recommendations.append("Your engagement was excellent! Continue with more advanced topics.")

        if len(topics_covered) > 3:
            recommendations.append("You explored multiple topics. Take time to review and connect the concepts.")

        if progress['questions_answered'] > 10:
            recommendations.append("You asked many great questions! Consider sharing your insights with others.")

        return recommendations

    def _generate_closing_message(self, summary: Dict[str, Any]) -> str:
        """Generate closing message for tutoring session"""
        message = f"Great session! You spent {summary['duration_minutes']} minutes learning "

        if summary['topics_covered']:
            message += f"about {', '.join(summary['topics_covered'][:3])}"
            if len(summary['topics_covered']) > 3:
                message += f" and {len(summary['topics_covered']) - 3} other topics"

        message += f". You asked {summary['questions_asked']} questions and showed {summary['engagement_level']} engagement. "

        message += "Keep up the excellent work in your learning journey!"

        return message