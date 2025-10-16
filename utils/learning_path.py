"""
Learning Path Generator for Jain Learning System
Creates personalized learning paths based on user profile, sect, and uploaded content
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import yaml
import random

from .vector_store_manager import VectorStoreManager
from .content_formatter import ContentFormatter

logger = logging.getLogger(__name__)

class LearningPathGenerator:
    """Generates personalized learning paths for users"""

    def __init__(self, config_path: str = None):
        """Initialize learning path generator with configuration"""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "config.yaml"

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.paths_dir = Path(__file__).parent.parent / "data" / "learning_paths"
        self.paths_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.vector_store = VectorStoreManager(self.config)
        self.content_formatter = ContentFormatter(config_path)

        # Learning path templates
        self.path_templates = self._initialize_path_templates()

    def generate_learning_path(self, user_profile: Dict[str, Any],
                             content_availability: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate personalized learning path for user

        Args:
            user_profile: User's profile data
            content_availability: Available content analysis

        Returns:
            Personalized learning path with milestones and activities
        """
        try:
            user_id = user_profile.get('user_id', 'anonymous')
            sect = user_profile.get('sect', 'digambara')
            language = user_profile.get('language', 'en')
            knowledge_level = user_profile.get('knowledge_level', 'beginner')
            age_group = user_profile.get('age_group', 'adult')

            # Analyze available content
            if not content_availability:
                content_availability = self._analyze_content_availability(sect, language)

            # Select appropriate path template
            template = self._select_path_template(knowledge_level, age_group, sect)

            # Customize path based on user profile and available content
            customized_path = self._customize_learning_path(
                template, user_profile, content_availability
            )

            # Generate specific milestones and activities
            path = self._generate_detailed_path(customized_path, user_profile, content_availability)

            # Save learning path
            path_id = f"path_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            path['id'] = path_id
            path['user_id'] = user_id
            path['created_timestamp'] = datetime.now().isoformat()

            self._save_learning_path(path)

            return path

        except Exception as e:
            logger.error(f"Error generating learning path: {e}")
            return self._create_fallback_path(user_profile)

    def _initialize_path_templates(self) -> Dict[str, Any]:
        """Initialize learning path templates for different user types"""
        return {
            'beginner_child': {
                'duration_weeks': 8,
                'sessions_per_week': 2,
                'session_duration': 20,
                'focus_areas': ['basic_concepts', 'stories', 'simple_practices'],
                'difficulty_progression': ['easy', 'easy', 'medium'],
                'activities': ['storytelling', 'games', 'simple_quiz', 'art_activities']
            },
            'beginner_teen': {
                'duration_weeks': 12,
                'sessions_per_week': 3,
                'session_duration': 30,
                'focus_areas': ['principles', 'history', 'modern_relevance', 'practices'],
                'difficulty_progression': ['easy', 'medium', 'medium'],
                'activities': ['reading', 'discussions', 'quizzes', 'projects', 'peer_interaction']
            },
            'beginner_adult': {
                'duration_weeks': 16,
                'sessions_per_week': 3,
                'session_duration': 45,
                'focus_areas': ['fundamentals', 'philosophy', 'practices', 'sect_specifics'],
                'difficulty_progression': ['easy', 'medium', 'hard'],
                'activities': ['comprehensive_reading', 'deep_analysis', 'practical_application', 'assessments']
            },
            'intermediate_teen': {
                'duration_weeks': 14,
                'sessions_per_week': 3,
                'session_duration': 40,
                'focus_areas': ['advanced_principles', 'comparative_study', 'practical_ethics', 'leadership'],
                'difficulty_progression': ['medium', 'hard', 'hard'],
                'activities': ['research', 'debates', 'community_service', 'mentoring_juniors']
            },
            'intermediate_adult': {
                'duration_weeks': 20,
                'sessions_per_week': 4,
                'session_duration': 60,
                'focus_areas': ['deep_philosophy', 'advanced_practices', 'scholarly_texts', 'teaching'],
                'difficulty_progression': ['medium', 'hard', 'expert'],
                'activities': ['scholarly_reading', 'original_research', 'teaching_others', 'community_leadership']
            },
            'advanced_adult': {
                'duration_weeks': 24,
                'sessions_per_week': 4,
                'session_duration': 75,
                'focus_areas': ['mastery', 'innovation', 'research', 'mentorship'],
                'difficulty_progression': ['hard', 'expert', 'expert'],
                'activities': ['original_research', 'publishing', 'mentoring', 'community_building']
            }
        }

    def _select_path_template(self, knowledge_level: str, age_group: str, sect: str) -> Dict[str, Any]:
        """Select appropriate path template based on user characteristics"""
        template_key = f"{knowledge_level}_{age_group}"

        # Handle special cases
        if age_group == 'senior':
            template_key = f"{knowledge_level}_adult"  # Use adult template for seniors
        elif knowledge_level == 'advanced' and age_group in ['child', 'teen']:
            template_key = f"intermediate_{age_group}"  # Cap difficulty for younger users

        template = self.path_templates.get(template_key)

        if not template:
            # Fallback to beginner adult
            template = self.path_templates['beginner_adult']

        return template.copy()

    def _analyze_content_availability(self, sect: str, language: str) -> Dict[str, Any]:
        """Analyze what content is available for the user's sect and language"""
        try:
            stats = self.vector_store.get_collection_stats(sect, language, 'jainism')

            if not stats or stats.get('document_count', 0) == 0:
                return {
                    'has_content': False,
                    'document_count': 0,
                    'topics_available': [],
                    'content_quality': 'none'
                }

            # Get available topics/sources
            sources = self.vector_store.get_unique_metadata_values(sect, language, 'jainism', 'source')

            # Sample content to analyze topics
            sample_queries = ['principles', 'practices', 'history', 'philosophy', 'stories']
            available_topics = []

            for query in sample_queries:
                docs = self.vector_store.query_vector_store(
                    query=query, sect=sect, language=language, religion='jainism', k=1
                )
                if docs and docs[0].get('relevance_score', 0) > 0.3:
                    available_topics.append(query)

            # Determine content quality
            doc_count = stats.get('document_count', 0)
            if doc_count > 100:
                quality = 'excellent'
            elif doc_count > 50:
                quality = 'good'
            elif doc_count > 10:
                quality = 'fair'
            else:
                quality = 'limited'

            return {
                'has_content': True,
                'document_count': doc_count,
                'sources': sources,
                'topics_available': available_topics,
                'content_quality': quality
            }

        except Exception as e:
            logger.error(f"Error analyzing content availability: {e}")
            return {'has_content': False, 'error': str(e)}

    def _customize_learning_path(self, template: Dict[str, Any], user_profile: Dict[str, Any],
                               content_availability: Dict[str, Any]) -> Dict[str, Any]:
        """Customize learning path template based on user profile and content"""

        customized = template.copy()

        # Adjust based on content availability
        if not content_availability.get('has_content', False):
            # Reduce reliance on uploaded content
            customized['content_based_activities'] = 0.2  # 20% content-based
            customized['general_activities'] = 0.8  # 80% general knowledge
            customized['warning'] = 'Limited to general Jain knowledge - upload content for personalized experience'
        else:
            content_quality = content_availability.get('content_quality', 'fair')
            if content_quality in ['excellent', 'good']:
                customized['content_based_activities'] = 0.8
                customized['general_activities'] = 0.2
            else:
                customized['content_based_activities'] = 0.5
                customized['general_activities'] = 0.5

        # Adjust for user preferences
        learning_style = user_profile.get('learning_style', 'mixed')
        if learning_style == 'visual':
            customized['activities'].extend(['infographics', 'mind_maps', 'visual_aids'])
        elif learning_style == 'interactive':
            customized['activities'].extend(['discussions', 'role_play', 'simulations'])

        # Adjust session parameters based on user patterns
        learning_patterns = user_profile.get('learning_patterns', {})
        if 'optimal_session_length' in learning_patterns:
            customized['session_duration'] = learning_patterns['optimal_session_length']

        # Factor in user goals
        goals = user_profile.get('goals', [])
        if 'teaching' in goals:
            customized['focus_areas'].append('pedagogical_skills')
            customized['activities'].extend(['lesson_planning', 'student_assessment'])
        elif 'research' in goals:
            customized['focus_areas'].append('research_methodology')
            customized['activities'].extend(['literature_review', 'data_analysis'])

        return customized

    def _generate_detailed_path(self, template: Dict[str, Any], user_profile: Dict[str, Any],
                              content_availability: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed learning path with specific milestones and activities"""

        sect = user_profile.get('sect', 'digambara')
        language = user_profile.get('language', 'en')
        knowledge_level = user_profile.get('knowledge_level', 'beginner')

        path = {
            'user_id': user_profile.get('user_id'),
            'sect': sect,
            'language': language,
            'knowledge_level': knowledge_level,
            'template_used': template,
            'estimated_duration_weeks': template['duration_weeks'],
            'sessions_per_week': template['sessions_per_week'],
            'session_duration_minutes': template['session_duration'],
            'total_sessions': template['duration_weeks'] * template['sessions_per_week'],
            'phases': [],
            'milestones': [],
            'resources_needed': [],
            'assessment_points': [],
            'adaptive_elements': []
        }

        # Generate learning phases
        phases = self._create_learning_phases(template, user_profile, content_availability)
        path['phases'] = phases

        # Generate milestones
        milestones = self._create_milestones(phases, template)
        path['milestones'] = milestones

        # Generate assessment points
        assessments = self._create_assessment_schedule(template, phases)
        path['assessment_points'] = assessments

        # Define resources needed
        path['resources_needed'] = self._identify_required_resources(template, content_availability)

        # Add adaptive elements
        path['adaptive_elements'] = self._define_adaptive_elements(template, user_profile)

        return path

    def _create_learning_phases(self, template: Dict[str, Any], user_profile: Dict[str, Any],
                              content_availability: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create detailed learning phases"""

        phases = []
        total_weeks = template['duration_weeks']
        focus_areas = template['focus_areas']
        difficulty_progression = template['difficulty_progression']

        weeks_per_phase = max(1, total_weeks // len(focus_areas))

        for i, focus_area in enumerate(focus_areas):
            phase_start_week = i * weeks_per_phase + 1
            phase_end_week = min((i + 1) * weeks_per_phase, total_weeks)

            # Determine difficulty for this phase
            difficulty_index = min(i, len(difficulty_progression) - 1)
            difficulty = difficulty_progression[difficulty_index]

            # Create phase details
            phase = {
                'phase_number': i + 1,
                'title': self._get_phase_title(focus_area, user_profile.get('sect')),
                'focus_area': focus_area,
                'difficulty': difficulty,
                'start_week': phase_start_week,
                'end_week': phase_end_week,
                'duration_weeks': phase_end_week - phase_start_week + 1,
                'learning_objectives': self._get_learning_objectives(focus_area, user_profile.get('sect')),
                'activities': self._get_phase_activities(focus_area, template['activities'], content_availability),
                'topics': self._get_phase_topics(focus_area, user_profile.get('sect')),
                'success_criteria': self._get_success_criteria(focus_area, difficulty),
                'resources': []
            }

            phases.append(phase)

        return phases

    def _get_phase_title(self, focus_area: str, sect: str) -> str:
        """Get appropriate title for learning phase"""
        sect_info = self.content_formatter.sect_terms.get(sect, {})
        sect_name = sect_info.get('display_name', sect.title())

        titles = {
            'basic_concepts': f'Foundation of {sect_name} Jainism',
            'fundamentals': f'Core Principles of {sect_name} Tradition',
            'principles': f'Fundamental {sect_name} Principles',
            'philosophy': f'{sect_name} Philosophical Foundations',
            'history': f'History of {sect_name} Tradition',
            'practices': f'{sect_name} Practices and Rituals',
            'sect_specifics': f'Unique Aspects of {sect_name} Tradition',
            'advanced_principles': f'Advanced {sect_name} Philosophy',
            'deep_philosophy': f'Deep Philosophical Study in {sect_name}',
            'stories': 'Sacred Stories and Parables',
            'modern_relevance': 'Jainism in Modern Context',
            'comparative_study': 'Comparative Religious Studies',
            'practical_ethics': 'Applied Jain Ethics',
            'leadership': 'Spiritual Leadership Development',
            'teaching': 'Becoming a Jain Teacher',
            'mastery': 'Mastery and Expertise',
            'research': 'Scholarly Research Methods'
        }

        return titles.get(focus_area, f'{focus_area.replace("_", " ").title()}')

    def _get_learning_objectives(self, focus_area: str, sect: str) -> List[str]:
        """Get learning objectives for specific focus area"""
        sect_info = self.content_formatter.sect_terms.get(sect, {})

        objectives = {
            'basic_concepts': [
                'Understand the basic concept of Jainism',
                'Learn about Ahimsa (non-violence)',
                'Know the importance of Tirthankaras',
                f'Recognize {sect_info.get("display_name", sect)} terminology'
            ],
            'fundamentals': [
                'Master the five main vows (Mahavratas)',
                'Understand karma theory',
                'Learn about soul and liberation',
                f'Appreciate {sect_info.get("display_name", sect)} perspectives'
            ],
            'philosophy': [
                'Analyze complex philosophical concepts',
                'Compare different schools of thought',
                'Develop critical thinking about spiritual matters',
                'Apply philosophical insights to daily life'
            ],
            'practices': [
                f'Learn {sect_info.get("display_name", sect)} practices',
                'Understand ritual significance',
                'Develop personal practice routine',
                'Connect practice with philosophy'
            ],
            'history': [
                f'Trace the history of {sect_info.get("display_name", sect)} tradition',
                'Understand key historical figures',
                'Learn about important texts and events',
                'Appreciate historical context of current practices'
            ]
        }

        return objectives.get(focus_area, [f'Master concepts in {focus_area.replace("_", " ")}'])

    def _get_phase_activities(self, focus_area: str, base_activities: List[str],
                            content_availability: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get specific activities for learning phase"""
        activities = []

        # Base activity types for different focus areas
        focus_activities = {
            'basic_concepts': ['reading', 'storytelling', 'simple_quiz', 'reflection'],
            'fundamentals': ['reading', 'analysis', 'quiz', 'practical_application'],
            'philosophy': ['deep_reading', 'critical_analysis', 'debate', 'essay_writing'],
            'practices': ['demonstration', 'practice_session', 'reflection', 'community_interaction'],
            'history': ['timeline_creation', 'biographical_study', 'historical_analysis', 'presentation']
        }

        activity_types = focus_activities.get(focus_area, base_activities)

        # Create specific activities
        for activity_type in activity_types[:4]:  # Limit to 4 activities per phase
            activity = {
                'type': activity_type,
                'title': self._get_activity_title(activity_type, focus_area),
                'description': self._get_activity_description(activity_type, focus_area),
                'duration_minutes': self._get_activity_duration(activity_type),
                'difficulty': 'medium',
                'requires_uploaded_content': self._activity_requires_content(activity_type),
                'alternative_available': True
            }

            # Adjust based on content availability
            if activity['requires_uploaded_content'] and not content_availability.get('has_content', False):
                activity['alternative_description'] = self._get_activity_alternative(activity_type)

            activities.append(activity)

        return activities

    def _get_activity_title(self, activity_type: str, focus_area: str) -> str:
        """Get title for specific activity"""
        titles = {
            'reading': f'Study {focus_area.replace("_", " ").title()}',
            'storytelling': 'Listen to Sacred Stories',
            'quiz': f'Test Your {focus_area.replace("_", " ").title()} Knowledge',
            'reflection': 'Personal Reflection and Journaling',
            'analysis': f'Analyze {focus_area.replace("_", " ").title()} Concepts',
            'debate': f'Discuss {focus_area.replace("_", " ").title()} Topics',
            'practice_session': f'Practice {focus_area.replace("_", " ").title()}',
            'presentation': f'Present on {focus_area.replace("_", " ").title()}'
        }

        return titles.get(activity_type, activity_type.replace('_', ' ').title())

    def _get_activity_description(self, activity_type: str, focus_area: str) -> str:
        """Get description for specific activity"""
        descriptions = {
            'reading': f'Read and study materials related to {focus_area.replace("_", " ")}',
            'storytelling': 'Listen to and reflect on traditional Jain stories and parables',
            'quiz': f'Test your understanding of {focus_area.replace("_", " ")} through interactive questions',
            'reflection': 'Reflect on learned concepts and write personal insights',
            'analysis': f'Critically analyze concepts and their applications in {focus_area.replace("_", " ")}',
            'debate': f'Engage in structured discussions about {focus_area.replace("_", " ")} topics',
            'practice_session': f'Engage in practical exercises related to {focus_area.replace("_", " ")}',
            'presentation': f'Create and deliver a presentation on {focus_area.replace("_", " ")} topics'
        }

        return descriptions.get(activity_type, f'Engage in {activity_type.replace("_", " ")} activities')

    def _get_activity_duration(self, activity_type: str) -> int:
        """Get typical duration for activity type"""
        durations = {
            'reading': 25,
            'storytelling': 15,
            'quiz': 10,
            'reflection': 15,
            'analysis': 30,
            'debate': 20,
            'practice_session': 20,
            'presentation': 25
        }

        return durations.get(activity_type, 20)

    def _activity_requires_content(self, activity_type: str) -> bool:
        """Check if activity requires uploaded content"""
        content_dependent = ['reading', 'analysis', 'deep_reading']
        return activity_type in content_dependent

    def _get_activity_alternative(self, activity_type: str) -> str:
        """Get alternative description when content is not available"""
        alternatives = {
            'reading': 'Use general Jain knowledge and provided materials',
            'analysis': 'Analyze fundamental concepts using basic knowledge',
            'deep_reading': 'Study from general Jain resources'
        }

        return alternatives.get(activity_type, 'Use general knowledge and available resources')

    def _get_phase_topics(self, focus_area: str, sect: str) -> List[str]:
        """Get specific topics for focus area"""
        sect_info = self.content_formatter.sect_terms.get(sect, {})

        topics = {
            'basic_concepts': ['What is Jainism?', 'Ahimsa - Non-violence', 'The Soul (Jiva)', 'Karma Basics'],
            'fundamentals': ['Five Main Vows', 'Right Knowledge, Faith, Conduct', 'Liberation (Moksha)', 'Karma Theory'],
            'philosophy': ['Nature of Soul', 'Karma Philosophy', 'Metaphysics', 'Logic and Epistemology'],
            'practices': [f'{sect_info.get("display_name", sect)} Rituals', 'Daily Practices', 'Festivals', 'Meditation'],
            'history': ['Origin of Jainism', '24 Tirthankaras', f'{sect_info.get("display_name", sect)} History', 'Sacred Texts'],
            'sect_specifics': [
                f'{sect_info.get("display_name", sect)} Unique Practices',
                f'Role of {sect_info.get("male_ascetic", "Sadhu")}s',
                f'Role of {sect_info.get("female_ascetic", "Sadhvi")}s',
                'Sect-specific Texts'
            ]
        }

        return topics.get(focus_area, [f'Topics in {focus_area.replace("_", " ")}'])

    def _get_success_criteria(self, focus_area: str, difficulty: str) -> List[str]:
        """Get success criteria for completing phase"""
        base_criteria = [
            'Complete all assigned activities',
            'Pass phase assessment with 70%+ score',
            'Demonstrate understanding in discussions',
            'Apply concepts in practical scenarios'
        ]

        if difficulty == 'hard' or difficulty == 'expert':
            base_criteria.extend([
                'Analyze complex scenarios',
                'Teach concepts to others',
                'Create original content or research'
            ])

        return base_criteria

    def _create_milestones(self, phases: List[Dict[str, Any]], template: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create learning milestones"""
        milestones = []

        for i, phase in enumerate(phases):
            milestone = {
                'milestone_number': i + 1,
                'title': f'Complete {phase["title"]}',
                'description': f'Successfully complete all activities and assessments in {phase["title"]}',
                'target_week': phase['end_week'],
                'criteria': phase['success_criteria'],
                'reward': self._get_milestone_reward(i + 1),
                'celebration': self._get_celebration_idea(phase['focus_area'])
            }
            milestones.append(milestone)

        # Add final milestone
        final_milestone = {
            'milestone_number': len(phases) + 1,
            'title': 'Complete Learning Path',
            'description': 'Successfully complete entire learning journey',
            'target_week': template['duration_weeks'],
            'criteria': ['Complete all phases', 'Pass final comprehensive assessment', 'Demonstrate mastery'],
            'reward': 'Learning Path Completion Certificate',
            'celebration': 'Graduation ceremony or recognition'
        }
        milestones.append(final_milestone)

        return milestones

    def _get_milestone_reward(self, milestone_number: int) -> str:
        """Get appropriate reward for milestone"""
        rewards = [
            'Foundation Badge',
            'Knowledge Seeker Badge',
            'Wisdom Learner Badge',
            'Understanding Master Badge',
            'Philosophy Student Badge',
            'Practice Devotee Badge'
        ]

        return rewards[min(milestone_number - 1, len(rewards) - 1)]

    def _get_celebration_idea(self, focus_area: str) -> str:
        """Get celebration idea for completing focus area"""
        celebrations = {
            'basic_concepts': 'Share what you learned with family or friends',
            'fundamentals': 'Write a reflection on your spiritual journey',
            'philosophy': 'Engage in a deep philosophical discussion',
            'practices': 'Attend a community gathering or religious event',
            'history': 'Visit a Jain temple or historical site',
            'sect_specifics': 'Connect with others from your tradition'
        }

        return celebrations.get(focus_area, 'Celebrate your achievement with gratitude')

    def _create_assessment_schedule(self, template: Dict[str, Any], phases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create assessment schedule throughout learning path"""
        assessments = []

        # Initial assessment
        assessments.append({
            'assessment_type': 'initial',
            'title': 'Initial Knowledge Assessment',
            'week': 1,
            'purpose': 'Baseline knowledge evaluation',
            'format': 'adaptive_quiz',
            'duration_minutes': 20
        })

        # Phase assessments
        for phase in phases:
            assessments.append({
                'assessment_type': 'phase',
                'title': f'{phase["title"]} Assessment',
                'week': phase['end_week'],
                'phase': phase['phase_number'],
                'purpose': f'Evaluate understanding of {phase["focus_area"]}',
                'format': 'comprehensive_quiz',
                'duration_minutes': 30
            })

        # Final assessment
        assessments.append({
            'assessment_type': 'final',
            'title': 'Comprehensive Final Assessment',
            'week': template['duration_weeks'],
            'purpose': 'Overall mastery evaluation',
            'format': 'comprehensive_exam',
            'duration_minutes': 45
        })

        return assessments

    def _identify_required_resources(self, template: Dict[str, Any],
                                   content_availability: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify resources needed for learning path"""
        resources = [
            {
                'type': 'uploaded_content',
                'title': 'Your Uploaded PDFs',
                'description': 'Sect-specific content you have uploaded',
                'required': False,
                'available': content_availability.get('has_content', False)
            },
            {
                'type': 'assessment_system',
                'title': 'Assessment Platform',
                'description': 'Interactive quizzes and evaluations',
                'required': True,
                'available': True
            },
            {
                'type': 'discussion_forum',
                'title': 'Discussion Platform',
                'description': 'Community discussions and Q&A',
                'required': False,
                'available': True
            }
        ]

        # Add activity-specific resources
        for activity_type in template.get('activities', []):
            if activity_type not in ['reading', 'quiz', 'discussion']:
                resources.append({
                    'type': 'activity_support',
                    'title': f'{activity_type.replace("_", " ").title()} Support',
                    'description': f'Tools and materials for {activity_type} activities',
                    'required': False,
                    'available': True
                })

        return resources

    def _define_adaptive_elements(self, template: Dict[str, Any], user_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Define adaptive elements of learning path"""
        return [
            {
                'element': 'difficulty_adjustment',
                'description': 'Automatically adjust difficulty based on performance',
                'trigger': 'assessment_results',
                'action': 'modify_content_difficulty'
            },
            {
                'element': 'pace_adjustment',
                'description': 'Adjust learning pace based on progress',
                'trigger': 'completion_time',
                'action': 'modify_session_frequency'
            },
            {
                'element': 'content_personalization',
                'description': 'Personalize content based on interests',
                'trigger': 'user_engagement',
                'action': 'recommend_additional_topics'
            },
            {
                'element': 'learning_style_adaptation',
                'description': 'Adapt activities to learning style preferences',
                'trigger': 'activity_performance',
                'action': 'adjust_activity_mix'
            }
        ]

    def _create_fallback_path(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Create simple fallback path when generation fails"""
        return {
            'id': f"fallback_path_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'user_id': user_profile.get('user_id', 'anonymous'),
            'sect': user_profile.get('sect', 'digambara'),
            'language': user_profile.get('language', 'en'),
            'title': 'Basic Jain Learning Path',
            'description': 'A simple learning path to get you started',
            'estimated_duration_weeks': 8,
            'sessions_per_week': 2,
            'session_duration_minutes': 30,
            'phases': [
                {
                    'phase_number': 1,
                    'title': 'Introduction to Jainism',
                    'duration_weeks': 4,
                    'topics': ['Basic Principles', 'Ahimsa', 'Tirthankaras'],
                    'activities': ['reading', 'quiz', 'reflection']
                },
                {
                    'phase_number': 2,
                    'title': 'Jain Practices',
                    'duration_weeks': 4,
                    'topics': ['Five Vows', 'Meditation', 'Festivals'],
                    'activities': ['study', 'practice', 'assessment']
                }
            ],
            'fallback': True,
            'error_recovery': True
        }

    def _save_learning_path(self, path: Dict[str, Any]) -> bool:
        """Save learning path to file"""
        try:
            path_id = path['id']
            path_file = self.paths_dir / f"{path_id}.json"

            with open(path_file, 'w', encoding='utf-8') as f:
                json.dump(path, f, indent=2, ensure_ascii=False)

            return True

        except Exception as e:
            logger.error(f"Error saving learning path: {e}")
            return False

    def get_learning_path(self, path_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve learning path by ID"""
        try:
            path_file = self.paths_dir / f"{path_id}.json"

            if not path_file.exists():
                return None

            with open(path_file, 'r', encoding='utf-8') as f:
                path = json.load(f)

            return path

        except Exception as e:
            logger.error(f"Error retrieving learning path: {e}")
            return None

    def update_path_progress(self, path_id: str, progress_data: Dict[str, Any]) -> bool:
        """Update learning path progress"""
        try:
            path = self.get_learning_path(path_id)
            if not path:
                return False

            # Initialize progress tracking if not exists
            if 'progress' not in path:
                path['progress'] = {
                    'current_phase': 1,
                    'current_week': 1,
                    'completed_activities': [],
                    'phase_completions': [],
                    'overall_completion': 0.0,
                    'last_updated': datetime.now().isoformat()
                }

            # Update progress
            progress = path['progress']
            progress.update(progress_data)
            progress['last_updated'] = datetime.now().isoformat()

            # Calculate overall completion
            total_phases = len(path.get('phases', []))
            completed_phases = len(progress.get('phase_completions', []))
            progress['overall_completion'] = (completed_phases / total_phases * 100) if total_phases > 0 else 0

            # Save updated path
            return self._save_learning_path(path)

        except Exception as e:
            logger.error(f"Error updating path progress: {e}")
            return False