"""
User Profiler for Jain Learning System
Handles user profile creation, sect detection, and personalization
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import yaml
import re

logger = logging.getLogger(__name__)

class UserProfiler:
    """Manages user profiles with sect detection and personalization"""

    def __init__(self, config_path: str = None):
        """Initialize user profiler with configuration"""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "config.yaml"

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.profiles_dir = Path(__file__).parent.parent / "data" / "user_profiles"
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

        # Extract configuration data
        self.sects = {sect['code']: sect for sect in self.config['sects']['supported']}
        self.age_groups = {age['code']: age for age in self.config['user_profiles']['age_groups']}
        self.knowledge_levels = {level['code']: level for level in self.config['user_profiles']['knowledge_levels']}
        self.languages = {lang['code']: lang for lang in self.config['languages']['supported']}

    def create_user_profile(self, user_id: str, initial_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a new user profile"""
        try:
            profile = {
                'user_id': user_id,
                'created_timestamp': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat(),

                # Basic Information
                'sect': initial_data.get('sect') if initial_data else None,
                'language': initial_data.get('language', 'en') if initial_data else 'en',
                'age_group': initial_data.get('age_group') if initial_data else None,
                'knowledge_level': initial_data.get('knowledge_level', 'beginner') if initial_data else 'beginner',

                # Learning Preferences
                'learning_style': initial_data.get('learning_style', 'mixed') if initial_data else 'mixed',
                'difficulty_preference': initial_data.get('difficulty_preference', 'medium') if initial_data else 'medium',
                'content_types': initial_data.get('content_types', ['text', 'interactive']) if initial_data else ['text', 'interactive'],

                # Progress Tracking
                'assessments_taken': [],
                'learning_sessions': [],
                'knowledge_scores': {
                    'principles': 0,
                    'history': 0,
                    'practices': 0,
                    'philosophy': 0
                },
                'total_study_time': 0,
                'streak_days': 0,
                'last_activity': None,

                # Personalization Data
                'interests': initial_data.get('interests', []) if initial_data else [],
                'goals': initial_data.get('goals', []) if initial_data else [],
                'preferred_topics': [],
                'completed_topics': [],

                # Sect Detection Data
                'sect_indicators': {
                    'terminology_usage': {},
                    'practice_preferences': {},
                    'text_references': {},
                    'confidence_scores': {}
                },

                # Adaptive Learning
                'learning_patterns': {
                    'optimal_session_length': 30,
                    'best_time_of_day': None,
                    'preferred_question_types': [],
                    'challenge_level': 'appropriate'
                },

                # Settings
                'settings': {
                    'notifications': True,
                    'reminders': True,
                    'progress_sharing': False,
                    'adaptive_difficulty': True
                }
            }

            # Save profile
            self._save_profile(profile)

            return profile

        except Exception as e:
            logger.error(f"Error creating user profile: {e}")
            return {}

    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve user profile by ID"""
        try:
            profile_file = self.profiles_dir / f"{user_id}.json"

            if not profile_file.exists():
                return None

            with open(profile_file, 'r', encoding='utf-8') as f:
                profile = json.load(f)

            return profile

        except Exception as e:
            logger.error(f"Error retrieving user profile: {e}")
            return None

    def update_user_profile(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """Update user profile with new data"""
        try:
            profile = self.get_user_profile(user_id)
            if not profile:
                return False

            # Update fields
            for key, value in updates.items():
                if key in profile:
                    profile[key] = value

            profile['last_updated'] = datetime.now().isoformat()

            # Save updated profile
            self._save_profile(profile)

            return True

        except Exception as e:
            logger.error(f"Error updating user profile: {e}")
            return False

    def detect_sect_from_responses(self, user_id: str, responses: List[str],
                                  question_contexts: List[str] = None) -> Dict[str, Any]:
        """
        Detect user's sect based on their responses and terminology usage

        Args:
            user_id: User identifier
            responses: List of user responses/answers
            question_contexts: Contexts where responses were given

        Returns:
            Sect detection results with confidence scores
        """
        try:
            profile = self.get_user_profile(user_id) or self.create_user_profile(user_id)

            sect_scores = {sect_code: 0 for sect_code in self.sects.keys()}
            terminology_usage = {}

            # Analyze terminology in responses
            for i, response in enumerate(responses):
                response_lower = response.lower()

                # Check for sect-specific terminology
                for sect_code, sect_info in self.sects.items():
                    score = 0

                    # Check for male ascetic terms
                    if sect_info['male_ascetic'].lower() in response_lower:
                        score += 3
                        terminology_usage[sect_info['male_ascetic']] = terminology_usage.get(sect_info['male_ascetic'], 0) + 1

                    # Check for female ascetic terms
                    if sect_info['female_ascetic'].lower() in response_lower:
                        score += 3
                        terminology_usage[sect_info['female_ascetic']] = terminology_usage.get(sect_info['female_ascetic'], 0) + 1

                    # Check for general sect preferences
                    if sect_code in response_lower or sect_info['name'].lower() in response_lower:
                        score += 5

                    sect_scores[sect_code] += score

            # Additional contextual analysis
            sect_scores = self._analyze_contextual_preferences(responses, sect_scores)

            # Normalize scores
            total_score = sum(sect_scores.values())
            confidence_scores = {}

            if total_score > 0:
                for sect, score in sect_scores.items():
                    confidence_scores[sect] = round((score / total_score) * 100, 2)
            else:
                # Equal probability if no indicators found
                confidence_scores = {sect: 25.0 for sect in self.sects.keys()}

            # Determine most likely sect
            predicted_sect = max(confidence_scores, key=confidence_scores.get)
            max_confidence = confidence_scores[predicted_sect]

            # Update profile with sect detection data
            detection_data = {
                'sect_indicators': {
                    'terminology_usage': terminology_usage,
                    'confidence_scores': confidence_scores,
                    'last_detection': datetime.now().isoformat()
                }
            }

            # Auto-assign sect if confidence is high enough
            if max_confidence > 60 and not profile.get('sect'):
                detection_data['sect'] = predicted_sect

            self.update_user_profile(user_id, detection_data)

            return {
                'predicted_sect': predicted_sect,
                'confidence': max_confidence,
                'confidence_scores': confidence_scores,
                'terminology_found': terminology_usage,
                'auto_assigned': max_confidence > 60 and not profile.get('sect')
            }

        except Exception as e:
            logger.error(f"Error detecting sect: {e}")
            return {
                'predicted_sect': 'digambara',  # Default
                'confidence': 0,
                'error': str(e)
            }

    def _analyze_contextual_preferences(self, responses: List[str],
                                      sect_scores: Dict[str, int]) -> Dict[str, int]:
        """Analyze contextual preferences that might indicate sect affiliation"""

        # Patterns that might indicate specific sects
        sect_patterns = {
            'digambara': [
                r'\b(naked|sky.?clad|cloth.?less)\b',
                r'\b(aryika|muni)\b',
                r'\b(kashaya|passions?)\b'
            ],
            'shwetambara': [
                r'\b(white.?clad|cloth)\b',
                r'\b(sadhu|sadhvi)\b',
                r'\b(agam|canon)\b',
                r'\b(temple|derasar)\b'
            ],
            'terapanthi': [
                r'\b(reform|modern|terapanth)\b',
                r'\b(acharya|tulsi)\b',
                r'\b(discipline|strict)\b'
            ],
            'sthanakvasi': [
                r'\b(non.?idol|without.?image)\b',
                r'\b(upashray)\b',
                r'\b(simple|plain)\b'
            ]
        }

        combined_responses = ' '.join(responses).lower()

        for sect, patterns in sect_patterns.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, combined_responses))
                sect_scores[sect] += matches * 2

        return sect_scores

    def determine_knowledge_level(self, user_id: str, assessment_results: List[Dict[str, Any]]) -> str:
        """Determine user's knowledge level based on assessment performance"""
        try:
            if not assessment_results:
                return 'beginner'

            # Calculate average performance
            total_percentage = sum(result.get('percentage', 0) for result in assessment_results)
            avg_percentage = total_percentage / len(assessment_results)

            # Determine knowledge level based on average
            for level_code, level_info in self.knowledge_levels.items():
                score_range = level_info['score_range']
                if score_range[0] <= avg_percentage <= score_range[1]:
                    # Update user profile
                    self.update_user_profile(user_id, {'knowledge_level': level_code})
                    return level_code

            return 'beginner'

        except Exception as e:
            logger.error(f"Error determining knowledge level: {e}")
            return 'beginner'

    def personalize_content_recommendations(self, user_id: str) -> Dict[str, Any]:
        """Generate personalized content recommendations for user"""
        try:
            profile = self.get_user_profile(user_id)
            if not profile:
                return {}

            recommendations = {
                'topics': [],
                'difficulty_level': profile.get('difficulty_preference', 'medium'),
                'content_types': profile.get('content_types', ['text']),
                'study_plan': [],
                'focus_areas': []
            }

            # Recommend topics based on interests and knowledge gaps
            knowledge_scores = profile.get('knowledge_scores', {})
            completed_topics = profile.get('completed_topics', [])

            # Find weak areas for improvement
            weak_areas = [
                category for category, score in knowledge_scores.items()
                if score < 60
            ]

            if weak_areas:
                recommendations['focus_areas'] = weak_areas[:3]  # Top 3 weak areas

            # Recommend next topics based on learning path
            sect = profile.get('sect')
            if sect:
                recommendations['topics'] = self._get_sect_specific_topics(sect, profile)

            # Generate study plan
            recommendations['study_plan'] = self._generate_study_plan(profile)

            return recommendations

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return {}

    def _get_sect_specific_topics(self, sect: str, profile: Dict[str, Any]) -> List[str]:
        """Get topics specific to user's sect"""
        base_topics = [
            'Basic Principles of Jainism',
            'Ahimsa and Non-violence',
            'The 24 Tirthankaras',
            'Karma and Liberation',
            'Five Main Vows'
        ]

        sect_specific_topics = {
            'digambara': [
                'Digambara Traditions',
                'Muni and Aryika Practices',
                'Sky-clad Philosophy',
                'Kashaya and Passions'
            ],
            'shwetambara': [
                'Shwetambara Scriptures',
                'Temple Worship',
                'Sadhu and Sadhvi Traditions',
                'Agam Literature'
            ],
            'terapanthi': [
                'Terapanthi Reforms',
                'Acharya Leadership',
                'Modern Jain Practices',
                'Disciplined Living'
            ],
            'sthanakvasi': [
                'Non-idol Worship',
                'Upashray Practices',
                'Simple Living',
                'Pure Meditation'
            ]
        }

        topics = base_topics + sect_specific_topics.get(sect, [])

        # Filter out completed topics
        completed = profile.get('completed_topics', [])
        return [topic for topic in topics if topic not in completed]

    def _generate_study_plan(self, profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate personalized study plan"""
        knowledge_level = profile.get('knowledge_level', 'beginner')
        learning_patterns = profile.get('learning_patterns', {})

        session_length = learning_patterns.get('optimal_session_length', 30)

        if knowledge_level == 'beginner':
            plan = [
                {
                    'week': 1,
                    'focus': 'Basic Principles',
                    'sessions_per_week': 3,
                    'session_length': session_length,
                    'activities': ['reading', 'quiz', 'reflection']
                },
                {
                    'week': 2,
                    'focus': 'Five Main Vows',
                    'sessions_per_week': 3,
                    'session_length': session_length,
                    'activities': ['reading', 'practical_examples', 'assessment']
                }
            ]
        elif knowledge_level == 'intermediate':
            plan = [
                {
                    'week': 1,
                    'focus': 'Advanced Philosophy',
                    'sessions_per_week': 4,
                    'session_length': session_length,
                    'activities': ['deep_reading', 'analysis', 'discussion']
                },
                {
                    'week': 2,
                    'focus': 'Sect-specific Practices',
                    'sessions_per_week': 4,
                    'session_length': session_length,
                    'activities': ['comparative_study', 'practical_application']
                }
            ]
        else:  # advanced
            plan = [
                {
                    'week': 1,
                    'focus': 'Scholarly Texts',
                    'sessions_per_week': 5,
                    'session_length': session_length,
                    'activities': ['research', 'analysis', 'teaching_others']
                }
            ]

        return plan

    def update_learning_progress(self, user_id: str, session_data: Dict[str, Any]) -> bool:
        """Update user's learning progress with session data"""
        try:
            profile = self.get_user_profile(user_id)
            if not profile:
                return False

            # Update learning sessions
            sessions = profile.get('learning_sessions', [])
            sessions.append({
                'timestamp': datetime.now().isoformat(),
                'duration': session_data.get('duration', 0),
                'topics_covered': session_data.get('topics', []),
                'performance': session_data.get('performance', {}),
                'satisfaction': session_data.get('satisfaction', None)
            })

            # Update total study time
            total_time = profile.get('total_study_time', 0)
            total_time += session_data.get('duration', 0)

            # Update streak
            last_activity = profile.get('last_activity')
            streak = profile.get('streak_days', 0)

            if last_activity:
                last_date = datetime.fromisoformat(last_activity).date()
                today = datetime.now().date()
                days_diff = (today - last_date).days

                if days_diff == 1:
                    streak += 1
                elif days_diff > 1:
                    streak = 1
            else:
                streak = 1

            # Update completed topics
            completed_topics = set(profile.get('completed_topics', []))
            new_topics = session_data.get('completed_topics', [])
            completed_topics.update(new_topics)

            # Update profile
            updates = {
                'learning_sessions': sessions[-50:],  # Keep last 50 sessions
                'total_study_time': total_time,
                'streak_days': streak,
                'last_activity': datetime.now().isoformat(),
                'completed_topics': list(completed_topics)
            }

            return self.update_user_profile(user_id, updates)

        except Exception as e:
            logger.error(f"Error updating learning progress: {e}")
            return False

    def _save_profile(self, profile: Dict[str, Any]) -> bool:
        """Save user profile to file"""
        try:
            user_id = profile['user_id']
            profile_file = self.profiles_dir / f"{user_id}.json"

            with open(profile_file, 'w', encoding='utf-8') as f:
                json.dump(profile, f, indent=2, ensure_ascii=False)

            return True

        except Exception as e:
            logger.error(f"Error saving profile: {e}")
            return False

    def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """Get user statistics and analytics"""
        try:
            profile = self.get_user_profile(user_id)
            if not profile:
                return {}

            sessions = profile.get('learning_sessions', [])
            assessments = profile.get('assessments_taken', [])

            stats = {
                'total_sessions': len(sessions),
                'total_study_time': profile.get('total_study_time', 0),
                'streak_days': profile.get('streak_days', 0),
                'assessments_completed': len(assessments),
                'knowledge_level': profile.get('knowledge_level', 'beginner'),
                'avg_session_duration': 0,
                'topics_mastered': len(profile.get('completed_topics', [])),
                'sect': profile.get('sect'),
                'join_date': profile.get('created_timestamp', ''),
                'last_activity': profile.get('last_activity', '')
            }

            # Calculate average session duration
            if sessions:
                total_duration = sum(s.get('duration', 0) for s in sessions)
                stats['avg_session_duration'] = round(total_duration / len(sessions), 1)

            # Calculate knowledge scores average
            knowledge_scores = profile.get('knowledge_scores', {})
            if knowledge_scores:
                avg_knowledge = sum(knowledge_scores.values()) / len(knowledge_scores)
                stats['avg_knowledge_score'] = round(avg_knowledge, 1)

            return stats

        except Exception as e:
            logger.error(f"Error getting user statistics: {e}")
            return {}