"""
Progress Tracker for Jain Learning System
Tracks user progress, achievements, and learning analytics
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import statistics

logger = logging.getLogger(__name__)

class ProgressTracker:
    """Tracks and analyzes user learning progress"""

    def __init__(self):
        """Initialize progress tracker"""
        self.data_dir = Path(__file__).parent.parent / "data"
        self.progress_dir = self.data_dir / "progress"
        self.progress_dir.mkdir(parents=True, exist_ok=True)

    def initialize_user_progress(self, user_id: str, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize progress tracking for a new user"""
        try:
            progress_data = {
                'user_id': user_id,
                'created_timestamp': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat(),

                # Learning Progress
                'sessions_completed': 0,
                'total_study_time': 0,  # in minutes
                'current_streak': 0,
                'longest_streak': 0,
                'last_activity_date': None,

                # Knowledge Progress
                'knowledge_scores': {
                    'principles': {'current': 0, 'best': 0, 'attempts': 0},
                    'history': {'current': 0, 'best': 0, 'attempts': 0},
                    'practices': {'current': 0, 'best': 0, 'attempts': 0},
                    'philosophy': {'current': 0, 'best': 0, 'attempts': 0}
                },
                'overall_knowledge_level': user_profile.get('knowledge_level', 'beginner'),

                # Assessment Progress
                'assessments_taken': [],
                'assessment_stats': {
                    'total_assessments': 0,
                    'average_score': 0,
                    'best_score': 0,
                    'improvement_trend': 0
                },

                # Learning Path Progress
                'current_learning_path': None,
                'completed_learning_paths': [],
                'path_progress': {},

                # Topic Mastery
                'topics_started': [],
                'topics_completed': [],
                'topics_mastered': [],
                'mastery_levels': {},

                # Achievements
                'achievements': [],
                'badges_earned': [],
                'milestones_reached': [],

                # Learning Analytics
                'learning_patterns': {
                    'preferred_study_times': [],
                    'optimal_session_duration': None,
                    'most_effective_activities': [],
                    'difficulty_preferences': [],
                    'engagement_patterns': []
                },

                # Goals and Targets
                'goals': {
                    'weekly_study_target': 180,  # 3 hours
                    'monthly_assessment_target': 4,
                    'streak_target': 7,
                    'mastery_target': 10  # topics
                },
                'goal_progress': {
                    'weekly_study_current': 0,
                    'monthly_assessments_current': 0,
                    'current_streak': 0,
                    'topics_mastered_count': 0
                }
            }

            self._save_progress_data(progress_data)
            return progress_data

        except Exception as e:
            logger.error(f"Error initializing user progress: {e}")
            return {}

    def record_study_session(self, user_id: str, session_data: Dict[str, Any]) -> bool:
        """Record a completed study session"""
        try:
            progress = self.get_user_progress(user_id)
            if not progress:
                return False

            # Extract session information
            duration = session_data.get('duration_minutes', 0)
            topics_covered = session_data.get('topics_covered', [])
            activities_completed = session_data.get('activities_completed', [])
            satisfaction_rating = session_data.get('satisfaction_rating', None)
            session_timestamp = session_data.get('timestamp', datetime.now().isoformat())

            # Update basic counters
            progress['sessions_completed'] += 1
            progress['total_study_time'] += duration

            # Update streak
            self._update_study_streak(progress, session_timestamp)

            # Track topics
            for topic in topics_covered:
                if topic not in progress['topics_started']:
                    progress['topics_started'].append(topic)

            # Update learning patterns
            self._update_learning_patterns(progress, session_data)

            # Update goal progress
            self._update_goal_progress(progress, {'study_time': duration})

            # Record detailed session
            session_record = {
                'timestamp': session_timestamp,
                'duration_minutes': duration,
                'topics_covered': topics_covered,
                'activities_completed': activities_completed,
                'satisfaction_rating': satisfaction_rating
            }

            # Keep last 100 sessions
            if 'detailed_sessions' not in progress:
                progress['detailed_sessions'] = []
            progress['detailed_sessions'].append(session_record)
            progress['detailed_sessions'] = progress['detailed_sessions'][-100:]

            progress['last_updated'] = datetime.now().isoformat()
            progress['last_activity_date'] = session_timestamp

            self._save_progress_data(progress)
            return True

        except Exception as e:
            logger.error(f"Error recording study session: {e}")
            return False

    def record_assessment_result(self, user_id: str, assessment_data: Dict[str, Any]) -> bool:
        """Record assessment completion and results"""
        try:
            progress = self.get_user_progress(user_id)
            if not progress:
                return False

            # Extract assessment information
            assessment_id = assessment_data.get('assessment_id', '')
            score = assessment_data.get('percentage', 0)
            category_scores = assessment_data.get('category_scores', {})
            knowledge_level = assessment_data.get('knowledge_level', 'beginner')
            timestamp = assessment_data.get('completion_timestamp', datetime.now().isoformat())

            # Record assessment
            assessment_record = {
                'assessment_id': assessment_id,
                'timestamp': timestamp,
                'score': score,
                'category_scores': category_scores,
                'knowledge_level': knowledge_level
            }

            progress['assessments_taken'].append(assessment_record)

            # Update assessment statistics
            stats = progress['assessment_stats']
            stats['total_assessments'] += 1

            # Calculate new average
            scores = [a['score'] for a in progress['assessments_taken']]
            stats['average_score'] = round(statistics.mean(scores), 1)
            stats['best_score'] = max(scores)

            # Calculate improvement trend (last 5 vs previous 5)
            if len(scores) >= 10:
                recent_avg = statistics.mean(scores[-5:])
                previous_avg = statistics.mean(scores[-10:-5])
                stats['improvement_trend'] = round(recent_avg - previous_avg, 1)
            elif len(scores) >= 2:
                stats['improvement_trend'] = round(scores[-1] - scores[0], 1)

            # Update knowledge scores by category
            for category, category_data in category_scores.items():
                if category in progress['knowledge_scores']:
                    category_progress = progress['knowledge_scores'][category]
                    category_score = (category_data['earned'] / category_data['total'] * 100) if category_data['total'] > 0 else 0

                    category_progress['current'] = round(category_score, 1)
                    category_progress['best'] = max(category_progress['best'], category_score)
                    category_progress['attempts'] += 1

            # Update overall knowledge level
            progress['overall_knowledge_level'] = knowledge_level

            # Update goal progress
            self._update_goal_progress(progress, {'assessment_completed': True})

            # Check for achievements
            self._check_assessment_achievements(progress, assessment_record)

            progress['last_updated'] = datetime.now().isoformat()
            progress['last_activity_date'] = timestamp

            self._save_progress_data(progress)
            return True

        except Exception as e:
            logger.error(f"Error recording assessment result: {e}")
            return False

    def update_topic_mastery(self, user_id: str, topic: str, mastery_level: str) -> bool:
        """Update mastery level for a specific topic"""
        try:
            progress = self.get_user_progress(user_id)
            if not progress:
                return False

            # Update mastery levels
            progress['mastery_levels'][topic] = {
                'level': mastery_level,
                'timestamp': datetime.now().isoformat()
            }

            # Update topic lists
            if topic not in progress['topics_started']:
                progress['topics_started'].append(topic)

            if mastery_level == 'completed' and topic not in progress['topics_completed']:
                progress['topics_completed'].append(topic)

            if mastery_level == 'mastered' and topic not in progress['topics_mastered']:
                progress['topics_mastered'].append(topic)

            # Update goal progress
            mastered_count = len(progress['topics_mastered'])
            progress['goal_progress']['topics_mastered_count'] = mastered_count

            # Check for achievements
            self._check_mastery_achievements(progress, topic, mastery_level)

            progress['last_updated'] = datetime.now().isoformat()
            self._save_progress_data(progress)
            return True

        except Exception as e:
            logger.error(f"Error updating topic mastery: {e}")
            return False

    def update_learning_path_progress(self, user_id: str, path_id: str,
                                    progress_data: Dict[str, Any]) -> bool:
        """Update progress on learning path"""
        try:
            progress = self.get_user_progress(user_id)
            if not progress:
                return False

            # Initialize path progress if needed
            if path_id not in progress['path_progress']:
                progress['path_progress'][path_id] = {
                    'started_timestamp': datetime.now().isoformat(),
                    'current_phase': 1,
                    'completed_phases': [],
                    'overall_completion': 0,
                    'activities_completed': [],
                    'assessments_passed': []
                }

            # Update path progress
            path_progress = progress['path_progress'][path_id]
            path_progress.update(progress_data)
            path_progress['last_updated'] = datetime.now().isoformat()

            # Set current learning path
            progress['current_learning_path'] = path_id

            # Check if path is completed
            completion_percentage = progress_data.get('overall_completion', 0)
            if completion_percentage >= 100 and path_id not in progress['completed_learning_paths']:
                progress['completed_learning_paths'].append({
                    'path_id': path_id,
                    'completion_timestamp': datetime.now().isoformat(),
                    'final_completion': completion_percentage
                })

                # Check for achievements
                self._check_path_completion_achievements(progress, path_id)

            progress['last_updated'] = datetime.now().isoformat()
            self._save_progress_data(progress)
            return True

        except Exception as e:
            logger.error(f"Error updating learning path progress: {e}")
            return False

    def award_achievement(self, user_id: str, achievement: Dict[str, Any]) -> bool:
        """Award an achievement to user"""
        try:
            progress = self.get_user_progress(user_id)
            if not progress:
                return False

            achievement_record = {
                'id': achievement['id'],
                'title': achievement['title'],
                'description': achievement['description'],
                'category': achievement.get('category', 'general'),
                'points': achievement.get('points', 0),
                'timestamp': datetime.now().isoformat()
            }

            # Check if already awarded
            existing_achievements = [a['id'] for a in progress['achievements']]
            if achievement['id'] not in existing_achievements:
                progress['achievements'].append(achievement_record)

                # Add badge if specified
                if 'badge' in achievement:
                    progress['badges_earned'].append({
                        'badge_id': achievement['badge'],
                        'title': achievement['title'],
                        'timestamp': datetime.now().isoformat()
                    })

                progress['last_updated'] = datetime.now().isoformat()
                self._save_progress_data(progress)

            return True

        except Exception as e:
            logger.error(f"Error awarding achievement: {e}")
            return False

    def get_user_progress(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve user progress data"""
        try:
            progress_file = self.progress_dir / f"{user_id}_progress.json"

            if not progress_file.exists():
                return None

            with open(progress_file, 'r', encoding='utf-8') as f:
                progress = json.load(f)

            return progress

        except Exception as e:
            logger.error(f"Error retrieving user progress: {e}")
            return None

    def get_progress_analytics(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Get detailed analytics for user progress"""
        try:
            progress = self.get_user_progress(user_id)
            if not progress:
                return {}

            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            # Get recent sessions
            recent_sessions = []
            for session in progress.get('detailed_sessions', []):
                session_date = datetime.fromisoformat(session['timestamp'])
                if start_date <= session_date <= end_date:
                    recent_sessions.append(session)

            # Get recent assessments
            recent_assessments = []
            for assessment in progress.get('assessments_taken', []):
                assessment_date = datetime.fromisoformat(assessment['timestamp'])
                if start_date <= assessment_date <= end_date:
                    recent_assessments.append(assessment)

            # Calculate analytics
            analytics = {
                'period_days': days,
                'study_analytics': self._calculate_study_analytics(recent_sessions),
                'assessment_analytics': self._calculate_assessment_analytics(recent_assessments),
                'progress_summary': self._calculate_progress_summary(progress),
                'learning_patterns': self._analyze_learning_patterns(recent_sessions),
                'recommendations': self._generate_progress_recommendations(progress, recent_sessions)
            }

            return analytics

        except Exception as e:
            logger.error(f"Error generating progress analytics: {e}")
            return {}

    def _update_study_streak(self, progress: Dict[str, Any], session_timestamp: str):
        """Update study streak based on session date"""
        try:
            session_date = datetime.fromisoformat(session_timestamp).date()
            last_activity = progress.get('last_activity_date')

            if last_activity:
                last_date = datetime.fromisoformat(last_activity).date()
                days_diff = (session_date - last_date).days

                if days_diff == 1:
                    # Consecutive day
                    progress['current_streak'] += 1
                elif days_diff == 0:
                    # Same day, no change to streak
                    pass
                else:
                    # Gap in streak
                    progress['current_streak'] = 1

            else:
                # First session
                progress['current_streak'] = 1

            # Update longest streak
            progress['longest_streak'] = max(
                progress['longest_streak'],
                progress['current_streak']
            )

        except Exception as e:
            logger.error(f"Error updating study streak: {e}")

    def _update_learning_patterns(self, progress: Dict[str, Any], session_data: Dict[str, Any]):
        """Update learning patterns based on session data"""
        try:
            patterns = progress['learning_patterns']

            # Track study times
            session_time = datetime.fromisoformat(session_data.get('timestamp', datetime.now().isoformat()))
            hour = session_time.hour
            patterns['preferred_study_times'].append(hour)

            # Keep last 30 study times
            patterns['preferred_study_times'] = patterns['preferred_study_times'][-30:]

            # Update optimal session duration
            duration = session_data.get('duration_minutes', 0)
            satisfaction = session_data.get('satisfaction_rating', None)

            if duration > 0 and satisfaction is not None:
                if not patterns['optimal_session_duration']:
                    patterns['optimal_session_duration'] = duration
                else:
                    # Simple adjustment based on satisfaction
                    if satisfaction >= 4:  # Assuming 1-5 scale
                        patterns['optimal_session_duration'] = int(
                            patterns['optimal_session_duration'] * 0.9 + duration * 0.1
                        )

            # Track effective activities
            activities = session_data.get('activities_completed', [])
            if satisfaction and satisfaction >= 4:
                patterns['most_effective_activities'].extend(activities)
                patterns['most_effective_activities'] = patterns['most_effective_activities'][-20:]

        except Exception as e:
            logger.error(f"Error updating learning patterns: {e}")

    def _update_goal_progress(self, progress: Dict[str, Any], update_data: Dict[str, Any]):
        """Update progress toward goals"""
        try:
            goal_progress = progress['goal_progress']
            current_date = datetime.now()

            # Reset weekly/monthly counters if needed
            last_update = progress.get('last_updated')
            if last_update:
                last_date = datetime.fromisoformat(last_update)

                # Reset weekly counter
                if (current_date - last_date).days >= 7:
                    goal_progress['weekly_study_current'] = 0

                # Reset monthly counter
                if current_date.month != last_date.month:
                    goal_progress['monthly_assessments_current'] = 0

            # Update study time goal
            if 'study_time' in update_data:
                goal_progress['weekly_study_current'] += update_data['study_time']

            # Update assessment goal
            if 'assessment_completed' in update_data:
                goal_progress['monthly_assessments_current'] += 1

            # Update streak goal
            goal_progress['current_streak'] = progress['current_streak']

        except Exception as e:
            logger.error(f"Error updating goal progress: {e}")

    def _check_assessment_achievements(self, progress: Dict[str, Any], assessment: Dict[str, Any]):
        """Check for assessment-related achievements"""
        achievements_to_award = []

        # First assessment
        if progress['assessment_stats']['total_assessments'] == 1:
            achievements_to_award.append({
                'id': 'first_assessment',
                'title': 'First Steps',
                'description': 'Completed your first assessment',
                'category': 'assessment',
                'points': 10
            })

        # High score achievements
        score = assessment['score']
        if score >= 90:
            achievements_to_award.append({
                'id': 'high_achiever',
                'title': 'High Achiever',
                'description': 'Scored 90% or higher on an assessment',
                'category': 'assessment',
                'points': 25
            })

        # Perfect score
        if score == 100:
            achievements_to_award.append({
                'id': 'perfect_score',
                'title': 'Perfect Score',
                'description': 'Achieved a perfect 100% score',
                'category': 'assessment',
                'points': 50
            })

        # Assessment milestones
        total_assessments = progress['assessment_stats']['total_assessments']
        if total_assessments in [5, 10, 25, 50]:
            achievements_to_award.append({
                'id': f'assessment_milestone_{total_assessments}',
                'title': f'{total_assessments} Assessments',
                'description': f'Completed {total_assessments} assessments',
                'category': 'milestone',
                'points': total_assessments * 2
            })

        # Award achievements
        for achievement in achievements_to_award:
            self.award_achievement(progress['user_id'], achievement)

    def _check_mastery_achievements(self, progress: Dict[str, Any], topic: str, mastery_level: str):
        """Check for topic mastery achievements"""
        if mastery_level == 'mastered':
            mastered_count = len(progress['topics_mastered'])

            # First mastery
            if mastered_count == 1:
                achievement = {
                    'id': 'first_mastery',
                    'title': 'First Mastery',
                    'description': 'Mastered your first topic',
                    'category': 'mastery',
                    'points': 20
                }
                self.award_achievement(progress['user_id'], achievement)

            # Mastery milestones
            if mastered_count in [5, 10, 20, 50]:
                achievement = {
                    'id': f'mastery_milestone_{mastered_count}',
                    'title': f'{mastered_count} Topics Mastered',
                    'description': f'Achieved mastery in {mastered_count} topics',
                    'category': 'mastery',
                    'points': mastered_count * 5
                }
                self.award_achievement(progress['user_id'], achievement)

    def _check_path_completion_achievements(self, progress: Dict[str, Any], path_id: str):
        """Check for learning path completion achievements"""
        completed_paths = len(progress['completed_learning_paths'])

        # First path completion
        if completed_paths == 1:
            achievement = {
                'id': 'first_path_completion',
                'title': 'Path Walker',
                'description': 'Completed your first learning path',
                'category': 'path',
                'points': 100
            }
            self.award_achievement(progress['user_id'], achievement)

        # Multiple paths
        if completed_paths in [3, 5, 10]:
            achievement = {
                'id': f'path_completion_{completed_paths}',
                'title': f'{completed_paths} Paths Completed',
                'description': f'Successfully completed {completed_paths} learning paths',
                'category': 'path',
                'points': completed_paths * 50
            }
            self.award_achievement(progress['user_id'], achievement)

    def _calculate_study_analytics(self, sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate study session analytics"""
        if not sessions:
            return {
                'total_sessions': 0,
                'total_time': 0,
                'average_session_duration': 0,
                'study_frequency': 0
            }

        total_time = sum(s.get('duration_minutes', 0) for s in sessions)
        avg_duration = total_time / len(sessions) if sessions else 0

        return {
            'total_sessions': len(sessions),
            'total_time': total_time,
            'average_session_duration': round(avg_duration, 1),
            'study_frequency': len(sessions) / 30,  # sessions per day
            'most_common_duration': statistics.mode([s.get('duration_minutes', 0) for s in sessions]) if sessions else 0
        }

    def _calculate_assessment_analytics(self, assessments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate assessment analytics"""
        if not assessments:
            return {
                'total_assessments': 0,
                'average_score': 0,
                'score_trend': 0,
                'best_category': None
            }

        scores = [a.get('score', 0) for a in assessments]
        avg_score = statistics.mean(scores)

        # Calculate trend
        if len(scores) >= 2:
            recent_half = scores[len(scores)//2:]
            earlier_half = scores[:len(scores)//2]
            trend = statistics.mean(recent_half) - statistics.mean(earlier_half)
        else:
            trend = 0

        return {
            'total_assessments': len(assessments),
            'average_score': round(avg_score, 1),
            'best_score': max(scores),
            'score_trend': round(trend, 1),
            'improvement_rate': trend / len(assessments) if assessments else 0
        }

    def _calculate_progress_summary(self, progress: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall progress summary"""
        return {
            'total_study_time': progress.get('total_study_time', 0),
            'current_streak': progress.get('current_streak', 0),
            'longest_streak': progress.get('longest_streak', 0),
            'topics_mastered': len(progress.get('topics_mastered', [])),
            'achievements_earned': len(progress.get('achievements', [])),
            'current_knowledge_level': progress.get('overall_knowledge_level', 'beginner'),
            'learning_paths_completed': len(progress.get('completed_learning_paths', [])),
            'assessment_average': progress.get('assessment_stats', {}).get('average_score', 0)
        }

    def _analyze_learning_patterns(self, sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze learning patterns from recent sessions"""
        if not sessions:
            return {}

        # Analyze preferred times
        study_hours = []
        for session in sessions:
            try:
                hour = datetime.fromisoformat(session['timestamp']).hour
                study_hours.append(hour)
            except:
                continue

        preferred_time = None
        if study_hours:
            hour_counts = {}
            for hour in study_hours:
                hour_counts[hour] = hour_counts.get(hour, 0) + 1
            preferred_hour = max(hour_counts, key=hour_counts.get)

            if preferred_hour < 12:
                preferred_time = 'morning'
            elif preferred_hour < 17:
                preferred_time = 'afternoon'
            else:
                preferred_time = 'evening'

        return {
            'preferred_study_time': preferred_time,
            'average_session_satisfaction': statistics.mean([
                s.get('satisfaction_rating', 3) for s in sessions
                if s.get('satisfaction_rating')
            ]) if any(s.get('satisfaction_rating') for s in sessions) else None,
            'most_studied_topics': self._get_most_frequent_items([
                topic for session in sessions
                for topic in session.get('topics_covered', [])
            ])
        }

    def _get_most_frequent_items(self, items: List[str]) -> List[str]:
        """Get most frequent items from list"""
        if not items:
            return []

        item_counts = {}
        for item in items:
            item_counts[item] = item_counts.get(item, 0) + 1

        # Sort by frequency and return top 3
        sorted_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)
        return [item for item, count in sorted_items[:3]]

    def _generate_progress_recommendations(self, progress: Dict[str, Any],
                                         recent_sessions: List[Dict[str, Any]]) -> List[str]:
        """Generate personalized recommendations based on progress"""
        recommendations = []

        # Study frequency recommendations
        if len(recent_sessions) < 10:  # Less than 10 sessions in 30 days
            recommendations.append("Try to study more regularly - aim for at least 3 sessions per week")

        # Streak recommendations
        current_streak = progress.get('current_streak', 0)
        if current_streak == 0:
            recommendations.append("Start a study streak by studying today!")
        elif current_streak < 3:
            recommendations.append(f"Great start! Try to extend your {current_streak}-day streak")

        # Assessment recommendations
        assessments_taken = len(progress.get('assessments_taken', []))
        if assessments_taken == 0:
            recommendations.append("Take your first assessment to evaluate your knowledge")
        elif assessments_taken < 5:
            recommendations.append("Take more assessments to track your progress better")

        # Topic mastery recommendations
        topics_mastered = len(progress.get('topics_mastered', []))
        if topics_mastered == 0:
            recommendations.append("Focus on mastering your first topic completely")
        elif topics_mastered < 5:
            recommendations.append("Continue building your expertise by mastering more topics")

        # Performance recommendations
        avg_score = progress.get('assessment_stats', {}).get('average_score', 0)
        if avg_score < 60:
            recommendations.append("Review fundamental concepts to improve your assessment scores")
        elif avg_score >= 80:
            recommendations.append("Excellent performance! Consider exploring advanced topics")

        return recommendations[:5]  # Limit to top 5 recommendations

    def _save_progress_data(self, progress: Dict[str, Any]) -> bool:
        """Save progress data to file"""
        try:
            user_id = progress['user_id']
            progress_file = self.progress_dir / f"{user_id}_progress.json"

            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress, f, indent=2, ensure_ascii=False)

            return True

        except Exception as e:
            logger.error(f"Error saving progress data: {e}")
            return False