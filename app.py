"""
Main Flask Application for Jain AI Learning Ecosystem
Comprehensive, modular AI-powered learning platform with all critical fixes
"""

import os
import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path
import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from flask import Flask, request, jsonify, render_template, session, redirect, url_for, flash
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
import secrets

# Import utility modules
from utils.rag_pipeline import RAGPipeline
from utils.pdf_processor_wrapper import PDFProcessor
from utils.vector_store_manager import VectorStoreManager
from utils.assessment_engine import AssessmentEngine
from utils.user_profiler import UserProfiler
from utils.learning_path import LearningPathGenerator
from utils.progress_tracker import ProgressTracker
from utils.tutor_engine import TutorEngine
from utils.content_formatter import ContentFormatter
from utils.learning_platform import MultilingualLearningPlatform
from utils.sentiment_analyzer import MultilingualSentimentAnalyzer

# Ensure logs directory exists before configuring logging
Path('logs').mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class JainLearningApp:
    """Main application class for Jain Learning Ecosystem"""

    def __init__(self):
        """Initialize Flask application and components"""
        self.app = Flask(__name__)
        self.setup_app()
        self.initialize_components()
        self.register_routes()

    def setup_app(self):
        """Setup Flask app configuration"""
        # Load configuration
        config_path = Path(__file__).parent / "config" / "config.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # Flask configuration
        self.app.config.update({
            'SECRET_KEY': os.environ.get('SECRET_KEY', secrets.token_hex(32)),
            'MAX_CONTENT_LENGTH': self.config['app']['max_content_length'],
            'UPLOAD_FOLDER': self.config['security']['upload_path'],
            'SESSION_PERMANENT': False,
            'PERMANENT_SESSION_LIFETIME': timedelta(seconds=self.config['security']['session_timeout']),
        })

        # Ensure upload directory exists
        Path(self.app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)
        Path('logs').mkdir(exist_ok=True)

        # Setup proxy fix for deployment
        self.app.wsgi_app = ProxyFix(self.app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

    def initialize_components(self):
        """Initialize all system components"""
        try:
            logger.info("Initializing system components...")

            self.rag_pipeline = RAGPipeline()
            self.pdf_processor = PDFProcessor(self.config)
            self.vector_store = VectorStoreManager(self.config)
            self.assessment_engine = AssessmentEngine()
            self.user_profiler = UserProfiler()
            self.learning_path_generator = LearningPathGenerator()
            self.progress_tracker = ProgressTracker()
            self.tutor_engine = TutorEngine()
            self.content_formatter = ContentFormatter()
            self.learning_platform = MultilingualLearningPlatform()
            self.sentiment_analyzer = MultilingualSentimentAnalyzer()

            logger.info("All components initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise

    def register_routes(self):
        """Register all application routes"""

        # Main routes
        @self.app.route('/')
        def index():
            """Main landing page with sect selection"""
            return render_template('index.html',
                                 sects=self.config['sects']['supported'],
                                 languages=self.config['languages']['supported'])

        @self.app.route('/set_profile', methods=['POST'])
        def set_profile():
            """Set user profile and preferences"""
            try:
                data = request.get_json()

                # Generate or get user ID
                user_id = session.get('user_id', str(uuid.uuid4()))
                session['user_id'] = user_id
                session['sect'] = data.get('sect', 'digambara')
                session['language'] = data.get('language', 'en')
                session['age_group'] = data.get('age_group', 'adult')

                # Create or update user profile
                profile_data = {
                    'sect': session['sect'],
                    'language': session['language'],
                    'age_group': session['age_group'],
                    'knowledge_level': data.get('knowledge_level', 'beginner')
                }

                profile = self.user_profiler.get_user_profile(user_id)
                if profile:
                    self.user_profiler.update_user_profile(user_id, profile_data)
                else:
                    self.user_profiler.create_user_profile(user_id, profile_data)

                return jsonify({'success': True, 'redirect': url_for('dashboard')})

            except Exception as e:
                logger.error(f"Error setting profile: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/dashboard')
        def dashboard():
            """Main dashboard with learning options"""
            if 'user_id' not in session:
                return redirect(url_for('index'))

            user_id = session['user_id']
            sect = session.get('sect', 'digambara')
            language = session.get('language', 'en')

            # Get user progress
            progress = self.progress_tracker.get_user_progress(user_id)
            if not progress:
                progress = self.progress_tracker.initialize_user_progress(user_id, {
                    'sect': sect,
                    'language': language
                })

            # Get content availability
            content_summary = self.rag_pipeline.get_context_summary(sect, language)

            return render_template('dashboard.html',
                                 user_progress=progress,
                                 content_available=content_summary.get('has_content', False),
                                 sect=sect,
                                 language=language)

        # PDF Upload and Processing
        @self.app.route('/upload_pdf', methods=['POST'])
        def upload_pdf():
            """Handle PDF upload and processing"""
            try:
                if 'pdf_file' not in request.files:
                    return jsonify({'success': False, 'error': 'No file selected'}), 400

                file = request.files['pdf_file']
                if file.filename == '':
                    return jsonify({'success': False, 'error': 'No file selected'}), 400

                if not file.filename.lower().endswith('.pdf'):
                    return jsonify({'success': False, 'error': 'Only PDF files are allowed'}), 400

                # Get user session data
                user_id = session.get('user_id', str(uuid.uuid4()))
                sect = session.get('sect', 'digambara')
                language = session.get('language', 'en')

                # Save file
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                safe_filename = f"{timestamp}_{user_id}_{filename}"
                file_path = os.path.join(self.app.config['UPLOAD_FOLDER'], safe_filename)
                file.save(file_path)

                # Process PDF
                result = self.pdf_processor.process_pdf(
                    file_path=file_path,
                    sect=sect,
                    language=language,
                    source_name=filename
                )

                if not result['success']:
                    os.unlink(file_path)  # Clean up failed upload
                    error_message = result.get('error', 'Unknown error processing PDF')

                    # Provide user-friendly error messages
                    if 'too large' in error_message.lower():
                        max_mb = self.config['file_processing']['max_file_size'] / (1024 * 1024)
                        error_message = f'PDF file is too large. Maximum allowed size is {max_mb:.0f} MB.'
                    elif 'no text' in error_message.lower():
                        error_message = 'Could not extract text from this PDF. It may be an image-only PDF or corrupted.'

                    return jsonify({'success': False, 'error': error_message}), 400

                # Add to vector store
                chunks = result['chunks']
                documents = [chunk['content'] for chunk in chunks]
                metadatas = [chunk['metadata'] for chunk in chunks]
                ids = [chunk['id'] for chunk in chunks]

                success = self.vector_store.add_documents(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids,
                    sect=sect,
                    language=language
                )

                if not success:
                    return jsonify({'success': False, 'error': 'Failed to process content'}), 500

                return jsonify({
                    'success': True,
                    'message': f'Successfully processed {result["total_chunks"]} sections from {filename}',
                    'chunks': result['total_chunks'],
                    'characters': result['total_characters']
                })

            except Exception as e:
                logger.error(f"Error uploading PDF: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        # Q&A System
        @self.app.route('/ask_question', methods=['POST'])
        def ask_question():
            """Handle Q&A requests using RAG pipeline"""
            try:
                data = request.get_json()
                question = data.get('question', '').strip()

                if not question:
                    return jsonify({'success': False, 'error': 'Question is required'}), 400

                user_id = session.get('user_id', str(uuid.uuid4()))
                sect = session.get('sect', 'digambara')
                language = session.get('language', 'en')

                # Get user profile for context
                user_profile = self.user_profiler.get_user_profile(user_id)

                # Get vector store for sect and language
                vector_store = self.vector_store.get_vector_store(sect, language)

                # Answer question using RAG pipeline
                response = self.rag_pipeline.answer_question(
                    question=question,
                    vector_store=vector_store,
                    sect=sect,
                    language=language,
                    user_context=user_profile
                )

                # Update user profile with sect detection if needed
                if user_profile and not user_profile.get('sect'):
                    self.user_profiler.detect_sect_from_responses(user_id, [question])

                return jsonify({
                    'success': True,
                    'answer': response['answer'],
                    'key_points': response.get('key_points', []),
                    'related_topics': response.get('related_topics', []),
                    'sources': response.get('sources', []),
                    'confidence': response.get('confidence', 0),
                    'based_on_content': response.get('based_on_uploaded_content', False)
                })

            except Exception as e:
                logger.error(f"Error processing question: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        # Assessment System
        @self.app.route('/start_assessment')
        def start_assessment():
            """Start assessment page"""
            if 'user_id' not in session:
                return redirect(url_for('index'))

            return render_template('assessment.html')

        @self.app.route('/create_assessment', methods=['POST'])
        def create_assessment():
            """Create new assessment"""
            try:
                data = request.get_json()

                user_id = session.get('user_id')
                sect = session.get('sect', 'digambara')
                language = session.get('language', 'en')

                # Get user profile
                user_profile = self.user_profiler.get_user_profile(user_id)
                difficulty = user_profile.get('knowledge_level', 'beginner') if user_profile else 'beginner'

                # Create assessment
                assessment = self.assessment_engine.create_assessment(
                    sect=sect,
                    language=language,
                    difficulty=data.get('difficulty', difficulty),
                    question_count=int(data.get('question_count', 10))
                )

                if 'error' in assessment:
                    return jsonify({'success': False, 'error': assessment['error']}), 500

                # Store assessment in session
                session['current_assessment'] = assessment

                return jsonify({
                    'success': True,
                    'assessment': {
                        'id': assessment['id'],
                        'question_count': assessment['question_count'],
                        'time_limit': assessment['time_limit_minutes'],
                        'questions': assessment['questions']
                    }
                })

            except Exception as e:
                logger.error(f"Error creating assessment: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/submit_assessment', methods=['POST'])
        def submit_assessment():
            """Submit assessment answers"""
            try:
                data = request.get_json()
                answers = data.get('answers', {})

                assessment = session.get('current_assessment')
                if not assessment:
                    return jsonify({'success': False, 'error': 'No active assessment'}), 400

                user_id = session.get('user_id')

                # Score assessment
                results = self.assessment_engine.score_assessment(assessment, answers)

                # Record results in progress tracker
                self.progress_tracker.record_assessment_result(user_id, results)

                # Update knowledge level
                new_level = self.user_profiler.determine_knowledge_level(user_id, [results])

                # Clear session assessment
                session.pop('current_assessment', None)

                return jsonify({
                    'success': True,
                    'results': {
                        'percentage': results['percentage'],
                        'correct_answers': results['correct_answers'],
                        'total_questions': results['total_questions'],
                        'knowledge_level': results['knowledge_level'],
                        'feedback': results['feedback']
                    }
                })

            except Exception as e:
                logger.error(f"Error submitting assessment: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        # Learning Path System
        @self.app.route('/learning_path')
        def learning_path():
            """Learning path page"""
            if 'user_id' not in session:
                return redirect(url_for('index'))

            return render_template('learning_path.html')

        @self.app.route('/generate_learning_path', methods=['POST'])
        def generate_learning_path():
            """Generate personalized learning path"""
            try:
                user_id = session.get('user_id')
                sect = session.get('sect', 'digambara')
                language = session.get('language', 'en')

                # Get user profile
                user_profile = self.user_profiler.get_user_profile(user_id)
                if not user_profile:
                    return jsonify({'success': False, 'error': 'User profile not found'}), 400

                # Get content availability
                content_availability = self.rag_pipeline.get_context_summary(sect, language)

                # Generate learning path
                learning_path = self.learning_path_generator.generate_learning_path(
                    user_profile=user_profile,
                    content_availability=content_availability
                )

                return jsonify({
                    'success': True,
                    'learning_path': {
                        'id': learning_path['id'],
                        'duration_weeks': learning_path['estimated_duration_weeks'],
                        'sessions_per_week': learning_path['sessions_per_week'],
                        'phases': learning_path['phases'],
                        'milestones': learning_path['milestones']
                    }
                })

            except Exception as e:
                logger.error(f"Error generating learning path: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        # Tutoring System
        @self.app.route('/tutor')
        def tutor():
            """AI Tutor page"""
            if 'user_id' not in session:
                return redirect(url_for('index'))

            return render_template('tutor.html')

        @self.app.route('/start_tutoring', methods=['POST'])
        def start_tutoring():
            """Start tutoring session"""
            try:
                data = request.get_json()
                user_id = session.get('user_id')

                # Start tutoring session
                session_data = self.tutor_engine.start_tutoring_session(
                    user_id=user_id,
                    topic=data.get('topic'),
                    learning_objective=data.get('learning_objective')
                )

                # Store session ID
                session['tutoring_session_id'] = session_data.get('session_id')

                return jsonify({
                    'success': True,
                    'session': session_data
                })

            except Exception as e:
                logger.error(f"Error starting tutoring: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/tutor_chat', methods=['POST'])
        def tutor_chat():
            """Handle tutor chat interaction"""
            try:
                data = request.get_json()
                user_input = data.get('message', '').strip()
                input_type = data.get('type', 'question')

                session_id = session.get('tutoring_session_id')
                if not session_id:
                    return jsonify({'success': False, 'error': 'No active tutoring session'}), 400

                # Process user input
                response = self.tutor_engine.process_user_input(
                    session_id=session_id,
                    user_input=user_input,
                    input_type=input_type
                )

                return jsonify({
                    'success': True,
                    'response': response
                })

            except Exception as e:
                logger.error(f"Error in tutor chat: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        # Progress and Analytics
        @self.app.route('/progress')
        def progress():
            """User progress page"""
            if 'user_id' not in session:
                return redirect(url_for('index'))

            user_id = session.get('user_id')

            # Get progress data
            progress_data = self.progress_tracker.get_user_progress(user_id)
            if not progress_data:
                # Initialize if doesn't exist
                sect = session.get('sect', 'digambara')
                language = session.get('language', 'en')
                progress_data = self.progress_tracker.initialize_user_progress(user_id, {
                    'sect': sect,
                    'language': language
                })

            analytics_data = self.progress_tracker.get_progress_analytics(user_id)

            # Format progress data for template
            formatted_progress = {
                'topics_studied': len(progress_data.get('topics_started', [])),
                'assessments_taken': len(progress_data.get('assessments_taken', [])),
                'total_hours': round(progress_data.get('total_study_time', 0) / 60, 1),  # Convert minutes to hours
                'achievements': len(progress_data.get('achievements', [])),
                'knowledge_level': progress_data.get('overall_knowledge_level', 'beginner')
            }

            # Format analytics data
            formatted_analytics = {}
            if analytics_data:
                # Assessment performance
                assessment_analytics = analytics_data.get('assessment_analytics', {})
                formatted_analytics['assessment_average'] = assessment_analytics.get('average_score', 0)
                formatted_analytics['highest_score'] = assessment_analytics.get('best_score', 0)
                formatted_analytics['lowest_score'] = min([a.get('score', 0) for a in progress_data.get('assessments_taken', [])] + [0]) if progress_data.get('assessments_taken') else 0

                # Recent sessions
                recent_sessions = []
                for study_session in progress_data.get('detailed_sessions', [])[-10:]:
                    recent_sessions.append({
                        'date': study_session.get('timestamp', '')[:10] if study_session.get('timestamp') else 'N/A',
                        'activity': 'Study Session',
                        'duration': study_session.get('duration_minutes', 0),
                        'type': 'study',
                        'score': None
                    })

                for assessment in progress_data.get('assessments_taken', [])[-5:]:
                    recent_sessions.append({
                        'date': assessment.get('timestamp', '')[:10] if assessment.get('timestamp') else 'N/A',
                        'activity': 'Assessment',
                        'duration': 30,
                        'type': 'assessment',
                        'score': assessment.get('score', 0)
                    })

                # Sort by date
                recent_sessions.sort(key=lambda x: x.get('date', ''), reverse=True)
                formatted_analytics['recent_sessions'] = recent_sessions[:10]

                # Strengths and improvement areas
                assessment_stats = progress_data.get('assessment_stats', {})
                formatted_analytics['strengths'] = []
                formatted_analytics['improvement_areas'] = []

                if assessment_stats.get('average_score', 0) >= 80:
                    formatted_analytics['strengths'].append('Excellent overall performance')
                if assessment_stats.get('improvement_trend', 0) > 0:
                    formatted_analytics['strengths'].append('Showing consistent improvement')
                if len(progress_data.get('topics_mastered', [])) > 5:
                    formatted_analytics['strengths'].append('Strong topic mastery')

                if assessment_stats.get('average_score', 0) < 60:
                    formatted_analytics['improvement_areas'].append('Review fundamental concepts')
                if progress_data.get('current_streak', 0) == 0:
                    formatted_analytics['improvement_areas'].append('Build a regular study habit')
                if len(progress_data.get('assessments_taken', [])) < 3:
                    formatted_analytics['improvement_areas'].append('Take more assessments to track progress')

            return render_template('progress.html',
                                 progress=formatted_progress,
                                 analytics=formatted_analytics)

        # API Endpoints
        @self.app.route('/api/content_summary')
        def api_content_summary():
            """Get content summary for current user's sect/language"""
            sect = session.get('sect', 'digambara')
            language = session.get('language', 'en')

            summary = self.rag_pipeline.get_context_summary(sect, language)
            return jsonify(summary)

        @self.app.route('/api/user_stats')
        def api_user_stats():
            """Get user statistics"""
            user_id = session.get('user_id')
            if not user_id:
                return jsonify({'error': 'No user session'}), 401

            stats = self.progress_tracker.get_user_statistics(user_id)
            return jsonify(stats)

        # Error handlers
        @self.app.errorhandler(404)
        def not_found(error):
            return render_template('404.html'), 404

        @self.app.errorhandler(500)
        def internal_error(error):
            logger.error(f"Internal error: {error}")
            return render_template('500.html'), 500

        @self.app.errorhandler(413)
        def too_large(error):
            return jsonify({'success': False, 'error': 'File too large'}), 413

    def run(self, debug=False, host='0.0.0.0', port=5000):
        """Run the Flask application"""
        logger.info(f"Starting Jain Learning Ecosystem on {host}:{port}")
        self.app.run(debug=debug, host=host, port=port)

# Create application instance
app_instance = JainLearningApp()
app = app_instance.app

if __name__ == '__main__':
    # Check for required environment variables
    if 'GOOGLE_API_KEY' not in os.environ:
        logger.warning("GOOGLE_API_KEY not found in environment. AI features may be limited.")

    # Run application
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    port = int(os.environ.get('PORT', 5000))

    app_instance.run(debug=debug_mode, port=port)