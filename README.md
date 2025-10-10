# Jain AI Learning Ecosystem

A comprehensive, modular AI-powered learning platform for Jainism targeting global audiences across different sects, languages, and knowledge levels. The system replaces generic terms with specific sect identifiers and provides personalized learning experiences.

## 🌟 Features

### Core Architecture
- **Central Jain AI Engine**: RAG Pipeline with OpenAI + Vector Store architecture
- **Multilingual Support**: Hindi, Gujarati, English interfaces
- **Sect Customization**: Content tailored for 4 major sects (Digambara, Shwetambara, Terapanthi, Sthanakvasi)
- **User Profiling**: Age, interests, sect, knowledge level detection
- **Offline Mode**: Downloadable content packs

### Modular App Ecosystem
1. **JainPath** – Guided Learning App with personalized paths
2. **JainGPT** – AI Q&A Chatbot with PDF-based answers
3. **JainQuiz** – Gamified Quiz System with adaptive difficulty
4. **Jain Stories & Saints** – Moral Story App with age-appropriate content
5. **Ahimsa Coach** – Mindfulness Tool with daily practices
6. **TirthYatra AI** – Virtual Temple Guide with 360° tours
7. **Pathshala Companion** – Teacher Assistant with lesson planning
8. **Jain Text Explorer** – Research Tool with advanced search

### Critical Fixes Implemented
- **RAG Pipeline**: Answers based ONLY on uploaded PDF content
- **Sect Terminology**: Proper replacement of generic terms with sect-specific identifiers
- **UI/UX**: Universal button loading states and proper modal management
- **Assessment System**: Proper flow with personalized tutoring
- **Progress Tracking**: Real-time updates after each interaction

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Git
- Google API Key (for Gemini AI)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd rag_teacher_qa_system
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set environment variables**
```bash
# Create .env file
GOOGLE_API_KEY=your_gemini_api_key_here
FLASK_DEBUG=False
SECRET_KEY=your_secret_key_here
```

5. **Initialize directories**
```bash
mkdir -p logs data/user_profiles data/learning_paths data/input_pdfs data/output models
```

6. **Run the application**
```bash
python main.py
```

The application will be available at `http://localhost:5000`

## 📁 Project Structure

```
rag_teacher_qa_system/
├── config/
│   └── config.yaml              # Configuration with sect mappings
├── data/
│   ├── user_profiles/          # JSON user profiles
│   ├── learning_paths/         # Personalized learning paths
│   ├── input_pdfs/            # Uploaded PDF storage
│   └── output/                # Vector stores by sect-language
├── models/
│   └── assessment_model.pkl   # ML models
├── static/
│   ├── style.css              # Responsive CSS with loading animations
│   └── script.js              # Universal button loading handlers
├── templates/
│   ├── base.html              # Base template with navigation
│   ├── index.html             # Landing page with sect selection
│   ├── dashboard.html         # Main dashboard
│   ├── assessment.html        # Assessment interface
│   └── tutor.html             # AI tutor chat interface
├── utils/
│   ├── assessment_engine.py   # Assessment creation and scoring
│   ├── content_formatter.py   # Sect terminology replacement
│   ├── learning_path.py       # Personalized learning paths
│   ├── learning_platform.py   # Multilingual support
│   ├── pdf_processor.py       # PDF processing with OCR
│   ├── progress_tracker.py    # User progress tracking
│   ├── rag_pipeline.py        # PDF-based Q&A system
│   ├── sentiment_analyzer.py  # Multilingual sentiment analysis
│   ├── tutor_engine.py        # Context-aware tutoring
│   ├── user_profiler.py       # User profiling and sect detection
│   └── vector_store.py        # ChromaDB with metadata filtering
├── app.py                     # Main Flask application
├── main.py                    # Entry point
└── requirements.txt           # Dependencies
```

## 🔧 Configuration

### Sect Mappings
The system supports four major Jain sects with proper terminology:

- **Digambara**: Digambara Muni (male), Aryika (female)
- **Shwetambara**: Sadhu (male), Sadhvi (female)
- **Terapanthi**: Terapanthi Sadhu/Sadhvi
- **Sthanakvasi**: Sthanakvasi Sadhu/Sadhvi

### Language Support
- English (en)
- Hindi (hi) - हिंदी
- Gujarati (gu) - ગુજરાતી

## 📖 Usage Guide

### 1. Initial Setup
1. Visit the homepage
2. Select your Jain tradition (sect)
3. Choose preferred language
4. Set age group and knowledge level
5. Begin your learning journey

### 2. Upload Content
- Upload sect-specific PDF files
- System processes with OCR if needed
- Content becomes available for personalized Q&A

### 3. Learning Modules
- **AI Tutor**: Get personalized guidance
- **Take Assessment**: Test knowledge and track progress
- **Learning Path**: Follow structured journey
- **View Progress**: Detailed analytics and achievements

### 4. Key Features
- **PDF-Based Answers**: All responses use only your uploaded content
- **Sect-Specific Terminology**: Proper terms for your tradition
- **Progress Tracking**: Real-time updates and achievements
- **Responsive Design**: Works on desktop and mobile

## 🧪 Testing

### Test Scenarios

**Scenario 1: PDF-Based Q&A**
1. Upload Bhagavad Gita PDF for Shwetambara sect in Hindi
2. Ask "Explain Shreemad Bhagwat Geeta"
3. Expected: Answer uses ONLY uploaded PDF content
4. Expected: Terminology uses "Sadhu/Sadhvi" not generic terms

**Scenario 2: Learning Path**
1. Start learning session for Digambara sect in Gujarati
2. Complete assessment → Get personalized path
3. Expected: Content uses "Digambara Muni/Aryika" terminology
4. Expected: Quizzes in Gujarati language

**Scenario 3: UI Interactions**
1. Click any button → Shows loading state
2. Complete action → Returns to normal state
3. Expected: No frozen interfaces, proper feedback

## 🔒 Security Features

- File upload validation (PDF only, size limits)
- Session management with timeouts
- Input sanitization and validation
- Secure file storage with unique naming
- Rate limiting and CSRF protection

## 🚀 Deployment

### Production Setup

1. **Environment Variables**
```bash
export GOOGLE_API_KEY=your_api_key
export FLASK_ENV=production
export SECRET_KEY=secure_random_key
```

2. **WSGI Server**
```bash
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

3. **Docker Deployment** (optional)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "app:app"]
```

## 🛠️ Development

### Local Development
```bash
export FLASK_DEBUG=True
export FLASK_ENV=development
python main.py
```

### Running Tests
```bash
pytest tests/
```

### Code Quality
```bash
black .               # Format code
flake8 .              # Lint code
mypy .                # Type checking
```

## 📊 Monitoring

- Application logs: `logs/app.log`
- Error tracking with Sentry (optional)
- Performance monitoring with built-in metrics
- User analytics and progress tracking

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- 🤖 Generated with [Claude Code](https://claude.ai/code)
- Supporting all Jain traditions: Digambara, Shwetambara, Terapanthi, and Sthanakvasi
- Built with modern web technologies and AI models

## 🆘 Support

For issues and support:
1. Check the troubleshooting section below
2. Search existing issues on GitHub
3. Create a new issue with detailed description

### Troubleshooting

**Q: PDF upload fails**
A: Check file size (max 10MB) and ensure it's a valid PDF

**Q: AI responses not working**
A: Verify GOOGLE_API_KEY is set correctly

**Q: Vector store errors**
A: Ensure data/output directory has write permissions

**Q: Assessment not loading**
A: Upload some PDF content first for content-based assessments

---

**Created with ❤️ for the global Jain community**