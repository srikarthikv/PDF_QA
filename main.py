"""
Entry Point for Jain AI Learning Ecosystem
Production-ready entry point with proper configuration
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

# Set environment variables if not already set
if 'FLASK_APP' not in os.environ:
    os.environ['FLASK_APP'] = 'app.py'

if 'FLASK_ENV' not in os.environ:
    os.environ['FLASK_ENV'] = 'production'

# Import the Flask app
try:
    from app import app

    if __name__ == '__main__':
        # Production settings
        debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
        port = int(os.environ.get('PORT', 5000))
        host = os.environ.get('HOST', '0.0.0.0')

        print(f"Starting Jain Learning Ecosystem...")
        print(f"Environment: {'Development' if debug_mode else 'Production'}")
        print(f"Host: {host}")
        print(f"Port: {port}")
        print(f"Access the application at: http://{host}:{port}")

        app.run(
            debug=debug_mode,
            host=host,
            port=port,
            threaded=True
        )

except ImportError as e:
    print(f"Error importing application: {e}")
    print("Please ensure all dependencies are installed:")
    print("pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"Error starting application: {e}")
    sys.exit(1)