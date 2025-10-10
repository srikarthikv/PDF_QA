"""
Quick test script to verify Vertex AI Gemini is working
Run this before starting the main application
"""

import os
import sys
import io

# Fix Windows encoding issue
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import vertexai
from vertexai.generative_models import GenerativeModel

# Set credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"D:\Downloads\med-ai-insight-d95d6368a765.json"

print("=" * 60)
print("Testing Vertex AI Gemini Connection")
print("=" * 60)

try:
    # Initialize Vertex AI
    print("\n1. Initializing Vertex AI...")
    vertexai.init(
        project="med-ai-insight",
        location="us-central1"
    )
    print("✅ Vertex AI initialized successfully")

    # Create model
    print("\n2. Creating Gemini model...")
    model = GenerativeModel("gemini-1.5-pro")
    print("✅ Model created successfully")

    # Test generation
    print("\n3. Testing content generation...")
    response = model.generate_content(
        "What is Jainism in one sentence?",
        generation_config={
            'temperature': 0.7,
            'max_output_tokens': 100
        }
    )
    print("✅ Content generated successfully")
    print(f"\nResponse: {response.text}")

    print("\n" + "=" * 60)
    print("🎉 SUCCESS! Vertex AI Gemini is working correctly!")
    print("=" * 60)
    print("\nYou can now run: python main.py")

except Exception as e:
    print("\n" + "=" * 60)
    print("❌ ERROR: Vertex AI connection failed")
    print("=" * 60)
    print(f"\nError: {str(e)}")
    print("\nPossible issues:")
    print("1. Vertex AI API not enabled in your Google Cloud project")
    print("   → Go to: https://console.cloud.google.com/apis/library/aiplatform.googleapis.com")
    print("   → Enable the API for project 'med-ai-insight'")
    print("\n2. Service account lacks permissions")
    print("   → Ensure the service account has 'Vertex AI User' role")
    print("\n3. Billing not enabled")
    print("   → Check: https://console.cloud.google.com/billing")
    sys.exit(1)