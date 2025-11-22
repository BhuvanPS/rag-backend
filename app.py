from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot import PortfolioChatbot
import os
from dotenv import load_dotenv

# Load environment variables from parent directory
load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize chatbot
try:
    chatbot = PortfolioChatbot()
    print("✅ Chatbot initialized successfully")
except Exception as e:
    print(f"❌ Error initializing chatbot: {e}")
    chatbot = None

@app.route('/api/chat', methods=['POST'])
def chat():
    if not chatbot:
        return jsonify({"error": "Chatbot not initialized"}), 500
    
    data = request.json
    user_message = data.get('message')
    
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    
    try:
        response = chatbot.chat(user_message)
        return jsonify({"response": response})
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True)
