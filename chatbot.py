import os
import json
import numpy as np
from typing import List, Dict, Any
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini
api_key = os.environ.get('GEMINI_API_KEY')
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

genai.configure(api_key=api_key)

class PortfolioChatbot:
    def __init__(self, embeddings_file: str = None):
        if embeddings_file is None:
            embeddings_file = os.path.join(os.path.dirname(__file__), "embeddings.json")
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.embedding_model = 'models/text-embedding-004'
        self.documents = []
        self.embeddings = []
        self.load_embeddings(embeddings_file)
        
        # System prompt to define the persona
        self.system_prompt = """
        You are Bhuvan Subramani. Answer questions about your professional background, skills, projects, and experience based ONLY on the provided context.
        
        CRITICAL GUIDELINES:
        1. IDENTITY: Always respond in FIRST PERSON ("I", "my", "I've"). You ARE Bhuvan, not an assistant.
        
        2. ACCURACY: Base ALL answers EXCLUSIVELY on the provided context. Never make up information.
        
        3. CONCISENESS: Keep responses brief and scannable:
           - 1-2 sentences for simple facts
           - 2-3 sentences maximum for complex topics
           - Use bullet points for lists (max 3 items)
        
        4. TONE: Professional yet approachable; enthusiastic about technical work but not overly casual.
        
        5. HANDLING QUESTIONS:
           - For skills/technologies: Mention specific tools and provide a brief example from context
           - For projects: Highlight key impact metrics and technologies used
           - For experience: Focus on achievements and responsibilities
           - For education: State degree, institution, and relevant specializations
           - Keep the answers summarised and concise
           
        6. UNKNOWN INFORMATION: If the answer isn't in the context, respond with:
           "I don't have that specific information in my portfolio. Feel free to ask about my [suggest 2-3 relevant topics from: projects, technical skills, work experience, education."
        
        7. FORMATTING:
           - Use line breaks for readability
           - Keep technical jargon appropriate for the audience
        
        8. PERSONALITY TRAITS:
           - Results-driven: Emphasize outcomes and impact
           - Technical depth: Show expertise without overwhelming
           - Growth-minded: Highlight learning and innovation
        
        REMEMBER: You're helping visitors understand Bhuvan's professional value. Be helpful, accurate, and engaging!
        """

    def load_embeddings(self, file_path: str):
        """Load pre-computed embeddings"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Embeddings file not found: {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        self.documents = data['documents']
        # Convert embeddings to numpy array for efficient cosine similarity
        self.embeddings = np.array([doc['embedding'] for doc in self.documents])
        print(f"✅ Loaded {len(self.documents)} documents from {file_path}")

    def get_query_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for the user query"""
        result = genai.embed_content(
            model=self.embedding_model,
            content=text,
            task_type="retrieval_query"
        )
        return np.array(result['embedding'])

    def find_relevant_context(self, query: str, top_k: int = 3) -> List[str]:
        """Find the most relevant documents for the query"""
        query_embedding = self.get_query_embedding(query)
        
        # Calculate cosine similarity
        # similarity = (A . B) / (||A|| * ||B||)
        # Since embeddings are usually normalized, we can just do dot product
        similarities = np.dot(self.embeddings, query_embedding)
        
        # Get top k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            # Only include if similarity is above a threshold (e.g., 0.3)
            if similarities[idx] > 0.3:
                results.append(self.documents[idx]['content'])
                
        return results

    def chat(self, user_query: str) -> str:
        """Process user query and generate response"""
        # 1. Retrieve relevant context
        relevant_docs = self.find_relevant_context(user_query)
        
        if not relevant_docs:
            return "I'm sorry, I couldn't find any specific information about that in Bhuvan's portfolio. Would you like to ask about his projects, skills, or experience instead?"
            
        context_str = "\n\n".join(relevant_docs)
        
        # 2. Construct prompt
        prompt = f"""
        {self.system_prompt}
        
        Context information:
        {context_str}
        
        User Question: {user_query}
        
        Answer:
        """
        
        # 3. Generate response
        response = self.model.generate_content(prompt)
        return response.text

def main():
    try:
        chatbot = PortfolioChatbot()
        print("\n🤖 Bhuvan's Portfolio Assistant is ready! (Type 'exit' to quit)")
        print("-" * 50)
        
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("Assistant: Goodbye! Have a great day!")
                break
                
            if not user_input.strip():
                continue
                
            print("Assistant: ", end="", flush=True)
            response = chatbot.chat(user_input)
            print(response)
            
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
    