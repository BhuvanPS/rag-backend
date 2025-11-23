import os
import json
import subprocess
import requests
from typing import List, Dict, Any

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def load_data_from_js(file_path: str) -> Dict[str, Any]:
    """
    Load data from a JavaScript file by executing it with Node.js 
    and printing the JSON representation.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        js_content = f.read()

    # Append the export logic to the JS content
    # We use a block to ensure we capture the variables defined in the file
    full_script = js_content + "\n" + """
    const exportData = {
        personalDetails: typeof personalDetails !== 'undefined' ? personalDetails : {},
        projects: typeof projects !== 'undefined' ? projects : [],
        skillData: typeof skillData !== 'undefined' ? skillData : {},
        journeyData: typeof journeyData !== 'undefined' ? journeyData : {}
    };
    console.log(JSON.stringify(exportData));
    """

    try:
        # Run node and pass script via stdin
        result = subprocess.run(
            ['node'],
            input=full_script,
            capture_output=True,
            text=True,
            check=True
        )
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running Node.js: {e.stderr}")
        raise
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON output from Node.js: {e}")
        if 'result' in locals():
            print(f"Stdout: {result.stdout}")
        raise


class EmbeddingGenerator:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.embedding_url = "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent"
        
    def chunk_text(self, text: str, chunk_size: int = 200) -> List[str]:
        """Split text into chunks by word count"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks if chunks else [text]
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding vector from Gemini API"""
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "models/text-embedding-004",
            "content": {
                "parts": [{"text": text}]
            }
        }
        
        response = requests.post(
            f"{self.embedding_url}?key={self.api_key}",
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise Exception(f"API Error: {response.status_code} - {response.text}")
        
        data = response.json()
        return data['embedding']['values']
    
    def process_portfolio(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process portfolio data and generate embeddings"""
        documents = []
        
        # 1. Process Personal Details (About)
        personal = data.get('personalDetails', {})
        if personal:
            # Basic Info
            about_text = (
                f"Name: {personal.get('name', '')}. "
                f"Title: {personal.get('title', '')}. "
                f"Meta Description: {personal.get('metaDescription', '')}. "
                f"About: {personal.get('about', '')}. "
                f"Long Bio: {' '.join(personal.get('aboutLong', []))}"
            )
            
            for chunk in self.chunk_text(about_text):
                print(f"Processing about chunk: {chunk[:50]}...")
                documents.append({
                    "content": chunk,
                    "embedding": self.get_embedding(chunk),
                    "metadata": {"type": "about"}
                })

            # Contact Info
            contact = personal.get('contact', {})
            if contact:
                contact_text = (
                    f"Contact Information: "
                    f"Email: {contact.get('email', 'N/A')}. "
                    f"Phone: {contact.get('phone', 'N/A')}. "
                    f"LinkedIn: {contact.get('linkedin', 'N/A')}."
                )
                documents.append({
                    "content": contact_text,
                    "embedding": self.get_embedding(contact_text),
                    "metadata": {"type": "contact"}
                })

            # Social Links
            socials = personal.get('socialLinks', [])
            if socials:
                social_text = "Social Media Profiles: " + ", ".join(
                    [f"{s.get('label', 'Profile')}: {s.get('url', '')}" for s in socials]
                )
                documents.append({
                    "content": social_text,
                    "embedding": self.get_embedding(social_text),
                    "metadata": {"type": "socials"}
                })

            # Resume
            resume = personal.get('resume', {})
            if resume:
                resume_text = f"Resume available at: {resume.get('url', '')} (Filename: {resume.get('filename', '')})"
                documents.append({
                    "content": resume_text,
                    "embedding": self.get_embedding(resume_text),
                    "metadata": {"type": "resume"}
                })

            # Key Metrics
            metrics = personal.get('metrics', [])
            if metrics:
                metrics_text = "Key Achievements & Metrics: " + ", ".join(
                    [f"{m.get('label', '')}: {m.get('value', '')}" for m in metrics]
                )
                documents.append({
                    "content": metrics_text,
                    "embedding": self.get_embedding(metrics_text),
                    "metadata": {"type": "metrics"}
                })

            # Services
            if 'services' in personal:
                for service in personal['services']:
                    service_text = f"Service Offered: {service.get('title', '')}. Description: {service.get('desc', '')}"
                    documents.append({
                        "content": service_text,
                        "embedding": self.get_embedding(service_text),
                        "metadata": {"type": "service", "title": service.get('title')}
                    })

        # 2. Process Skills
        skill_data = data.get('skillData', {})
        if skill_data:
            all_skills = []
            for category, skills in skill_data.items():
                skill_names = [s['name'] for s in skills]
                category_text = f"Skills in {category}: {', '.join(skill_names)}"
                
                print(f"Processing skills category: {category}...")
                documents.append({
                    "content": category_text,
                    "embedding": self.get_embedding(category_text),
                    "metadata": {"type": "skills", "category": category}
                })
                all_skills.extend(skill_names)
            
            # Aggregate summary of all skills
            summary_text = f"All Technical Skills: {', '.join(all_skills)}"
            documents.append({
                "content": summary_text,
                "embedding": self.get_embedding(summary_text),
                "metadata": {"type": "skills_summary"}
            })

        # 3. Process Projects
        projects = data.get('projects', [])
        for project in projects:
            description = project.get('description', '')
            if isinstance(description, list):
                description = ' '.join(description)

            project_text = (
                f"Project: {project.get('title', '')}. "
                f"Category: {', '.join(project.get('category', []))}. "
                f"Description: {description}. "
                f"Technologies used: {', '.join(project.get('tags', []))}. "
                f"Key Metrics: {', '.join(project.get('metrics', []))}. "
                f"Link: {project.get('link', '')}."
            )
            
            for chunk in self.chunk_text(project_text):
                print(f"Processing project: {project.get('title', '')[:30]}...")
                documents.append({
                    "content": chunk,
                    "embedding": self.get_embedding(chunk),
                    "metadata": {
                        "type": "project",
                        "title": project.get('title'),
                        "id": project.get('id')
                    }
                })

        # 4. Process Journey (Experience & Education)
        journey = data.get('journeyData', {})
        
        # Experience
        # Experience
        if 'experience' in journey:
            all_experiences = []
            for exp in journey['experience']:
                # Handle description as either array or string for backwards compatibility
                description = exp.get('description', '')
                if isinstance(description, list):
                    description = ' '.join(description)
                
                exp_text = (
                    f"Work Experience: {exp.get('title', '')} at {exp.get('company', '')} "
                    f"({exp.get('period', '')}). "
                    f"Description: {description}. "
                    f"Highlights: {', '.join(exp.get('highlights', []))}."
                )
                all_experiences.append(exp_text)
            
            if all_experiences:
                combined_experience_text = " ".join(all_experiences)
                print(f"Processing all experiences as a single document...")
                documents.append({
                    "content": combined_experience_text,
                    "embedding": self.get_embedding(combined_experience_text),
                    "metadata": {
                        "type": "experience_summary",
                        "count": len(all_experiences)
                    }
                })


        # Education
        if 'education' in journey:
            for edu in journey['education']:
                edu_text = (
                    f"Education: {edu.get('degree', '')} at {edu.get('institution', '')} "
                    f"({edu.get('period', '')}). "
                    f"Description: {edu.get('description', '')}."
                )
                documents.append({
                    "content": edu_text,
                    "embedding": self.get_embedding(edu_text),
                    "metadata": {
                        "type": "education",
                        "institution": edu.get('institution')
                    }
                })

        # Certifications
        if 'certificates' in journey:
            certs = [f"{c.get('name')} from {c.get('issuer')}" for c in journey['certificates']]
            cert_text = f"Certifications: {'; '.join(certs)}"
            documents.append({
                "content": cert_text,
                "embedding": self.get_embedding(cert_text),
                "metadata": {"type": "certifications"}
            })

        return documents


def main():
    # Load .env file if available
    if load_dotenv:
        load_dotenv()
    else:
        print("⚠️  python-dotenv not installed. Skipping .env load.")
        print("   To use .env files, run: pip install python-dotenv")

    # Get API key from environment variable
    api_key = os.environ.get('GEMINI_API_KEY')
    
    if not api_key:
        print("❌ Error: GEMINI_API_KEY environment variable not set")
        print("Set it with: export GEMINI_API_KEY='your-api-key'")
        return
    
    # Path to data.js
    data_js_path = '../Portfolio/data.js'
    
    print("🚀 Starting embedding generation...")
    print(f"📄 Loading data from {data_js_path}...")
    
    try:
        # Load portfolio data
        portfolio_data = load_data_from_js(data_js_path)
        print(f"✅ Successfully loaded data.js")
        print(f"   Found keys: {list(portfolio_data.keys())}")
        
        # Initialize generator
        print(f"🔗 Connecting to Gemini API (text-embedding-004 model)...")
        generator = EmbeddingGenerator(api_key)
        
        # Process portfolio data
        documents = generator.process_portfolio(portfolio_data)
        
        # Prepare output
        output = {
            "documents": documents,
            "metadata": {
                "model": "text-embedding-004",
                "dimension": len(documents[0]['embedding']) if documents else 0,
                "total_chunks": len(documents),
                "source": "data.js"
            }
        }
        
        # Save to file
        output_file = "embeddings.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✅ Successfully generated {len(documents)} embeddings")
        print(f"📁 Saved to: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()