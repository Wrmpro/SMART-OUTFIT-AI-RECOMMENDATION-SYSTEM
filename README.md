
# Smart Outfit AI Recommendation System
An intelligent, context-aware fashion recommendation engine that uses computer vision and an LLM-orchestrated RAG (Retrieval-Augmented Generation) pipeline to provide personalized styling advice.
## 🚀 Overview
Smart Outfit AI moves beyond static database filtering. It detects user attributes through a computer vision pipeline and uses an LLM to reason about fashion, providing personalized styling rationales. When local data is unavailable, the system intelligently falls back to real-time web retrieval to ensure a seamless user experience.
## ⚙️ Key Features
 * **Computer Vision Pipeline:** Uses OpenCV and a custom CNN to classify user skin tone, serving as the foundation for all styling recommendations.
 * **Hybrid Retrieval System:** Combines high-speed local categorical indexing (Pandas) with an LLM reasoning layer to ground recommendations in real data.
 * **Intelligent Fallback:** An agentic workflow that triggers when local data is sparse ("zero-result" scenarios), using the LLM to synthesize outfit concepts and a search API to pull live product images.
 * **Personalized Stylist Rationale:** Every recommendation includes an AI-generated explanation of *why* the outfit works, based on color theory and the user's occasion.
## 🛠 Tech Stack
 * **Backend:** Python, Flask
 * **AI/ML:** OpenCV, TensorFlow/Keras (CNN), Pandas
 * **LLM Integration:** Groq API (llama-3.3-70b-versatile)
 * **Search Engine:** googlesearch-python
## 📋 Setup & Installation
 1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/smart-outfit-ai.git
   cd smart-outfit-ai
   
   ```
 2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   
   ```
 3. **Configure Environment Variables:**
   Create a .env file in the root directory and add your API keys:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   
   ```
 4. **Run the Application:**
   ```bash
   python app.py
   
   ```
   Access the app via http://localhost:5000 in your browser.
## 🧠 How it Works (The RAG Pattern)
This system treats your local CSV files as a **Knowledge Corpus**:
 1. **Retrieval Phase:** The system searches the local corpus for valid matches.
 2. **Augmentation Phase:** Metadata from retrieved outfits is bundled with the user's skin tone and occasion context.
 3. **Generation Phase:** The LLM receives the bundled context and outputs a structured styling rationale, ensuring the response is grounded in actual fashion data.
## 🤝 Engineering Challenges
 * **Latency Optimization:** We moved heavy filtering logic to local Python/Pandas code and limited LLM calls to the final reasoning step, reducing response times from 3s+ to <500ms.
 * **Resilience:** By implementing a search-based fallback, the system avoids "empty result" errors, handling niche requests (like "Festival" attire) dynamically.
## 📄 License
This project is for educational and screening purposes only.
