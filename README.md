# Multimodel-emotion-detection
An **empathetic conversational AI system** built with **Flask**, **OpenCV**, **DeepFace**, **TensorFlow/Keras**, **Google Gemini API**, and **Speech Recognition**.  
This project detects **facial emotions**, **voice emotions**, and **text sentiment** to generate **empathetic, context-aware responses** in real time.  
It also provides a video feed, audio interaction, text-to-speech, and database logging for conversation and emotion history.

---

## ✨ Features

- 🎥 **Real-time facial emotion detection** using DeepFace + OpenCV  
- 🎙️ **Voice emotion detection** via a TensorFlow/Keras model (with fallback heuristic analysis)  
- 🗣️ **Speech-to-text** using Google Speech Recognition API  
- 🔊 **Text-to-speech** with `pyttsx3`  
- 🤖 **Context-aware responses** powered by **Google Gemini API**  
- 🧠 **Sentiment analysis** with TextBlob  
- 💾 **SQLite database integration** for:
  - User sessions  
  - Conversation history  
  - Periodic face emotion tracking  
- 📊 **Analytics endpoints** to get emotion history and session statistics  
- 📥 **Download session data** as JSON  
- 🔄 Reset and clear session data dynamically  

---

## 🛠️ Tech Stack

- **Backend**: Flask (Python)  
- **AI/ML**: TensorFlow/Keras, DeepFace, librosa  
- **NLP**: Google Gemini API, TextBlob, SpeechRecognition  
- **Database**: SQLite  
- **Other Tools**: OpenCV, FFmpeg, Pyttsx3  

---

## 📂 Project Structure
📦 emotion-aware-conversational-ai
├── app.py # Main Flask app
├── emotions.db # SQLite database (auto-created)
├── templates/
│ └── index.html # Frontend UI
├── static/ # (Optional) for CSS/JS
├── emotion.h5 # Pre-trained voice emotion model (if available)
└── README.md # Project documentation

## 🚀 Getting Started

### 1️⃣ Clone the repository
```bash
git clone (https://github.com/Arfasil/Multimodel-emotion-detection.git)
cd emotion-aware-conversational-ai
2️⃣ Create a virtual environment & install dependencies
bash
Copy code
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows

pip install -r requirements.txt
3️⃣ Install system dependencies
FFmpeg (required for audio conversion)

PortAudio (required for SpeechRecognition)

On Ubuntu/Debian:

bash
Copy code
sudo apt-get install ffmpeg portaudio19-dev
On Mac (Homebrew):

bash
Copy code
brew install ffmpeg portaudio
On Windows:
Download FFmpeg from ffmpeg.org and add it to PATH.

4️⃣ Configure Google Gemini API
Edit app.py and replace the placeholder key with your own Gemini API key:

python
Copy code
genai.configure(api_key="YOUR_API_KEY_HERE")
5️⃣ Run the application
bash
Copy code
python app.py
Then open your browser at:
👉 http://127.0.0.1:5000/

🎯 API Endpoints
Endpoint	Method	Description
/	GET	Home page with video feed
/video_feed	GET	Live webcam stream with emotion overlay
/get_emotions	GET	Current face & voice emotions
/process_audio	POST	Process uploaded audio (emotion + speech-to-text + bot response)
/get_conversation_history	GET	Fetch full conversation log
/get_face_emotion_history	GET	Fetch periodic face emotion tracking
/clear_conversation	POST	Clear current session’s conversation
/reset_session	POST	Reset entire session
/model_status	GET	Model & session status
/download_data	GET	Download all session data as JSON
/get_session_stats	GET	Session statistics summary

📊 Example Workflow
App detects stable initial face emotion → greets the user empathetically

User speaks → system detects voice emotion + transcribes text

Sentiment + emotion context sent to Gemini API → empathetic response generated

Response spoken aloud via TTS + saved in database

Face emotions logged every 15 seconds for analysis

Session data & analytics available via API

🧑‍💻 Author
A R Mohammed Fasil
💡 Passionate about AI, Deep Learning, and Emotion-Aware Systems.

📜 License
This project is open-source and available under the MIT License.

🙌 Acknowledgements
DeepFace

TensorFlow/Keras

Google Gemini API

SpeechRecognition

TextBlob

OpenCV

