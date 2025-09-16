from flask import Flask, render_template, Response, jsonify, request
import cv2
from deepface import DeepFace
import numpy as np
import base64
import sqlite3
import datetime
import librosa
import tensorflow as tf
from tensorflow import keras
import io
import wave
import google.generativeai as genai
import speech_recognition as sr
import pyttsx3
import threading
import json
import os
from werkzeug.utils import secure_filename
import tempfile
import subprocess
import logging
from textblob import TextBlob
import time

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini API
genai.configure(api_key="AIzaSyBgU9aZ-CwmQoGkEc_OUXUZPveMNXI44xQ")  # Replace with your actual API key
model = genai.GenerativeModel("gemini-1.5-flash")

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load voice emotion model - Updated with compatibility handling
voice_emotion_model = None

# Print current working directory for debugging
current_dir = os.getcwd()
logger.info(f"Current working directory: {current_dir}")

# List files in current directory
files_in_dir = os.listdir(current_dir)
h5_files = [f for f in files_in_dir if f.endswith('.h5')]
logger.info(f"Found .h5 files in current directory: {h5_files}")

model_paths = [
    "emotion.h5",
    "./emotion.h5",
    os.path.join(current_dir, "emotion.h5"),
]

def load_model_with_compatibility(model_path):
    """Try multiple methods to load the model with compatibility fixes"""
    
    # Method 1: Direct load
    try:
        model = keras.models.load_model(model_path)
        return model
    except Exception as e:
        logger.warning(f"Direct load failed: {e}")
    
    # Method 2: Load with compile=False
    try:
        model = keras.models.load_model(model_path, compile=False)
        logger.info("Model loaded with compile=False")
        return model
    except Exception as e:
        logger.warning(f"Load with compile=False failed: {e}")
    
    # Method 3: Load with custom objects
    try:
        import tensorflow.keras.utils as utils
        with utils.custom_object_scope({'InputLayer': keras.layers.InputLayer}):
            model = keras.models.load_model(model_path, compile=False)
        logger.info("Model loaded with custom object scope")
        return model
    except Exception as e:
        logger.warning(f"Load with custom objects failed: {e}")
    
    # Method 4: Load weights only if model architecture is known
    try:
        model = keras.Sequential([
            keras.layers.Input(shape=(40, 1)),
            keras.layers.Conv1D(128, 5, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Conv1D(64, 5, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.GlobalMaxPooling1D(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(8, activation='softmax')  # 8 emotions
        ])
        
        model.load_weights(model_path)
        logger.info("Model weights loaded into new architecture")
        return model
    except Exception as e:
        logger.warning(f"Load weights only failed: {e}")
    
    return None

for model_path in model_paths:
    try:
        logger.info(f"Trying to load model from: {model_path}")
        if os.path.exists(model_path):
            logger.info(f"File exists at: {model_path}")
            voice_emotion_model = load_model_with_compatibility(model_path)
            if voice_emotion_model is not None:
                logger.info(f"✅ Voice emotion model loaded successfully from: {model_path}")
                break
            else:
                logger.error(f"❌ All loading methods failed for: {model_path}")
        else:
            logger.info(f"File does not exist at: {model_path}")
    except Exception as e:
        logger.error(f"❌ Unexpected error loading model from {model_path}: {e}")
        continue

if voice_emotion_model is None:
    logger.warning("❌ Voice emotion model could not be loaded due to compatibility issues.")
    logger.info("🔧 This is likely a TensorFlow/Keras version compatibility problem.")
    logger.info("💡 Solutions:")
    logger.info("   1. Use the fallback emotion detection (app will still work)")
    logger.info("   2. Retrain the model with current TensorFlow version")
    logger.info("   3. Try: pip install tensorflow==2.10.0 keras==2.10.0")
    logger.info("🚀 App will continue with fallback emotion detection...")
else:
    logger.info("📊 Model loaded successfully!")
    try:
        logger.info(f"Model input shape: {voice_emotion_model.input_shape}")
        logger.info(f"Model output shape: {voice_emotion_model.output_shape}")
        logger.info(f"Number of parameters: {voice_emotion_model.count_params()}")
    except:
        logger.info("Model info not available, but model loaded successfully")

# Initialize text-to-speech engine
try:
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 150)
    tts_engine.setProperty('volume', 0.8)
    logger.info("TTS engine initialized successfully")
except Exception as e:
    logger.error(f"Warning: Could not initialize TTS engine: {e}")
    tts_engine = None

# Global variables
current_face_emotion = "neutral"
current_voice_emotion = "neutral"
face_emotion_confidence = 0.0
conversation_started = False
initial_emotion_detected = False
user_session_id = None
last_face_update = time.time()
face_emotion_timer = None

# Voice emotion labels
VOICE_EMOTIONS = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# Helper function to ensure JSON serializable values
def make_json_serializable(value):
    """Convert numpy/tensorflow types to native Python types for JSON serialization"""
    if isinstance(value, (np.integer, np.floating, np.ndarray)):
        return float(value)
    elif hasattr(value, 'numpy'):  # TensorFlow tensors
        return float(value.numpy())
    elif hasattr(value, 'item'):  # NumPy scalars
        return value.item()
    return value

# Initialize database
def init_db():
    conn = sqlite3.connect('emotions.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE,
            initial_face_emotion TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            face_emotion TEXT,
            voice_emotion TEXT,
            user_message TEXT,
            bot_response TEXT,
            emotion_confidence REAL,
            sentiment_polarity REAL,
            sentiment_subjectivity REAL,
            sentiment_label TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES user_sessions (session_id)
        )
    ''')
    
    # New table for face emotion tracking every 15 seconds
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS face_emotion_tracking (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            face_emotion TEXT,
            emotion_confidence REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES user_sessions (session_id)
        )
    ''')
    
    conn.commit()
    conn.close()

def save_initial_emotion(session_id, face_emotion):
    conn = sqlite3.connect('emotions.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT OR REPLACE INTO user_sessions (session_id, initial_face_emotion)
        VALUES (?, ?)
    ''', (session_id, face_emotion))
    
    conn.commit()
    conn.close()

def save_face_emotion_periodic(session_id, face_emotion, confidence):
    """Save face emotion to database every 15 seconds"""
    try:
        conn = sqlite3.connect('emotions.db')
        cursor = conn.cursor()
        
        # Ensure confidence is JSON serializable
        confidence = make_json_serializable(confidence)
        
        cursor.execute('''
            INSERT INTO face_emotion_tracking (session_id, face_emotion, emotion_confidence)
            VALUES (?, ?, ?)
        ''', (session_id, face_emotion, confidence))
        
        conn.commit()
        conn.close()
        logger.info(f"✅ Face emotion tracked: {face_emotion} (confidence: {confidence:.2f})")
        return True
    except Exception as e:
        logger.error(f"❌ Error saving face emotion tracking: {e}")
        return False

def save_conversation(session_id, face_emotion, voice_emotion, user_message, bot_response, 
                     emotion_confidence=0.5, sentiment_polarity=0.0, sentiment_subjectivity=0.0, sentiment_label='neutral'):
    try:
        conn = sqlite3.connect('emotions.db')
        cursor = conn.cursor()
        
        # Ensure all numeric values are JSON serializable
        emotion_confidence = make_json_serializable(emotion_confidence)
        sentiment_polarity = make_json_serializable(sentiment_polarity)
        sentiment_subjectivity = make_json_serializable(sentiment_subjectivity)
        
        cursor.execute('''
            INSERT INTO conversation_history 
            (session_id, face_emotion, voice_emotion, user_message, bot_response, 
             emotion_confidence, sentiment_polarity, sentiment_subjectivity, sentiment_label)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (session_id, face_emotion, voice_emotion, user_message, bot_response,
              emotion_confidence, sentiment_polarity, sentiment_subjectivity, sentiment_label))
        
        conn.commit()
        conn.close()
        logger.info(f"✅ Conversation saved: session={session_id}, face={face_emotion}, voice={voice_emotion}")
        return True
    except Exception as e:
        logger.error(f"❌ Error saving conversation: {e}")
        return False

def schedule_face_emotion_update():
    """Schedule periodic face emotion updates every 15 seconds"""
    global face_emotion_timer, user_session_id, current_face_emotion, face_emotion_confidence
    
    if user_session_id and conversation_started:
        save_face_emotion_periodic(user_session_id, current_face_emotion, face_emotion_confidence)
    
    # Schedule next update
    face_emotion_timer = threading.Timer(15.0, schedule_face_emotion_update)
    face_emotion_timer.start()

def convert_audio_to_wav(input_path, output_path):
    """Convert audio to WAV format using ffmpeg"""
    try:
        cmd = [
            'ffmpeg', '-i', input_path, '-acodec', 'pcm_s16le', 
            '-ar', '16000', '-ac', '1', output_path, '-y'
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion failed: {e}")
        try:
            cmd = ['ffmpeg', '-i', input_path, output_path, '-y']
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except:
            return False
    except Exception as e:
        logger.error(f"Audio conversion error: {e}")
        return False

def extract_voice_features(audio_file_path):
    """Extract MFCC features from audio file for emotion detection"""
    try:
        audio_data, sr = librosa.load(audio_file_path, sr=16000)
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfccs, axis=1)
        features = mfcc_mean.reshape(1, 40, 1)
        return features
    except Exception as e:
        logger.error(f"Error extracting voice features: {e}")
        return np.zeros((1, 40, 1))

def predict_voice_emotion(audio_file_path):
    """Predict emotion from voice using the trained model"""
    global voice_emotion_model
    
    if voice_emotion_model is None:
        return predict_emotion_fallback(audio_file_path)
    
    try:
        features = extract_voice_features(audio_file_path)
        if features is not None:
            prediction = voice_emotion_model.predict(features, verbose=0)
            emotion_index = np.argmax(prediction)
            confidence = np.max(prediction)
            
            if emotion_index < len(VOICE_EMOTIONS):
                emotion = VOICE_EMOTIONS[emotion_index]
            else:
                emotion = 'neutral'
                
            # Ensure confidence is JSON serializable
            confidence = make_json_serializable(confidence)
            return emotion, confidence
        return "neutral", 0.5
    except Exception as e:
        logger.error(f"Error predicting voice emotion: {e}")
        return predict_emotion_fallback(audio_file_path)

def predict_emotion_fallback(audio_file_path):
    """Fallback method for emotion prediction when model is not available"""
    try:
        audio_data, sr = librosa.load(audio_file_path, sr=16000)
        
        energy = np.sum(audio_data ** 2) / len(audio_data)
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio_data))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sr))
        
        try:
            tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sr)
        except:
            tempo = 100
        
        logger.info(f"Audio features - Energy: {energy:.6f}, ZCR: {zcr:.3f}, Spectral Centroid: {spectral_centroid:.1f}, Tempo: {tempo:.1f}")
        
        if energy > 0.01 and spectral_centroid > 2000:
            if tempo > 120:
                emotion = "happy"
                confidence = 0.6
            else:
                emotion = "angry"
                confidence = 0.6
        elif energy < 0.005:
            emotion = "sad"
            confidence = 0.5
        elif zcr > 0.1:
            emotion = "surprised"
            confidence = 0.5
        elif spectral_centroid > 3000:
            emotion = "fearful"
            confidence = 0.4
        else:
            emotion = "neutral"
            confidence = 0.5
            
        logger.info(f"Predicted emotion (fallback): {emotion} with confidence: {confidence}")
        return emotion, confidence
            
    except Exception as e:
        logger.error(f"Error in fallback emotion prediction: {e}")
        return "neutral", 0.3

def analyze_sentiment(text):
    """Analyze sentiment of text using TextBlob"""
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        if polarity > 0.1:
            sentiment_label = 'positive'
        elif polarity < -0.1:
            sentiment_label = 'negative'
        else:
            sentiment_label = 'neutral'
            
        # Ensure values are JSON serializable
        polarity = make_json_serializable(polarity)
        subjectivity = make_json_serializable(subjectivity)
        return polarity, subjectivity, sentiment_label
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {e}")
        return 0.0, 0.0, 'neutral'

def generate_contextual_response(face_emotion, voice_emotion, user_message, session_id):
    """Generate response using Gemini API based on emotions and user message"""
    try:
        conn = sqlite3.connect('emotions.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT user_message, bot_response FROM conversation_history 
            WHERE session_id = ? ORDER BY timestamp DESC LIMIT 3
        ''', (session_id,))
        history = cursor.fetchall()
        conn.close()
        
        polarity, subjectivity, sentiment_label = analyze_sentiment(user_message)
        
        emotion_context = {
            'happy': "The user sounds happy and excited. Match their positive energy and enthusiasm.",
            'sad': "The user sounds sad or down. Respond with empathy, comfort, and gentle encouragement.",
            'angry': "The user sounds frustrated or angry. Respond calmly and try to de-escalate while being understanding.",
            'fearful': "The user sounds worried or anxious. Respond with reassurance and support.",
            'surprised': "The user sounds surprised or amazed. Respond with curiosity and engagement.",
            'disgust': "The user sounds disgusted or displeased. Respond with understanding and try to address their concerns.",
            'calm': "The user sounds calm and composed. Respond in a balanced, thoughtful manner.",
            'neutral': "The user sounds neutral. Respond naturally and try to engage them positively."
        }
        
        context = f"""
        You are an empathetic AI assistant engaged in a voice conversation with video emotion detection.
        
        Current user emotions:
        - Face emotion: {face_emotion}
        - Voice emotion: {voice_emotion}
        - Text sentiment: {sentiment_label} (polarity: {polarity:.2f})
        
        User message: "{user_message}"
        
        Emotional context: {emotion_context.get(voice_emotion, "Respond naturally and empathetically.")}
        
        Recent conversation history:
        """
        
        for msg, response in reversed(history):
            context += f"User: {msg}\nBot: {response}\n"
        
        context += """
        
        Guidelines:
        1. Respond empathetically based on the detected emotions from both face and voice
        2. Keep responses conversational and natural for voice interaction
        3. If emotions seem negative (sad, angry, fear), be supportive and understanding
        4. If emotions are positive (happy, surprise), be enthusiastic and engaging
        5. Keep responses concise (2-3 sentences max) as this is voice-based
        6. Don't mention the emotion detection explicitly unless directly asked
        7. Consider both visual and audio emotional cues in your response
        """
        
        response = model.generate_content(context)
        return response.text.strip(), polarity, subjectivity, sentiment_label
    
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "I'm here to listen. Could you please repeat that?", 0.0, 0.0, 'neutral'

def generate_initial_greeting(face_emotion):
    """Generate initial greeting based on detected face emotion"""
    try:
        context = f"""
        You are an empathetic AI assistant starting a voice conversation with video emotion detection. 
        The user's initial face emotion detected is: {face_emotion}
        
        Generate a warm, natural greeting that subtly acknowledges their emotional state without explicitly mentioning emotion detection.
        Keep it conversational and inviting for voice interaction (1-2 sentences max).
        
        Examples based on emotions:
        - Happy: Sound welcoming and positive
        - Sad: Sound gentle and caring
        - Angry: Sound calm and understanding
        - Neutral: Sound friendly and open
        """
        
        response = model.generate_content(context)
        return response.text
    except Exception as e:
        logger.error(f"Error generating initial greeting: {e}")
        return "Hello! I'm here to chat with you. How are you feeling today?"

def generate_frames():
    global current_face_emotion, face_emotion_confidence, conversation_started, initial_emotion_detected, user_session_id, last_face_update
    cap = cv2.VideoCapture(0)
    emotion_detection_count = 0
    stable_emotion_threshold = 10
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
        
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            face_roi = rgb_frame[y:y + h, x:x + w]
            
            try:
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                detected_emotion = result[0]['dominant_emotion']
                confidence = result[0]['emotion'][detected_emotion] / 100.0  # Convert to 0-1 range
                
                # Always update current face emotion and confidence (ensure JSON serializable)
                current_face_emotion = detected_emotion
                face_emotion_confidence = make_json_serializable(confidence)
                
                if not conversation_started:
                    if detected_emotion == current_face_emotion:
                        emotion_detection_count += 1
                    else:
                        emotion_detection_count = 1
                    
                    if emotion_detection_count >= stable_emotion_threshold and not initial_emotion_detected:
                        user_session_id = f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        save_initial_emotion(user_session_id, current_face_emotion)
                        initial_emotion_detected = True
                        conversation_started = True
                        
                        # Start periodic face emotion tracking
                        schedule_face_emotion_update()
                        
                        logger.info(f"🎯 Initial emotion detected: {current_face_emotion}, Session: {user_session_id}")
                        
                        greeting = generate_initial_greeting(current_face_emotion)
                        logger.info(f"📢 Speaking greeting: {greeting}")
                        threading.Thread(target=speak_text, args=(greeting,)).start()
                
                # Draw rectangle and emotion label
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{current_face_emotion} ({confidence:.2f})", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if not conversation_started:
                    cv2.putText(frame, f"Detecting emotion: {emotion_detection_count}/{stable_emotion_threshold}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                else:
                    # Show session info on video feed
                    cv2.putText(frame, f"Session: {user_session_id[-8:]}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
            except Exception as e:
                logger.error(f"Error in emotion detection: {e}")
        
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + base64.b64decode(frame_base64) + b'\r\n')
    
    cap.release()

def speak_text(text):
    """Convert text to speech"""
    if tts_engine is None:
        logger.error("TTS engine not available")
        return
        
    try:
        tts_engine.say(text)
        tts_engine.runAndWait()
    except Exception as e:
        logger.error(f"Error in text-to-speech: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_emotions')
def get_emotions():
    return jsonify({
        "face_emotion": current_face_emotion,
        "voice_emotion": current_voice_emotion,
        "face_confidence": make_json_serializable(face_emotion_confidence),
        "conversation_started": initial_emotion_detected,
        "session_id": user_session_id
    })

@app.route('/process_audio', methods=['POST'])
def process_audio():
    global current_voice_emotion, conversation_started
    
    if not initial_emotion_detected:
        return jsonify({"error": "Please wait for initial emotion detection to complete"}), 400
    
    try:
        logger.info("Received audio processing request")
        
        audio_file = request.files['audio']
        
        if audio_file:
            logger.info(f"Processing audio file: {audio_file.filename}")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp_webm:
                audio_file.save(tmp_webm.name)
                webm_path = tmp_webm.name

            wav_path = webm_path.replace('.webm', '.wav')
            conversion_success = convert_audio_to_wav(webm_path, wav_path)

            if not conversion_success or not os.path.exists(wav_path):
                if os.path.exists(webm_path):
                    os.remove(webm_path)
                return jsonify({'error': 'Audio conversion failed'}), 500

            try:
                logger.info("Loading audio with librosa...")
                audio_data, sample_rate = librosa.load(wav_path, sr=16000)
                audio_duration = len(audio_data) / sample_rate
                logger.info(f"Audio loaded: duration={audio_duration:.2f}s, sr={sample_rate}")

                logger.info("Predicting voice emotion...")
                current_voice_emotion, emotion_confidence = predict_voice_emotion(wav_path)
                logger.info(f"Detected voice emotion: {current_voice_emotion} (confidence: {emotion_confidence:.2f})")
                
                logger.info("Converting speech to text...")
                recognizer = sr.Recognizer()
                user_message = ""
                
                try:
                    with sr.AudioFile(wav_path) as source:
                        recognizer.adjust_for_ambient_noise(source, duration=0.5)
                        audio_data_sr = recognizer.record(source)
                        user_message = recognizer.recognize_google(audio_data_sr)
                        logger.info(f"Recognized text: {user_message}")
                except sr.UnknownValueError:
                    user_message = "I couldn't understand what you said, but I can sense your emotions from your voice."
                    logger.warning("Speech recognition: Could not understand audio")
                except sr.RequestError as e:
                    logger.error(f"Speech recognition error: {e}")
                    user_message = "Sorry, there was an error processing your speech."
                
            finally:
                if os.path.exists(wav_path):
                    os.remove(wav_path)
                if os.path.exists(webm_path):
                    os.remove(webm_path)
            
            logger.info("Generating contextual response...")
            bot_response, polarity, subjectivity, sentiment_label = generate_contextual_response(
                current_face_emotion, current_voice_emotion, user_message, user_session_id
            )
            logger.info(f"Generated response: {bot_response}")
            
            logger.info("Saving conversation to database...")
            save_success = save_conversation(user_session_id, current_face_emotion, current_voice_emotion, 
                            user_message, bot_response, emotion_confidence, polarity, subjectivity, sentiment_label)
            if save_success:
                logger.info("✅ Conversation saved to database successfully")
            else:
                logger.warning("❌ Failed to save conversation to database")
            
            logger.info("Speaking response...")
            threading.Thread(target=speak_text, args=(bot_response,)).start()
            
            conversation_started = True
            
            return jsonify({
                "user_message": user_message,
                "bot_response": bot_response,
                "face_emotion": current_face_emotion,
                "voice_emotion": current_voice_emotion,
                "emotion_confidence": make_json_serializable(emotion_confidence),
                "face_confidence": make_json_serializable(face_emotion_confidence),
                "sentiment": {
                    "polarity": make_json_serializable(polarity),
                    "subjectivity": make_json_serializable(subjectivity),
                    "label": sentiment_label
                },
                "audio_duration": make_json_serializable(audio_duration),
                "database_saved": save_success
            })
        
        return jsonify({"error": "No audio file received"}), 400
    
    except Exception as e:
        import traceback
        logger.error(f"Error processing audio: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/get_conversation_history')
def get_conversation_history():
    """Get conversation history for current session"""
    try:
        if not user_session_id:
            return jsonify({"error": "No active session"}), 400
            
        conn = sqlite3.connect('emotions.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM conversation_history 
            WHERE session_id = ? 
            ORDER BY timestamp ASC
        ''', (user_session_id,))
        
        conversations = cursor.fetchall()
        conn.close()
        
        conversation_list = []
        for conv in conversations:
            conversation_list.append({
                'id': conv[0],
                'session_id': conv[1],
                'face_emotion': conv[2],
                'voice_emotion': conv[3],
                'user_message': conv[4],
                'bot_response': conv[5],
                'emotion_confidence': make_json_serializable(conv[6]),
                'sentiment_polarity': make_json_serializable(conv[7]),
                'sentiment_subjectivity': make_json_serializable(conv[8]),
                'sentiment_label': conv[9],
                'timestamp': conv[10]
            })
        
        return jsonify(conversation_list)
        
    except Exception as e:
        logger.error(f"Error fetching conversation history: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_face_emotion_history')
def get_face_emotion_history():
    """Get face emotion tracking history for current session"""
    try:
        if not user_session_id:
            return jsonify({"error": "No active session"}), 400
            
        conn = sqlite3.connect('emotions.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT face_emotion, emotion_confidence, timestamp 
            FROM face_emotion_tracking 
            WHERE session_id = ? 
            ORDER BY timestamp ASC
        ''', (user_session_id,))
        
        tracking_data = cursor.fetchall()
        conn.close()
        
        tracking_list = []
        for data in tracking_data:
            tracking_list.append({
                'face_emotion': data[0],
                'emotion_confidence': make_json_serializable(data[1]),
                'timestamp': data[2]
            })
        
        return jsonify(tracking_list)
        
    except Exception as e:
        logger.error(f"Error fetching face emotion history: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/clear_conversation', methods=['POST'])
def clear_conversation():
    """Clear conversation history for current session"""
    try:
        if not user_session_id:
            return jsonify({"error": "No active session"}), 400
            
        conn = sqlite3.connect('emotions.db')
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM conversation_history WHERE session_id = ?', (user_session_id,))
        cursor.execute('DELETE FROM face_emotion_tracking WHERE session_id = ?', (user_session_id,))
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Cleared conversation history for session: {user_session_id}")
        return jsonify({"status": "Conversation cleared successfully"})
        
    except Exception as e:
        logger.error(f"Error clearing conversation: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/reset_session', methods=['POST'])
def reset_session():
    """Reset the entire session"""
    global conversation_started, initial_emotion_detected, user_session_id, current_face_emotion, current_voice_emotion, face_emotion_timer
    
    try:
        # Stop the face emotion timer if running
        if face_emotion_timer:
            face_emotion_timer.cancel()
            face_emotion_timer = None
        
        # Reset global variables
        conversation_started = False
        initial_emotion_detected = False
        user_session_id = None
        current_face_emotion = "neutral"
        current_voice_emotion = "neutral"
        
        logger.info("🔄 Session reset successfully")
        return jsonify({"status": "Session reset successfully"})
        
    except Exception as e:
        logger.error(f"Error resetting session: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/model_status')
def model_status():
    """Check the status of the emotion detection model"""
    return jsonify({
        "voice_model_loaded": voice_emotion_model is not None,
        "voice_model_path": "emotion.h5" if voice_emotion_model is not None else None,
        "face_detection_available": True,
        "available_emotions": VOICE_EMOTIONS,
        "fallback_mode": voice_emotion_model is None,
        "session_active": conversation_started,
        "session_id": user_session_id
    })

@app.route('/download_data')
def download_data():
    """Download all session data as JSON"""
    try:
        if not user_session_id:
            return jsonify({"error": "No active session"}), 400
            
        conn = sqlite3.connect('emotions.db')
        cursor = conn.cursor()
        
        # Get session info
        cursor.execute('SELECT * FROM user_sessions WHERE session_id = ?', (user_session_id,))
        session_info = cursor.fetchone()
        
        # Get conversation history
        cursor.execute('SELECT * FROM conversation_history WHERE session_id = ?', (user_session_id,))
        conversations = cursor.fetchall()
        
        # Get face emotion tracking
        cursor.execute('SELECT * FROM face_emotion_tracking WHERE session_id = ?', (user_session_id,))
        face_tracking = cursor.fetchall()
        
        conn.close()
        
        # Format data
        data = {
            "session_info": {
                "session_id": session_info[1] if session_info else user_session_id,
                "initial_emotion": session_info[2] if session_info else "unknown",
                "created_at": session_info[3] if session_info else "unknown"
            },
            "conversations": [],
            "face_emotion_tracking": []
        }
        
        # Format conversations
        for conv in conversations:
            data["conversations"].append({
                "face_emotion": conv[2],
                "voice_emotion": conv[3],
                "user_message": conv[4],
                "bot_response": conv[5],
                "emotion_confidence": make_json_serializable(conv[6]),
                "sentiment_polarity": make_json_serializable(conv[7]),
                "sentiment_subjectivity": make_json_serializable(conv[8]),
                "sentiment_label": conv[9],
                "timestamp": conv[10]
            })
        
        # Format face tracking
        for track in face_tracking:
            data["face_emotion_tracking"].append({
                "face_emotion": track[2],
                "emotion_confidence": make_json_serializable(track[3]),
                "timestamp": track[4]
            })
        
        return jsonify(data)
        
    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_session_stats')
def get_session_stats():
    """Get session statistics"""
    try:
        if not user_session_id:
            return jsonify({"error": "No active session"}), 400
            
        conn = sqlite3.connect('emotions.db')
        cursor = conn.cursor()
        
        # Get conversation count
        cursor.execute('SELECT COUNT(*) FROM conversation_history WHERE session_id = ?', (user_session_id,))
        conversation_count = cursor.fetchone()[0]
        
        # Get face emotion tracking count
        cursor.execute('SELECT COUNT(*) FROM face_emotion_tracking WHERE session_id = ?', (user_session_id,))
        face_tracking_count = cursor.fetchone()[0]
        
        # Get most frequent face emotion
        cursor.execute('''
            SELECT face_emotion, COUNT(*) as count 
            FROM face_emotion_tracking 
            WHERE session_id = ? 
            GROUP BY face_emotion 
            ORDER BY count DESC 
            LIMIT 1
        ''', (user_session_id,))
        most_frequent_emotion = cursor.fetchone()
        
        # Get session duration (if we have face tracking data)
        cursor.execute('''
            SELECT MIN(timestamp), MAX(timestamp) 
            FROM face_emotion_tracking 
            WHERE session_id = ?
        ''', (user_session_id,))
        time_data = cursor.fetchone()
        
        conn.close()
        
        session_duration = "Unknown"
        if time_data[0] and time_data[1]:
            try:
                start_time = datetime.datetime.fromisoformat(time_data[0])
                end_time = datetime.datetime.fromisoformat(time_data[1])
                duration = end_time - start_time
                session_duration = str(duration).split('.')[0]  # Remove microseconds
            except:
                pass
        
        stats = {
            "session_id": user_session_id,
            "conversation_count": conversation_count,
            "face_emotion_updates": face_tracking_count,
            "most_frequent_emotion": most_frequent_emotion[0] if most_frequent_emotion else "Unknown",
            "session_duration": session_duration,
            "current_face_emotion": current_face_emotion,
            "current_voice_emotion": current_voice_emotion
        }
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error getting session stats: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    init_db()
    try:
        app.run(debug=True, threaded=True)
    finally:
        # Cleanup when app shuts down
        if face_emotion_timer:
            face_emotion_timer.cancel()
