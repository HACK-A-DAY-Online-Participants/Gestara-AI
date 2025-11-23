"""
Gestara Complete Backend API
- Sign-to-Text (ISL Recognition)
- Text-to-Sign (Video Playback)
- Autocomplete & Autocorrect
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import tensorflow as tf
import base64
import copy
import itertools
import string
import json
import os
import random
from pathlib import Path
from difflib import get_close_matches

app = Flask(__name__)
CORS(app)

class GesturaISL:
    def __init__(self, model_path='model.h5', word_json='../word_videos.json'):
        print("="*60)
        print("GESTARA - Complete ISL System")
        print("="*60)
        
        # Load recognition model
        self.load_recognition_model(model_path)
        
        # Load text-to-sign mapping
        self.load_word_videos(word_json)
        
        # Load dictionary
        self.load_dictionary()
        
        print("="*60)
        print("✓ System Ready!")
        print("="*60)
    
    def load_recognition_model(self, model_path):
        """Load ISL recognition model"""
        print("\n[1/3] Loading Recognition Model...")
        
        self.model = tf.keras.models.load_model(model_path)
        print(f"  ✓ Model loaded from {model_path}")
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            model_complexity=0,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Classes: 1-9, A-Z
        self.alphabet = ['1','2','3','4','5','6','7','8','9']
        self.alphabet += list(string.ascii_uppercase)
        
        print(f"  ✓ Recognizes {len(self.alphabet)} classes (1-9, A-Z)")
    
    def load_word_videos(self, word_json):
        """Load text-to-sign word-video mapping"""
        print("\n[2/3] Loading Text-to-Sign Mapping...")
        
        if not Path(word_json).exists():
            print(f"  ⚠ {word_json} not found, creating sample...")
            self.create_sample_word_map(word_json)
        
        try:
            with open(word_json, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            
            # Normalize keys
            self.word_map = {}
            for k, v in raw.items():
                nk = self.normalize_word(k)
                self.word_map[nk] = v
            
            print(f"  ✓ Loaded {len(self.word_map)} word-to-video mappings")
        except Exception as e:
            print(f"  ⚠ Error: {e}")
            self.word_map = {}
    
    def load_dictionary(self):
        """Load dictionary for autocomplete"""
        print("\n[3/3] Loading Dictionary...")
        
        self.dictionary = set([
            "hello", "hi", "hey", "good", "morning", "afternoon", "evening", "night",
            "goodbye", "bye", "thanks", "thank", "please", "sorry", "yes", "no",
            "okay", "water", "food", "help", "home", "work", "school", "friend",
            "family", "mother", "father", "brother", "sister", "child", "baby",
            "happy", "sad", "love", "like", "want", "need", "know", "think",
            "come", "go", "eat", "drink", "sleep", "play", "study", "read",
            "today", "tomorrow", "yesterday", "time", "day", "week", "month",
            "what", "when", "where", "who", "how", "why", "which"
        ])
        
        # Add words from word_map
        self.dictionary.update(self.word_map.keys())
        
        print(f"  ✓ Dictionary loaded: {len(self.dictionary)} words")
    
    def create_sample_word_map(self, path):
        """Create sample word_gifs.json"""
        sample = {
            "hello": ["gifs/hello.gif"],
            "good": ["gifs/good.gif"],
            "morning": ["gifs/morning.gif"],
            "good morning": ["gifs/good_morning.gif"],
            "thank you": ["gifs/thank_you.gif"],
            "please": ["gifs/please.gif"],
            "sorry": ["gifs/sorry.gif"],
            "yes": ["gifs/yes.gif"],
            "no": ["gifs/no.gif"],
            "help": ["gifs/help.gif"],
            "water": ["gifs/water.gif"],
            "food": ["gifs/food.gif"]
        }
        
        
        
        with open(path, 'w') as f:
            json.dump(sample, f, indent=2)
    
    # ========== SIGN-TO-TEXT METHODS ==========
    
    def calc_landmark_list(self, image, landmarks):
        """Calculate landmark coordinates"""
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_point = []
        
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_point.append([landmark_x, landmark_y])
        
        return landmark_point
    
    def pre_process_landmark(self, landmark_list):
        """Preprocess landmarks: relative + normalized"""
        temp_landmark_list = copy.deepcopy(landmark_list)
        
        # Relative coordinates
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]
            
            temp_landmark_list[index][0] -= base_x
            temp_landmark_list[index][1] -= base_y
        
        # Flatten
        temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
        
        # Normalize
        max_value = max(list(map(abs, temp_landmark_list)))
        
        def normalize_(n):
            return n / max_value if max_value != 0 else 0
        
        temp_landmark_list = list(map(normalize_, temp_landmark_list))
        
        return temp_landmark_list
    
    def predict_sign(self, frame):
        """Predict letter from frame"""
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image)
        
        if not results.multi_hand_landmarks:
            return None, 0
        
        hand_landmarks = results.multi_hand_landmarks[0]
        landmark_list = self.calc_landmark_list(frame, hand_landmarks)
        pre_processed = self.pre_process_landmark(landmark_list)
        
        df = pd.DataFrame(pre_processed).transpose()
        predictions = self.model.predict(df, verbose=0)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = float(predictions[0][predicted_class])
        
        letter = self.alphabet[predicted_class]
        
        return letter, confidence
    
    # ========== TEXT-TO-SIGN METHODS ==========
    
    def normalize_word(self, w):
        """Normalize word"""
        w = w.lower().strip()
        return w.translate(str.maketrans('', '', string.punctuation))
    
    def parse_text_to_words(self, text):
        """
        Parse text into ISL words - FIXED version
        Handles multi-word phrases and individual words correctly
        """
        # Normalize the input
        text_normalized = self.normalize_word(text)
        
        # Split by spaces first
        tokens = text_normalized.split()
        
        result = []
        i = 0
        
        while i < len(tokens):
            matched = False
            
            # Try matching longest phrase first (4 words, 3 words, 2 words, 1 word)
            for phrase_length in range(min(4, len(tokens) - i), 0, -1):
                # Combine tokens into a phrase
                phrase_tokens = tokens[i:i+phrase_length]
                phrase = ' '.join(phrase_tokens)
                
                # Check if phrase exists in word map
                if phrase in self.word_map:
                    result.append(phrase)
                    i += phrase_length
                    matched = True
                    break
            
            if not matched:
                # Word not found in vocabulary, skip it or add as-is
                # You can choose to skip or add the unknown word
                # result.append(tokens[i])  # Uncomment to include unknown words
                i += 1
        
        return result

    # ========== AUTOCOMPLETE METHODS ==========
    
    def autocomplete(self, partial, max_results=10):
        """Get autocomplete suggestions"""
        if not partial or len(partial) < 2:
            return []
        
        partial_lower = partial.lower()
        matches = [w for w in self.dictionary if w.startswith(partial_lower)]
        
        return sorted(matches, key=len)[:max_results]
    
    def autocorrect(self, word, max_results=5):
        """Auto-correct typos"""
        if not word:
            return []
        
        word_lower = word.lower()
        
        if word_lower in self.dictionary:
            return [word_lower]
        
        matches = get_close_matches(word_lower, self.dictionary, 
                                    n=max_results, cutoff=0.6)
        return matches


# Global system instance
system = None

# ========== API ENDPOINTS ==========
@app.route('/debug-parse', methods=['POST'])
def debug_parse():
    """Debug text parsing"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        normalized = system.normalize_word(text)
        tokens = normalized.split()
        parsed_words = system.parse_text_to_words(text)
        
        return jsonify({
            'success': True,
            'original': text,
            'normalized': normalized,
            'tokens': tokens,
            'parsed_words': parsed_words,
            'available_in_map': {
                word: (word in system.word_map)
                for word in tokens
            }
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/')
def index():
    return jsonify({
        'status': 'online',
        'name': 'Gestara Complete ISL System',
        'version': '3.0',
        'features': [
            'Sign-to-Text Recognition (A-Z, 1-9)',
            'Text-to-Sign Video Playback',
            'Autocomplete & Autocorrect',
            'Word Suggestions'
        ],
        'endpoints': {
            '/health': 'System health check',
            '/predict-letter': 'Recognize sign from image',
            '/text-to-sign': 'Convert text to sign videos',
            '/autocomplete': 'Get word suggestions',
            '/autocorrect': 'Get spelling corrections',
            '/suggest-words': 'Smart suggestions',
            '/available-words': 'List all known words'
        }
    })

@app.route('/health')
def health():
    if system is None:
        return jsonify({'status': 'error', 'message': 'System not loaded'}), 503
    
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'recognition_classes': len(system.alphabet),
        'known_words': len(system.word_map),
        'dictionary_size': len(system.dictionary)
    })

@app.route('/predict-letter', methods=['POST'])
def predict_letter():
    """Sign-to-Text: Predict letter from image"""
    if system is None:
        return jsonify({'success': False, 'error': 'System not loaded'}), 503
    
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'success': False, 'error': 'No image'}), 400
        
        # Decode image
        img_str = data['image']
        if ',' in img_str:
            img_str = img_str.split(',')[1]
        
        img_data = base64.b64decode(img_str)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'success': False, 'error': 'Invalid image'}), 400
        
        # Mirror for natural interaction
        frame = cv2.flip(frame, 1)
        
        # Predict
        letter, confidence = system.predict_sign(frame)
        
        if letter is None:
            return jsonify({
                'success': False,
                'error': 'No hand detected'
            })
        
        return jsonify({
            'success': True,
            'letter': letter,
            'confidence': confidence
        })
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

from flask import send_file

# Add this endpoint to your Flask app

@app.route('/gifs/<path:filename>')
def serve_gif(filename):
    """Serve GIF files"""
    try:
        gif_path = Path('gifs') / filename
        
        if not gif_path.exists():
            return jsonify({'error': 'GIF not found'}), 404
        
        return send_file(
            gif_path,
            mimetype='image/gif',
            as_attachment=False
        )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/text-to-sign', methods=['POST'])
def text_to_sign():
    """Text-to-Sign: Convert text to video URLs"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'success': False, 'error': 'No text'}), 400
        
        # Parse text
        words = system.parse_text_to_words(text)
        
        if not words:
            return jsonify({
                'success': False,
                'error': 'No recognized words',
                'text': text
            })
        
        # Get videos
        videos = []
        for word in words:
            video_paths = system.word_map.get(word, [])
            
            if video_paths:
                video_path = random.choice(video_paths)
                videos.append({
                    'word': word,
                    'video': video_path
                })
            else:
                videos.append({
                    'word': word,
                    'video': None,
                    'error': 'No video available'
                })
        
        return jsonify({
            'success': True,
            'text': text,
            'words': words,
            'videos': videos,
            'count': len(videos)
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/autocomplete', methods=['POST'])
def autocomplete():
    """Autocomplete suggestions"""
    try:
        data = request.get_json()
        partial = data.get('partial', '').strip()
        max_suggestions = data.get('max_suggestions', 10)
        
        if not partial:
            return jsonify({'success': False, 'error': 'No input'}), 400
        
        suggestions = system.autocomplete(partial, max_suggestions)
        
        return jsonify({
            'success': True,
            'partial': partial,
            'suggestions': suggestions,
            'count': len(suggestions)
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/autocorrect', methods=['POST'])
def autocorrect():
    """Auto-correct spelling"""
    try:
        data = request.get_json()
        word = data.get('word', '').strip()
        max_suggestions = data.get('max_suggestions', 5)
        
        if not word:
            return jsonify({'success': False, 'error': 'No word'}), 400
        
        corrections = system.autocorrect(word, max_suggestions)
        
        return jsonify({
            'success': True,
            'word': word,
            'corrections': corrections,
            'count': len(corrections),
            'is_correct': word.lower() in system.dictionary
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/suggest-words', methods=['POST'])
def suggest_words():
    """Smart suggestions"""
    try:
        data = request.get_json()
        current_input = data.get('current_input', '').strip()
        max_suggestions = data.get('max_suggestions', 10)
        
        if not current_input:
            return jsonify({'success': False, 'error': 'No input'}), 400
        
        # Try autocomplete
        suggestions = system.autocomplete(current_input, max_suggestions)
        
        if not suggestions:
            # Try autocorrect
            suggestions = system.autocorrect(current_input, max_suggestions)
            suggestion_type = 'autocorrect'
        else:
            suggestion_type = 'autocomplete'
        
        return jsonify({
            'success': True,
            'input': current_input,
            'suggestions': suggestions,
            'type': suggestion_type,
            'count': len(suggestions)
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/available-words', methods=['GET'])
def available_words():
    """Get all available ISL words"""
    return jsonify({
        'success': True,
        'words': sorted(system.word_map.keys()),
        'count': len(system.word_map)
    })

# ========== MAIN ==========

if __name__ == '__main__':
    print("\n" + "="*60)
    print("GESTARA - Complete ISL Backend")
    print("="*60)
    
    try:
        system = GesturaISL(
            model_path='model.h5',
            word_json='word_videos.json'
        )
        
        print("\n🚀 Server starting on http://localhost:5000")
        print("="*60 + "\n")
        
        app.run(host='0.0.0.0', port=5000, debug=True)
    
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Make sure model.h5 is in the current directory")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
