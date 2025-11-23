"""
Add this to your existing fixed_backend.py
Text-to-Sign functionality
"""

import json
import os
import string
import random
from pathlib import Path

# Add these endpoints to your existing Flask app

# Global word map
word_map = {}

def load_word_videos():
    """Load word-to-video mapping"""
    global word_map
    
    word_json_path = 'word_videos.json'
    
    if not Path(word_json_path).exists():
        print(f"⚠ {word_json_path} not found. Creating sample mapping...")
        create_sample_word_map()
        return
    
    try:
        with open(word_json_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        
        # Normalize keys
        word_map = {}
        for k, v in raw.items():
            nk = normalize_word(k)
            word_map[nk] = v
        
        print(f"✓ Loaded {len(word_map)} word-to-video mappings")
    except Exception as e:
        print(f"⚠ Error loading word_videos.json: {e}")

def normalize_word(w):
    """Normalize word: lowercase, strip, remove punctuation"""
    w = w.lower().strip()
    return w.translate(str.maketrans('', '', string.punctuation))

def create_sample_word_map():
    """Create sample word_videos.json"""
    sample = {
        "hello": ["videos/hello_1.mp4", "videos/hello_2.mp4"],
        "good": ["videos/good_1.mp4"],
        "morning": ["videos/morning_1.mp4"],
        "good morning": ["videos/good_morning_1.mp4"],
        "thank you": ["videos/thank_you_1.mp4"],
        "please": ["videos/please_1.mp4"],
        "sorry": ["videos/sorry_1.mp4"],
        "yes": ["videos/yes_1.mp4"],
        "no": ["videos/no_1.mp4"],
        "help": ["videos/help_1.mp4"],
        "water": ["videos/water_1.mp4"],
        "food": ["videos/food_1.mp4"]
    }
    
    with open('word_videos.json', 'w') as f:
        json.dump(sample, f, indent=2)
    
    print("✓ Created sample word_videos.json")

def parse_text_to_words(text):
    """
    Parse text into ISL words using greedy longest-match
    Prioritizes phrases over individual words
    """
    phrase = normalize_word(text)
    
    # Sort vocabulary by length (longest first)
    vocab_sorted = sorted(word_map.keys(), key=len, reverse=True)
    
    result = []
    idx = 0
    
    # Greedy matching
    while idx < len(phrase):
        matched = False
        
        for vocab_word in vocab_sorted:
            if phrase[idx:].startswith(vocab_word):
                result.append(vocab_word)
                idx += len(vocab_word)
                matched = True
                break
        
        if not matched:
            # Skip character (space or unknown)
            idx += 1
    
    return result

@app.route('/text-to-sign', methods=['POST'])
def text_to_sign():
    """
    Convert text to ISL sign video URLs
    
    Request JSON:
    {
        "text": "hello good morning"
    }
    
    Response:
    {
        "success": true,
        "text": "hello good morning",
        "words": ["hello", "good morning"],
        "videos": [
            {"word": "hello", "video": "videos/hello_1.mp4"},
            {"word": "good morning", "video": "videos/good_morning_1.mp4"}
        ]
    }
    """
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({
                'success': False,
                'error': 'No text provided'
            }), 400
        
        # Parse text into words
        words = parse_text_to_words(text)
        
        if not words:
            return jsonify({
                'success': False,
                'error': 'No recognized words in input',
                'text': text
            })
        
        # Get videos for each word
        videos = []
        for word in words:
            video_paths = word_map.get(word, [])
            
            if video_paths:
                # Pick random video if multiple available
                video_path = random.choice(video_paths)
                videos.append({
                    'word': word,
                    'video': video_path
                })
            else:
                # Word not found
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
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/available-words', methods=['GET'])
def available_words():
    """Get list of all available ISL words"""
    return jsonify({
        'success': True,
        'words': sorted(word_map.keys()),
        'count': len(word_map)
    })

# Call this in __main__ after loading the model
# load_word_videos()
