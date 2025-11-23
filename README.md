Gestara - Indian Sign Language Recognition System

🌟 Features

 Sign-to-Text Recognition
- Real-time webcam recognition of ISL alphabets (A-Z) and numbers (1-9)
- MediaPipe hand detection with proper landmark preprocessing
- TensorFlow model for accurate sign classification
- Smart autocomplete - Type partial words and get suggestions
- Auto-correct - Fixes spelling mistakes automatically
- Live confidence display with visual feedback
- Keyboard shortcuts for faster interaction

 Text-to-Sign Conversion
- Convert text to animated ISL signs using GIF animations
- Phrase-aware parsing - Recognizes multi-word phrases (e.g., "good morning")
- Quick-add word buttons for common phrases
- Sequential playback of multiple signs
- Expandable word library - Easy to add new signs

 🛠️ Tech Stack

 Backend
- Python 3.8+
- Flask - Web framework
- TensorFlow/Keras - Deep learning model
- MediaPipe - Hand landmark detection
- OpenCV - Image processing
- Pandas - Data handling
- NumPy - Numerical operations

 Frontend
- HTML5/CSS3
- JavaScript (Vanilla)
- Responsive design - Works on desktop and mobile

 AI/ML
- Pre-trained ISL recognition model (model.h5)
- 42-dimensional landmark features
- Random Forest Classifier alternative support



 🚀 Installation

 Prerequisites
- Python 3.8 or higher
- Webcam (for sign recognition)
- Modern web browser (Chrome/Firefox recommended)

 Step 1: Clone Repository


git clone https://github.com/yourusername/gestara-isl.git
cd gestara-isl


 Step 2: Install Dependencies


pip install -r requirements.txt



 Step 4: Prepare GIF Assets

Option A: Use existing GIF files
Place your ISL sign GIFs in the `gifs/` folder.

Option B: Convert videos to GIFs
If you have video files, use the provided converter:


python convert_to_gif.py


 Step 5: Configure Word Mappings

Edit `word_gifs.json` to map words to GIF files:


{
  "hello": ["gifs/hello.gif"],
  "good": ["gifs/good.gif"],
  "morning": ["gifs/morning.gif"],
  "good morning": ["gifs/good_morning.gif"]
}


 ▶️ Usage

 Start Backend Server


python gestara_backend.py


The server will start on `http://localhost:5000`

 Start Frontend Server

In a new terminal:


cd frontend
python -m http.server 8000


Access the application at `http://localhost:8000`

 Using Sign-to-Text Recognition

1. Navigate to `http://localhost:8000/recognition.html`
2. Click "Start Camera"
3. Show ISL signs to the camera
4. Press Space to add detected letters
5. Watch autocomplete suggestions appear
6. Click suggestions or press Enter to add spaces

Keyboard Shortcuts:
- `Space` - Add detected letter
- `Backspace` - Delete last character
- `Enter` - Add space between words
- `C` - Clear all

 Using Text-to-Sign Conversion

1. Navigate to `http://localhost:8000/text_to_sign.html`
2. Type your message in the text box
3. Click "Convert to Sign Language"
4. Watch animated GIF signs appear sequentially
5. Use quick-add buttons for common phrases

 🔧 Configuration

 Add More Words

Edit `word_gifs.json`:


{
  "your_word": ["gifs/your_word.gif"]
}


Then create the corresponding GIF file in the `gifs/` folder.

 Adjust Model Settings

In `gestara_backend.py`, modify MediaPipe parameters:

python
self.hands = self.mp_hands.Hands(
    model_complexity=0,         0=fast, 1=accurate
    max_num_hands=2,            Support 2 hands
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


 Customize Dictionary

Add words to the autocomplete dictionary in `gestara_backend.py`:

python
self.dictionary = set([
    "your", "custom", "words", "here"
])


 🐛 Troubleshooting

Issue: GIFs not displaying
Solution: 
- Check file paths in `word_gifs.`
- Ensure GIF files exist in `gifs/` folder
- Verify backend endpoint `/gifs/<filename>` is working

Issue: Low recognition accuracy
Solution:
- Ensure good lighting
- Position hand clearly in frame
- Hold sign steady for 1-2 seconds
- Check camera focus

Issue: Words not parsing correctly
Solution:
- Verify words exist in `word_gifs.json`
- Check backend console for parsing debug info
- Use `/debug-parse` endpoint to test

 🎯 Performance

- Recognition Speed: ~200ms per frame (5 FPS)
- Accuracy: 85-92% for A-Z, 1-9
- Supported Signs: 35 classes (26 letters + 9 numbers)
- Word Library: Expandable (currently 50+ words)

 🤝 Contributing

Contributions are welcome! To add new ISL signs:

1. Record video of the sign
2. Convert to GIF: `python convert_to_gif.py`
3. Add entry to `word_gifs.`
4. Test with text-to-sign converter
5. Submit pull request

 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

 🙏 Acknowledgments

- Dataset: (https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl)
- MediaPipe: Google's hand tracking solution
- ISL Community: For sign language resources and validation
Thank You
 📧 Contact

Fo
