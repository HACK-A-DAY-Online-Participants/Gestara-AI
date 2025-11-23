/**
 * Gestara ISL Recognition App
 * Connects frontend to backend API
 */

const API_URL = 'http://localhost:5000';

class ISLApp {
    constructor() {
        // DOM elements
        this.webcam = document.getElementById('webcam');
        this.statusBadge = document.getElementById('status');
        this.currentLetterDisplay = document.getElementById('current-letter');
        this.confidenceFill = document.getElementById('confidence-fill');
        this.confidenceText = document.getElementById('confidence-text');
        this.sentenceDisplay = document.getElementById('sentence');
        this.currentWordDisplay = document.getElementById('current-word');
        this.suggestionsGrid = document.getElementById('suggestions-grid');
        this.suggestionType = document.getElementById('suggestion-type');
        
        // State
        this.isRunning = false;
        this.sentence = '';
        this.currentWord = '';
        this.currentLetter = '';
        this.confidence = 0;
        this.stableFrames = 0;
        
        // Stats
        this.stats = {
            letters: 0,
            words: 0,
            confidences: []
        };
        
        this.init();
    }
    
    init() {
        // Button events
        document.getElementById('start-btn').addEventListener('click', () => this.toggleCamera());
        document.getElementById('add-btn').addEventListener('click', () => this.addLetter());
        document.getElementById('delete-btn').addEventListener('click', () => this.deleteLast());
        document.getElementById('space-btn').addEventListener('click', () => this.addSpace());
        document.getElementById('clear-btn').addEventListener('click', () => this.clearAll());
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (!this.isRunning) return;
            
            if (e.key === ' ') {
                e.preventDefault();
                this.addLetter();
            } else if (e.key === 'Backspace') {
                e.preventDefault();
                this.deleteLast();
            } else if (e.key === 'Enter') {
                e.preventDefault();
                this.addSpace();
            } else if (e.key === 'c' || e.key === 'C') {
                this.clearAll();
            }
        });
        
        this.updateDisplay();
        this.checkAPIConnection();
    }
    
    async checkAPIConnection() {
        try {
            const response = await fetch(`${API_URL}/health`);
            if (response.ok) {
                console.log('✓ API connected');
            }
        } catch (error) {
            console.error('❌ API not reachable. Make sure backend is running on port 5000');
            alert('Backend not running! Please start: python pretrained_backend.py');
        }
    }
    
    async toggleCamera() {
        if (!this.isRunning) {
            await this.startCamera();
        } else {
            this.stopCamera();
        }
    }
    
    async startCamera() {
        try {
            this.statusBadge.textContent = '🔄 Starting camera...';
            
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 1280, height: 720, facingMode: 'user' }
            });
            
            this.webcam.srcObject = stream;
            this.isRunning = true;
            
            document.getElementById('start-btn').textContent = '⏹️ Stop Camera';
            document.getElementById('start-btn').classList.remove('btn-primary');
            document.getElementById('start-btn').style.background = '#ff4444';
            
            document.getElementById('add-btn').disabled = false;
            document.getElementById('delete-btn').disabled = false;
            document.getElementById('space-btn').disabled = false;
            document.getElementById('clear-btn').disabled = false;
            
            this.statusBadge.textContent = '🟢 Camera Active';
            
            // Start recognition loop
            this.recognitionLoop();
            
        } catch (error) {
            this.statusBadge.textContent = '❌ Camera Error';
            alert('Cannot access camera. Please allow camera permissions.');
            console.error(error);
        }
    }
    
    stopCamera() {
        if (this.webcam.srcObject) {
            this.webcam.srcObject.getTracks().forEach(track => track.stop());
        }
        
        this.isRunning = false;
        document.getElementById('start-btn').textContent = '🎥 Start Camera';
        document.getElementById('start-btn').classList.add('btn-primary');
        document.getElementById('start-btn').style.background = '';
        
        document.getElementById('add-btn').disabled = true;
        document.getElementById('delete-btn').disabled = true;
        document.getElementById('space-btn').disabled = true;
        document.getElementById('clear-btn').disabled = true;
        
        this.statusBadge.textContent = '⚪ Camera Stopped';
        this.currentLetterDisplay.textContent = '-';
        this.confidenceFill.style.width = '0%';
        this.confidenceText.textContent = 'Confidence: 0%';
    }
    
    async recognitionLoop() {
        if (!this.isRunning) return;
        
        try {
            // Capture frame
            const canvas = document.createElement('canvas');
            canvas.width = this.webcam.videoWidth;
            canvas.height = this.webcam.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(this.webcam, 0, 0);
            
            // Convert to base64
            const imageData = canvas.toDataURL('image/jpeg', 0.7);
            
            // Send to API
            const response = await fetch(`${API_URL}/predict-letter`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: imageData })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.handlePrediction(result.letter, result.confidence);
            } else {
                this.handleNoPrediction();
            }
            
        } catch (error) {
            console.error('Recognition error:', error);
        }
        
        // Continue loop (5 FPS)
        setTimeout(() => this.recognitionLoop(), 200);
    }
    
    handlePrediction(letter, confidence) {
        this.currentLetterDisplay.textContent = letter;
        this.confidence = confidence;
        
        // Update confidence bar
        const confidencePercent = Math.round(confidence * 100);
        this.confidenceFill.style.width = `${confidencePercent}%`;
        this.confidenceText.textContent = `Confidence: ${confidencePercent}%`;
        
        // Track stability
        if (letter === this.currentLetter && confidence > 0.7) {
            this.stableFrames++;
        } else {
            this.stableFrames = 0;
            this.currentLetter = letter;
        }
        
        // Visual feedback
        if (this.stableFrames > 5) {
            const addBtn = document.getElementById('add-btn');
            addBtn.style.background = '#00ff00';
            addBtn.style.animation = 'pulse 1s infinite';
            addBtn.textContent = `➕ Add "${letter}"`;
        } else {
            const addBtn = document.getElementById('add-btn');
            addBtn.style.background = '';
            addBtn.style.animation = '';
            addBtn.textContent = '➕ Add Letter';
        }
        
        // Track confidence
        this.stats.confidences.push(confidence);
        if (this.stats.confidences.length > 50) {
            this.stats.confidences.shift();
        }
        
        this.updateStats();
    }
    
    handleNoPrediction() {
        this.currentLetterDisplay.textContent = '?';
        this.confidenceFill.style.width = '0%';
        this.confidenceText.textContent = 'No hand detected';
        this.stableFrames = 0;
    }
    
    addLetter() {
        if (!this.currentLetter || this.confidence < 0.6) return;
        
        this.currentWord += this.currentLetter;
        this.stats.letters++;
        this.stableFrames = 0;
        
        // Reset add button
        const addBtn = document.getElementById('add-btn');
        addBtn.style.background = '';
        addBtn.style.animation = '';
        addBtn.textContent = '➕ Add Letter';
        
        this.updateDisplay();
        this.fetchSuggestions();
    }
    
    deleteLast() {
        if (this.currentWord) {
            this.currentWord = this.currentWord.slice(0, -1);
            this.stats.letters = Math.max(0, this.stats.letters - 1);
        } else if (this.sentence) {
            this.sentence = this.sentence.slice(0, -1);
        }
        
        this.updateDisplay();
        this.fetchSuggestions();
    }
    
    addSpace() {
        if (this.currentWord) {
            this.sentence += this.currentWord + ' ';
            this.currentWord = '';
            this.stats.words++;
            this.updateDisplay();
            this.clearSuggestions();
        }
    }
    
    clearAll() {
        this.sentence = '';
        this.currentWord = '';
        this.stats.letters = 0;
        this.stats.words = 0;
        this.updateDisplay();
        this.clearSuggestions();
    }
    
    selectWord(word) {
        this.sentence += word + ' ';
        this.currentWord = '';
        this.stats.words++;
        this.updateDisplay();
        this.clearSuggestions();
    }
    
    async fetchSuggestions() {
        if (this.currentWord.length < 2) {
            this.clearSuggestions();
            return;
        }
        
        try {
            const response = await fetch(`${API_URL}/suggest-words`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    current_input: this.currentWord.toLowerCase(),
                    max_suggestions: 12
                })
            });
            
            const result = await response.json();
            
            if (result.success && result.suggestions.length > 0) {
                this.displaySuggestions(result.suggestions, result.type);
            } else {
                this.clearSuggestions();
            }
        } catch (error) {
            console.error('Suggestion error:', error);
        }
    }
    
    displaySuggestions(suggestions, type) {
        this.suggestionType.textContent = type === 'autocorrect' ? 'Did you mean?' : 'Autocomplete';
        this.suggestionsGrid.innerHTML = '';
        
        suggestions.forEach(word => {
            const btn = document.createElement('button');
            btn.className = 'suggestion-item';
            btn.textContent = word;
            btn.onclick = () => this.selectWord(word);
            this.suggestionsGrid.appendChild(btn);
        });
    }
    
    clearSuggestions() {
        this.suggestionsGrid.innerHTML = '<div style="grid-column: 1/-1; text-align: center; color: #999; padding: 2rem;">Sign more letters to see suggestions</div>';
        this.suggestionType.textContent = '';
    }
    
    updateDisplay() {
        this.sentenceDisplay.textContent = this.sentence + this.currentWord || 'Start signing...';
        this.currentWordDisplay.innerHTML = `Current word: <span style="color: #000;">${this.currentWord || '(empty)'}</span>`;
    }
    
    updateStats() {
        document.getElementById('stat-letters').textContent = this.stats.letters;
        document.getElementById('stat-words').textContent = this.stats.words;
        
        if (this.stats.confidences.length > 0) {
            const avg = this.stats.confidences.reduce((a, b) => a + b) / this.stats.confidences.length;
            document.getElementById('stat-confidence').textContent = `${Math.round(avg * 100)}%`;
        }
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new ISLApp();
    console.log('✓ Gestara ISL App initialized');
});
