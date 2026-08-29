const chatBox = document.getElementById('chat-box');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const micBtn = document.getElementById('mic-btn');
const statusSpan = document.getElementById('agent-status');

const API_BASE = 'http://127.0.0.1:8000';

const VOICE_CONFIG = {
    lang: 'en-US',
    pitch: 1.0,
    rate: 0.95, // Slightly slower for natural conversational feel
    preferredVoices: ['Samantha', 'Google US English', 'Microsoft Zira', 'Alex']
};

let selectedVoice = null;

function loadVoices() {
    const voices = window.speechSynthesis.getVoices();
    if (!voices.length) return;
    
    // Try to find a preferred voice
    for (const name of VOICE_CONFIG.preferredVoices) {
        const voice = voices.find(v => v.name.includes(name) && v.lang.startsWith('en'));
        if (voice) {
            selectedVoice = voice;
            return;
        }
    }
    
    // Fallback to first available English voice
    selectedVoice = voices.find(v => v.lang.startsWith('en')) || voices[0];
}

if ('speechSynthesis' in window) {
    loadVoices();
    window.speechSynthesis.onvoiceschanged = loadVoices;
}

let currentState = 'IDLE';

function setStatus(state) {
    currentState = state;
    statusSpan.textContent = state;
    document.body.setAttribute('data-state', state);
    
    if (state === 'SPEAKING') {
        sendBtn.setAttribute('aria-label', 'Stop speaking');
    } else {
        sendBtn.setAttribute('aria-label', 'Send message');
    }
}

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition = null;
let isRecording = false;

if (!SpeechRecognition) {
    micBtn.style.display = 'none'; // Hide mic if not supported
    console.warn("Web Speech API not supported in this browser.");
}

function startListening() {
    if (!SpeechRecognition) return;
    
    // Force cleanup of any previous instance to release microphone in Safari
    if (recognition) {
        try { recognition.abort(); } catch(e) {}
    }
    
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';

    recognition.onstart = function() {
        isRecording = true;
        setStatus('LISTENING');
        userInput.placeholder = "Listening...";
    };

    recognition.onresult = function(event) {
        const transcript = event.results[0][0].transcript;
        userInput.value = transcript;
        sendMessage(true); // true for voice
        
        // Explicitly abort to force Safari to release the microphone immediately
        // since we already got our single-turn result
        try { recognition.abort(); } catch(e) {}
    };

    recognition.onerror = function(event) {
        console.error("Speech recognition error", event.error);
        if (event.error !== 'aborted') {
            setStatus('ERROR');
            stopRecording();
            setTimeout(() => { if (currentState === 'ERROR') setStatus('IDLE'); }, 2000);
        } else {
            stopRecording();
            if (currentState === 'LISTENING') setStatus('IDLE');
        }
    };

    recognition.onend = function() {
        stopRecording();
        if (currentState === 'LISTENING') {
            setStatus('IDLE');
        }
    };
    
    recognition.start();
}

function stopRecording() {
    isRecording = false;
    userInput.placeholder = "Message P.I.X.I.E...";
}

micBtn.addEventListener('click', () => {
    // Interruption support: stop speech if currently speaking
    if (currentState === 'SPEAKING') {
        if ('speechSynthesis' in window) {
            window.speechSynthesis.cancel();
        }
        setStatus('IDLE');
    }

    if (isRecording) {
        // If user manually clicks mic while recording, stop it
        if (recognition) {
            try { recognition.stop(); } catch(e) {}
        }
    } else {
        startListening();
    }
});

function appendMessage(text, sender) {
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('message');
    msgDiv.classList.add(sender === 'user' ? 'user-msg' : 'system-msg');
    
    const contentDiv = document.createElement('div');
    contentDiv.classList.add('message-content');
    contentDiv.textContent = text;
    
    msgDiv.appendChild(contentDiv);
    
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function speakText(text) {
    if ('speechSynthesis' in window) {
        // Cancel any ongoing speech
        window.speechSynthesis.cancel();

        setStatus('SPEAKING');
        const utterance = new SpeechSynthesisUtterance(text);
        
        if (selectedVoice) {
            utterance.voice = selectedVoice;
        }
        utterance.lang = VOICE_CONFIG.lang;
        utterance.pitch = VOICE_CONFIG.pitch;
        utterance.rate = VOICE_CONFIG.rate;

        utterance.onend = () => {
            if (currentState === 'SPEAKING') {
                setStatus('IDLE');
            }
        };
        utterance.onerror = (e) => {
            if (e.error === 'canceled' || e.error === 'interrupted') return;
            console.error('Speech synthesis error:', e);
            if (currentState === 'SPEAKING') {
                setStatus('ERROR');
                setTimeout(() => {
                    if (currentState === 'ERROR') setStatus('IDLE');
                }, 2000);
            }
        };
        window.speechSynthesis.speak(utterance);
    } else {
        setStatus('IDLE');
    }
}

async function sendMessage(isVoice = false) {
    const text = userInput.value.trim();
    if (!text) return;

    // Interruption support for typed text
    if (currentState === 'SPEAKING') {
        if ('speechSynthesis' in window) {
            window.speechSynthesis.cancel();
        }
    }

    appendMessage(text, 'user');
    userInput.value = '';
    
    // Disable input while waiting
    userInput.disabled = true;
    sendBtn.disabled = true;
    
    setStatus('PROCESSING');

    const endpoint = isVoice ? '/voice' : '/chat';

    try {
        const response = await fetch(`${API_BASE}${endpoint}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: text })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        appendMessage(data.response, 'system');
        const textToSpeak = data.spoken_response ? data.spoken_response : data.response;
        speakText(textToSpeak); // This transitions to SPEAKING and then IDLE
    } catch (error) {
        console.error('Error connecting to backend:', error);
        setStatus('ERROR');
        appendMessage('Error: Cannot connect to P.I.X.I.E. core backend.', 'system');
        setTimeout(() => setStatus('IDLE'), 2000);
    } finally {
        // Re-enable input
        userInput.disabled = false;
        sendBtn.disabled = false;
        userInput.focus();
    }
}

sendBtn.addEventListener('click', () => {
    if (currentState === 'SPEAKING') {
        if ('speechSynthesis' in window) {
            window.speechSynthesis.cancel();
        }
        setStatus('IDLE');
        return;
    }
    sendMessage(false);
});
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendMessage(false);
    }
});
