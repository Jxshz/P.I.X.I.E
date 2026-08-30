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

function appendConfirmationCard(actionRequired) {
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('message', 'system-msg');

    const contentDiv = document.createElement('div');
    contentDiv.classList.add('message-content', 'confirmation-card');

    const header = document.createElement('p');
    header.textContent = `P.I.X.I.E. wants to execute: ${actionRequired.tool_name}`;
    header.style.fontWeight = 'bold';

    const params = document.createElement('pre');
    params.textContent = `Parameters:\n${JSON.stringify(actionRequired.arguments, null, 2)}`;
    params.style.background = 'rgba(0,0,0,0.2)';
    params.style.padding = '10px';
    params.style.borderRadius = '5px';
    params.style.marginTop = '10px';
    params.style.whiteSpace = 'pre-wrap';
    params.style.fontFamily = 'monospace';

    const btnContainer = document.createElement('div');
    btnContainer.classList.add('confirmation-buttons');
    btnContainer.style.marginTop = '15px';

    const allowBtn = document.createElement('button');
    allowBtn.textContent = 'Allow';
    allowBtn.classList.add('action-btn');

    const rejectBtn = document.createElement('button');
    rejectBtn.textContent = 'Reject';
    rejectBtn.classList.add('action-btn', 'danger-btn');

    btnContainer.appendChild(allowBtn);
    btnContainer.appendChild(rejectBtn);

    contentDiv.appendChild(header);
    contentDiv.appendChild(params);
    contentDiv.appendChild(btnContainer);
    msgDiv.appendChild(contentDiv);
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;

    const handleConfirm = async (approved) => {
        allowBtn.disabled = true;
        rejectBtn.disabled = true;
        userInput.disabled = true;
        sendBtn.disabled = true;
        setStatus('PROCESSING');

        try {
            const response = await fetch(`${API_BASE}/confirm`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    confirmation_id: actionRequired.confirmation_id,
                    approved: approved
                })
            });

            if (!response.ok) {
                if (response.status === 429) {
                    const data = await response.json();
                    appendMessage(data.response, 'system');
                    const textToSpeak = data.spoken_response ? data.spoken_response : data.response;
                    speakText(textToSpeak);
                    return;
                }
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            appendMessage(data.response, 'system');

            if (data.action_required) {
                appendConfirmationCard(data.action_required);
            }

            const textToSpeak = data.spoken_response ? data.spoken_response : data.response;
            speakText(textToSpeak);
        } catch (error) {
            console.error('Error in confirmation:', error);
            setStatus('ERROR');
            appendMessage('Error: Confirmation request failed.', 'system');
            setTimeout(() => setStatus('IDLE'), 2000);
        } finally {
            userInput.disabled = false;
            sendBtn.disabled = false;
            userInput.focus();
        }
    };

    allowBtn.addEventListener('click', () => handleConfirm(true));
    rejectBtn.addEventListener('click', () => handleConfirm(false));
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
            if (response.status === 429) {
                const data = await response.json();
                appendMessage(data.response, 'system');
                const textToSpeak = data.spoken_response ? data.spoken_response : data.response;
                speakText(textToSpeak);
                return;
            }
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        appendMessage(data.response, 'system');

        if (data.action_required) {
            appendConfirmationCard(data.action_required);
        }

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

// --------------------------------------------------
// Usage Dashboard Logic
// --------------------------------------------------

const usageToggleBtn = document.getElementById('usage-toggle-btn');
const usageDashboard = document.getElementById('usage-dashboard');
const inputContainer = document.querySelector('.input-container');

let dashboardInterval = null;
let isDashboardVisible = false;

usageToggleBtn.addEventListener('click', () => {
    isDashboardVisible = !isDashboardVisible;

    if (isDashboardVisible) {
        chatBox.classList.add('hidden');
        inputContainer.classList.add('hidden');
        usageDashboard.classList.remove('hidden');
        updateDashboard();
        dashboardInterval = setInterval(updateDashboard, 5000);
    } else {
        chatBox.classList.remove('hidden');
        inputContainer.classList.remove('hidden');
        usageDashboard.classList.add('hidden');
        if (dashboardInterval) {
            clearInterval(dashboardInterval);
            dashboardInterval = null;
        }
    }
});

function formatNumber(num) {
    return new Intl.NumberFormat('en-US').format(num);
}

function formatDate(dateStr) {
    const d = new Date(dateStr);
    const today = new Date();

    if (d.getUTCFullYear() === today.getUTCFullYear() &&
        d.getUTCMonth() === today.getUTCMonth() &&
        d.getUTCDate() === today.getUTCDate()) {
        return 'TODAY';
    }

    const options = { month: 'short', day: 'numeric', timeZone: 'UTC' };
    return d.toLocaleDateString('en-US', options).toUpperCase();
}

async function updateDashboard() {
    if (!isDashboardVisible) return;

    try {
        // Fetch live telemetry
        const statusRes = await fetch(`${API_BASE}/status`);
        if (!statusRes.ok) throw new Error('Status fetch failed');
        const statusData = await statusRes.json();

        // Update model
        document.getElementById('active-model').textContent = statusData.model.toUpperCase();

        // Update LIVE metrics
        document.getElementById('tokens-today').textContent = formatNumber(statusData.tokens_day);

        let percentUsed = 0;
        if (statusData.tpd_limit > 0) {
            percentUsed = (statusData.tokens_day / statusData.tpd_limit) * 100;
        }

        let percentStr = percentUsed.toFixed(2);
        if (percentStr.endsWith('.00')) {
            percentStr = percentUsed.toFixed(0);
        }

        const barLength = 20;
        const filledLength = Math.min(barLength, Math.ceil((percentUsed / 100) * barLength));
        const emptyLength = barLength - filledLength;
        const bar = '[' + '-'.repeat(filledLength) + ' '.repeat(emptyLength) + ']';

        document.getElementById('tokens-percent').innerHTML = `<span style="font-family: monospace; white-space: pre;">${bar}</span> ${percentStr}%`;

        document.getElementById('req-today').textContent = `${formatNumber(statusData.requests_day)} / ${formatNumber(statusData.rpd_limit)}`;
        document.getElementById('tok-today').textContent = `${formatNumber(statusData.tokens_day)} / ${formatNumber(statusData.tpd_limit)}`;

        document.getElementById('req-min').textContent = `${formatNumber(statusData.requests_minute)} / ${formatNumber(statusData.rpm_limit)}`;
        document.getElementById('tok-min').textContent = `${formatNumber(statusData.tokens_minute)} / ${formatNumber(statusData.tpm_limit)}`;

        document.getElementById('rem-req-min').textContent = formatNumber(statusData.rpm_remaining);
        document.getElementById('rem-tok-min').textContent = formatNumber(statusData.tpm_remaining);
        document.getElementById('rem-req-day').textContent = formatNumber(statusData.rpd_remaining);
        document.getElementById('rem-tok-day').textContent = formatNumber(statusData.tpd_remaining);

        // Fetch historical usage
        const historyRes = await fetch(`${API_BASE}/usage/history`);
        if (!historyRes.ok) throw new Error('History fetch failed');
        const historyData = await historyRes.json();

        const historyList = document.getElementById('history-list');
        historyList.innerHTML = '';

        let totalBlocks = 0;

        if (historyData.days && historyData.days.length > 0) {
            // Header
            const headerItem = document.createElement('div');
            headerItem.className = 'history-header';
            headerItem.innerHTML = `
                <span>DATE</span>
                <span>REQUESTS</span>
                <span>TOKENS</span>
                <span>BLOCKS</span>
            `;
            historyList.appendChild(headerItem);

            historyData.days.slice(0, 7).forEach(day => {
                totalBlocks += day.rate_limit_blocks;

                const item = document.createElement('div');
                item.className = 'history-row';

                item.innerHTML = `
                    <span data-label="DATE">${formatDate(day.date)}</span>
                    <span data-label="REQUESTS">${formatNumber(day.requests)}</span>
                    <span data-label="TOKENS">${formatNumber(day.tokens)}</span>
                    <span data-label="BLOCKS">${formatNumber(day.rate_limit_blocks)}</span>
                `;

                historyList.appendChild(item);
            });
        } else {
            const empty = document.createElement('div');
            empty.className = 'error-state';
            empty.textContent = 'No usage history available yet.';
            historyList.appendChild(empty);
        }

        // Update PROTECTION
        document.getElementById('protection-blocks').textContent = `${formatNumber(totalBlocks)} RATE-LIMIT BLOCKS`;

    } catch (e) {
        console.error('Error updating dashboard:', e);
        const historyList = document.getElementById('history-list');
        if (historyList.children.length === 0 || historyList.querySelector('.error-state')) {
            historyList.innerHTML = '<div class="error-state">USAGE HISTORY UNAVAILABLE</div>';
        }
    }
}
