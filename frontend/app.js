/**
 * CodeChat AI - Frontend Application
 * Handles repository loading and chat interactions
 */

// API Base URL
const API_BASE = 'http://127.0.0.1:8000';

// DOM Elements
const elements = {
    repoPath: document.getElementById('repoPath'),
    loadRepoBtn: document.getElementById('loadRepoBtn'),
    repoStatus: document.getElementById('repoStatus'),
    statsSection: document.getElementById('statsSection'),
    statFiles: document.getElementById('statFiles'),
    statChunks: document.getElementById('statChunks'),
    statVectors: document.getElementById('statVectors'),
    chatContainer: document.getElementById('chatContainer'),
    messages: document.getElementById('messages'),
    messageInput: document.getElementById('messageInput'),
    sendBtn: document.getElementById('sendBtn'),
    inputHint: document.getElementById('inputHint'),
    loadingOverlay: document.getElementById('loadingOverlay'),
    loadingText: document.getElementById('loadingText'),
    quickBtns: document.querySelectorAll('.quick-btn')
};

// Application State
const state = {
    repositoryLoaded: false,
    isLoading: false,
    currentPath: null
};

// ============================================
// Initialization
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    initEventListeners();
    checkStatus();
});

function initEventListeners() {
    // Load repository
    elements.loadRepoBtn.addEventListener('click', loadRepository);
    elements.repoPath.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') loadRepository();
    });
    
    // Send message
    elements.sendBtn.addEventListener('click', sendMessage);
    elements.messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // Auto-resize textarea
    elements.messageInput.addEventListener('input', autoResizeTextarea);
    
    // Quick action buttons
    elements.quickBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const question = btn.dataset.question;
            if (state.repositoryLoaded && question) {
                elements.messageInput.value = question;
                sendMessage();
            }
        });
    });
}

// ============================================
// API Functions
// ============================================

async function checkStatus() {
    try {
        const response = await fetch(`${API_BASE}/status`);
        const data = await response.json();
        
        if (data.repository_loaded) {
            state.repositoryLoaded = true;
            state.currentPath = data.repository_path;
            elements.repoPath.value = data.repository_path;
            updateRepoStatus('loaded', `Loaded: ${getBasename(data.repository_path)}`);
            updateStats(data.stats);
            enableChat();
        }
    } catch (error) {
        console.log('API not available yet');
    }
}

async function loadRepository() {
    const path = elements.repoPath.value.trim();
    
    if (!path) {
        showError('Please enter a repository path');
        return;
    }
    
    showLoading('Loading repository...');
    updateRepoStatus('loading', 'Scanning files...');
    
    try {
        const response = await fetch(`${API_BASE}/load`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || 'Failed to load repository');
        }
        
        state.repositoryLoaded = true;
        state.currentPath = path;
        
        updateRepoStatus('loaded', `Loaded: ${getBasename(path)}`);
        updateStats(data.stats.vectors);
        enableChat();
        
        // Add success message
        addMessage('assistant', `
            <p>✅ <strong>Repository loaded successfully!</strong></p>
            <p>${data.message}</p>
            <p>You can now ask me anything about your code. Try questions like:</p>
            <ul>
                <li>What is the main purpose of this project?</li>
                <li>Explain the file structure</li>
                <li>How does the authentication work?</li>
            </ul>
        `);
        
    } catch (error) {
        updateRepoStatus('error', error.message);
        addMessage('assistant', `
            <p>❌ <strong>Failed to load repository</strong></p>
            <p>${error.message}</p>
            <p class="hint">Make sure the path exists and contains supported files (.py, .js, .ts, .java, .cpp, .md)</p>
        `);
    } finally {
        hideLoading();
    }
}

async function sendMessage() {
    const question = elements.messageInput.value.trim();
    
    if (!question || !state.repositoryLoaded || state.isLoading) {
        return;
    }
    
    // Add user message
    addMessage('user', `<p>${escapeHtml(question)}</p>`);
    elements.messageInput.value = '';
    autoResizeTextarea();
    
    // Show typing indicator
    const typingId = showTypingIndicator();
    state.isLoading = true;
    disableSend();
    
    try {
        const response = await fetch(`${API_BASE}/ask`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, top_k: 5 })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || 'Failed to get answer');
        }
        
        // Remove typing indicator
        removeTypingIndicator(typingId);
        
        // Format the answer
        let answerHtml = formatAnswer(data.answer);
        
        // Add source files
        if (data.source_files && data.source_files.length > 0) {
            answerHtml += `
                <div class="source-files">
                    <div class="source-files-label">📁 Source Files</div>
                    ${data.source_files.map(f => `<span class="source-file-tag">${escapeHtml(f)}</span>`).join('')}
                </div>
            `;
        }
        
        addMessage('assistant', answerHtml);
        
    } catch (error) {
        removeTypingIndicator(typingId);
        addMessage('assistant', `
            <p>❌ <strong>Error</strong></p>
            <p>${escapeHtml(error.message)}</p>
        `);
    } finally {
        state.isLoading = false;
        enableSend();
    }
}

// ============================================
// UI Functions
// ============================================

function addMessage(role, content) {
    const avatar = role === 'assistant' 
        ? `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
             <path d="M16 18l6-6-6-6M8 6l-6 6 6 6"/>
           </svg>`
        : 'You';
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    messageDiv.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">
            <div class="message-text">${content}</div>
        </div>
    `;
    
    elements.messages.appendChild(messageDiv);
    scrollToBottom();
}

function showTypingIndicator() {
    const id = 'typing-' + Date.now();
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    messageDiv.id = id;
    messageDiv.innerHTML = `
        <div class="message-avatar">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M16 18l6-6-6-6M8 6l-6 6 6 6"/>
            </svg>
        </div>
        <div class="message-content">
            <div class="message-text">
                <div class="typing-indicator">
                    <span></span><span></span><span></span>
                </div>
            </div>
        </div>
    `;
    
    elements.messages.appendChild(messageDiv);
    scrollToBottom();
    return id;
}

function removeTypingIndicator(id) {
    const indicator = document.getElementById(id);
    if (indicator) {
        indicator.remove();
    }
}

function updateRepoStatus(status, text) {
    elements.repoStatus.className = `repo-status ${status}`;
    elements.repoStatus.querySelector('.status-text').textContent = text;
}

function updateStats(stats) {
    if (!stats) return;
    
    elements.statFiles.textContent = stats.unique_files || 0;
    elements.statChunks.textContent = stats.total_chunks || 0;
    elements.statVectors.textContent = stats.total_vectors || 0;
    elements.statsSection.classList.remove('hidden');
}

function enableChat() {
    elements.messageInput.disabled = false;
    elements.sendBtn.disabled = false;
    elements.inputHint.textContent = 'Ask anything about your code';
}

function disableSend() {
    elements.sendBtn.disabled = true;
}

function enableSend() {
    elements.sendBtn.disabled = false;
}

function showLoading(text) {
    elements.loadingText.textContent = text;
    elements.loadingOverlay.classList.remove('hidden');
}

function hideLoading() {
    elements.loadingOverlay.classList.add('hidden');
}

function scrollToBottom() {
    elements.chatContainer.scrollTop = elements.chatContainer.scrollHeight;
}

function autoResizeTextarea() {
    const textarea = elements.messageInput;
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
}

// ============================================
// Utility Functions
// ============================================

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function getBasename(path) {
    return path.split(/[\\/]/).pop();
}

function formatAnswer(text) {
    // Convert markdown-like formatting to HTML
    let html = text
        // Code blocks
        .replace(/```(\w+)?\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>')
        // Inline code
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        // Bold
        .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
        // Italic
        .replace(/\*([^*]+)\*/g, '<em>$1</em>')
        // Line breaks
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br>');
    
    // Wrap in paragraphs
    if (!html.startsWith('<')) {
        html = '<p>' + html + '</p>';
    }
    
    return html;
}

function showError(message) {
    // Simple alert for now, could be enhanced
    updateRepoStatus('error', message);
    setTimeout(() => {
        if (!state.repositoryLoaded) {
            updateRepoStatus('', 'No repository loaded');
        }
    }, 3000);
}
