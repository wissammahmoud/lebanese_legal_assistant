const chatMessages = document.getElementById('chat-messages');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');

let messageHistory = [];

// Adjust textarea height automatically
userInput.addEventListener('input', function () {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});

// Handle Enter to send
userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

sendBtn.addEventListener('click', sendMessage);

async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    // Clear input
    userInput.value = '';
    userInput.style.height = 'auto';

    // Add user message to UI
    appendMessage('user', text);

    // Disable input while processing
    setLoading(true);

    let assistantMessageDiv = null;
    let assistantMessageContent = null;
    let fullResponse = "";
    const API_BASE = ['localhost', '127.0.0.1', '0.0.0.0'].includes(window.location.hostname)
        ? ''
        : '/api/v1/chat/stream';

    try {
        const response = await fetch(`${API_BASE}/api/v1/chat/stream`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-SERVICE-KEY': 'Secret_key'
            },
            body: JSON.stringify({
                query: text,
                history: messageHistory,
                user_context: { platform: 'web_demo' }
            })
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split('\n');

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = JSON.parse(line.slice(6));

                    if (data.type === 'sources') {
                        console.log('Sources received:', data.sources);
                        // Optional: Show sources in a specific UI element
                    } else if (data.type === 'content') {
                        if (!assistantMessageDiv) {
                            // Hide loading indicator as soon as first content arrives
                            setLoading(false);
                            [assistantMessageDiv, assistantMessageContent] = createMessageElement('assistant');
                        }

                        fullResponse += data.content;
                        updateMessageLanguage(assistantMessageContent, fullResponse);
                        assistantMessageContent.innerHTML = formatMarkdown(fullResponse);
                        chatMessages.scrollTop = chatMessages.scrollHeight;
                    } else if (data.type === 'error') {
                        appendMessage('assistant', `⚠️ Error: ${data.content}`);
                    }
                }
            }
        }

        // Update history for context
        messageHistory.push({ role: 'user', content: text });
        messageHistory.push({ role: 'assistant', content: fullResponse });
        if (messageHistory.length > 10) messageHistory = messageHistory.slice(-10);

    } catch (error) {
        console.error('Fetch error:', error);
        appendMessage('assistant', 'Sorry, I couldn\'t connect to the Adl Legal Service. Please make sure the backend is running.');
    } finally {
        setLoading(false);
    }
}

function formatMarkdown(content) {
    let formatted = content.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    formatted = formatted.replace(/\n/g, '<br>');
    return formatted;
}

function updateMessageLanguage(element, content) {
    const containsArabic = /[\u0600-\u06FF]/.test(content);
    if (containsArabic) {
        element.style.direction = 'rtl';
        element.style.textAlign = 'right';
    } else {
        element.style.direction = 'ltr';
        element.style.textAlign = 'left';
    }
}

function createMessageElement(role) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    return [messageDiv, contentDiv];
}

function appendMessage(role, content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    // Auto-detect direction based on content
    const containsArabic = /[\u0600-\u06FF]/.test(content);
    if (containsArabic) {
        contentDiv.style.direction = 'rtl';
        contentDiv.style.textAlign = 'right';
    } else {
        contentDiv.style.direction = 'ltr';
        contentDiv.style.textAlign = 'left';
    }

    // Handle basic markdown-like bolding for demo
    let formattedContent = content.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    formattedContent = formattedContent.replace(/\n/g, '<br>');

    contentDiv.innerHTML = formattedContent;
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);

    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function setLoading(isLoading) {
    if (isLoading) {
        sendBtn.disabled = true;
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message assistant typing-indicator-container';
        typingDiv.id = 'typing-indicator';
        typingDiv.innerHTML = `
            <div class="message-content">
                <div class="typing-indicator">
                    <div class="dot"></div>
                    <div class="dot"></div>
                    <div class="dot"></div>
                </div>
            </div>
        `;
        chatMessages.appendChild(typingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    } else {
        sendBtn.disabled = false;
        const indicator = document.getElementById('typing-indicator');
        if (indicator) indicator.remove();
    }
}
