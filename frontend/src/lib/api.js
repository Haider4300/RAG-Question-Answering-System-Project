const API_BASE = '/api';

// ─── Health ───────────────────────────────────────────────────────────────────

export async function checkHealth() {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error('Health check failed');
  return res.json();
}

// ─── Query ────────────────────────────────────────────────────────────────────

export async function askQuestion(question, chatId = null) {
  const res = await fetch(`${API_BASE}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, chat_id: chatId }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(err.detail || 'Request failed');
  }
  return res.json();
}

/**
 * Streaming query — calls onRoute, onToken, onDone callbacks.
 * onRoute(routeStr)     — called immediately with "general"|"documents"|"web"
 * onToken(tokenStr)     — called for each streamed token
 * onDone({sources, source_type}) — called when stream completes
 */
export async function askQuestionStream(question, chatId, { onRoute, onToken, onDone, onError }) {
  try {
    const res = await fetch(`${API_BASE}/query/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, chat_id: chatId }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(err.detail || 'Stream request failed');
    }

    const reader  = res.body.getReader();
    const decoder = new TextDecoder();
    let   buffer  = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop(); // keep incomplete line in buffer

      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          const msg = JSON.parse(line);
          if (msg.type === 'route')  onRoute?.(msg.route);
          if (msg.type === 'token')  onToken?.(msg.text);
          if (msg.type === 'done')   onDone?.(msg);
        } catch { /* skip malformed lines */ }
      }
    }
  } catch (err) {
    onError?.(err);
  }
}

// ─── Documents ────────────────────────────────────────────────────────────────

export async function uploadDocument(file) {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${API_BASE}/documents/upload`, {
    method: 'POST',
    body: form,
    signal: AbortSignal.timeout(120_000),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Upload failed' }));
    throw new Error(err.detail || 'Upload failed');
  }
  return res.json();
}

export async function listDocuments() {
  const res = await fetch(`${API_BASE}/documents`);
  if (!res.ok) throw new Error('Failed to fetch documents');
  return res.json();
}

export async function getDocumentStatus(docId) {
  const res = await fetch(`${API_BASE}/documents/${docId}/status`);
  if (!res.ok) throw new Error('Failed to fetch document status');
  return res.json();   // { id, status: "pending"|"indexing"|"ready"|"error" }
}

export async function deleteDocument(docId) {
  const res = await fetch(`${API_BASE}/documents/${docId}`, { method: 'DELETE' });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Delete failed' }));
    throw new Error(err.detail || 'Delete failed');
  }
  return res.json();
}

// ─── Chats ────────────────────────────────────────────────────────────────────

export async function listChats() {
  const res = await fetch(`${API_BASE}/chats`);
  if (!res.ok) throw new Error('Failed to fetch chats');
  return res.json();
}

export async function createChat() {
  const res = await fetch(`${API_BASE}/chats`, { method: 'POST' });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Create chat failed' }));
    throw new Error(err.detail || 'Create chat failed');
  }
  return res.json();
}

export async function getChat(chatId) {
  const res = await fetch(`${API_BASE}/chats/${chatId}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Fetch chat failed' }));
    throw new Error(err.detail || 'Fetch chat failed');
  }
  return res.json();
}

export async function deleteChat(chatId) {
  const res = await fetch(`${API_BASE}/chats/${chatId}`, { method: 'DELETE' });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Delete chat failed' }));
    throw new Error(err.detail || 'Delete chat failed');
  }
  return res.json();
}

export async function updateChatTitle(chatId, title) {
  const res = await fetch(`${API_BASE}/chats/${chatId}/title`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Update title failed' }));
    throw new Error(err.detail || 'Update title failed');
  }
  return res.json();
}