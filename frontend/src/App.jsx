import { useState, useEffect, useRef, useCallback } from 'react';
import {
  checkHealth,
  askQuestion,
  askQuestionStream,
  uploadDocument,
  listDocuments,
  deleteDocument,
  listChats,
  createChat,
  getChat,
  deleteChat,
} from './lib/api';

// ─── Icons ────────────────────────────────────────────────────────────────────

const PlusIcon    = () => <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" /></svg>;
const TrashIcon   = () => <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg>;
const PaperclipIcon = () => <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L21 7" /></svg>;
const SendIcon    = () => <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" /></svg>;
const XIcon       = () => <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>;
const MenuIcon    = () => <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" /></svg>;
const ChevronIcon = () => <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" /></svg>;
const DocumentIcon = () => <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg>;
const ChatIcon    = () => <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" /></svg>;
const SpinnerIcon = () => <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" /><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" /></svg>;
const CheckIcon   = () => <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" /></svg>;

// ─── Route badge config ───────────────────────────────────────────────────────

const ROUTE_CONFIG = {
  documents: { label: '📄 Documents',   cls: 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' },
  general:   { label: '💡 General AI',  cls: 'bg-indigo-500/20  text-indigo-400  border border-indigo-500/30'  },
  web:       { label: '🌐 Web Search',  cls: 'bg-amber-500/20   text-amber-400   border border-amber-500/30'   },
};

// ─── Helpers ──────────────────────────────────────────────────────────────────

const formatTimestamp = (ts) => {
  if (!ts) return '';
  const date = new Date(ts);
  const now  = new Date();
  const diff = now - date;
  const days = Math.floor(diff / 86400000);
  if (days === 0) return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  if (days === 1) return 'Yesterday';
  if (days < 7)   return date.toLocaleDateString([], { weekday: 'short' });
  return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
};

const FILE_COLORS = {
  pdf:  'bg-red-500/20  text-red-400  border-red-500/30',
  docx: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
  txt:  'bg-gray-500/20 text-gray-400 border-gray-500/30',
};
const fileColor = (t) => FILE_COLORS[t] || 'bg-slate-500/20 text-slate-400 border-slate-500/30';

// ─── Sub-components ───────────────────────────────────────────────────────────

/** Animated typing indicator */
const TypingDots = () => (
  <div className="flex justify-start">
    <div className="bg-slate-800 rounded-2xl rounded-bl-md px-4 py-3">
      <div className="flex gap-1 items-center">
        {[0, 1, 2].map(i => (
          <span
            key={i}
            className="w-2 h-2 rounded-full bg-slate-400 animate-bounce"
            style={{ animationDelay: `${i * 0.15}s` }}
          />
        ))}
      </div>
    </div>
  </div>
);

/** Single message bubble */
const MessageBubble = ({ msg, expandedSources, onToggleSources }) => {
  const isUser = msg.role === 'user';
  const route  = msg.route || msg.sourceType || 'general';
  const badge  = ROUTE_CONFIG[route] || ROUTE_CONFIG.general;

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-[85%] lg:max-w-[75%] rounded-2xl px-4 py-3 ${
        isUser ? 'bg-indigo-600 text-white rounded-br-sm' : 'bg-slate-800 rounded-bl-sm'
      }`}>
        {/* Message text */}
        <p className="whitespace-pre-wrap text-sm leading-relaxed">{msg.content}</p>

        {/* Route badge (assistant only) */}
        {!isUser && (
          <span className={`inline-block mt-2 text-xs px-2 py-0.5 rounded-full font-medium ${badge.cls}`}>
            {badge.label}
          </span>
        )}

        {/* Sources toggle */}
        {!isUser && msg.sources?.length > 0 && (
          <div className="mt-3 border-t border-slate-700/60 pt-3">
            <button
              onClick={() => onToggleSources(msg.id)}
              className="flex items-center gap-1.5 text-xs font-medium text-emerald-400 hover:text-emerald-300 transition-colors"
            >
              <DocumentIcon />
              {expandedSources[msg.id] ? 'Hide' : 'Show'} sources ({msg.sources.length})
            </button>

            {expandedSources[msg.id] && (
              <div className="mt-2 space-y-2">
                {msg.sources.map((src, i) => (
                  <div key={i} className="p-3 bg-slate-900/60 rounded-xl border border-slate-700/50">
                    <div className="flex items-center gap-2 mb-1.5 flex-wrap">
                      <span className="text-xs px-2 py-0.5 bg-indigo-500/20 text-indigo-300 rounded-full">
                        Source {i + 1}
                      </span>
                      <span className="text-xs text-slate-400 truncate">{src.document_name}</span>
                      {src.page > 0 && (
                        <span className="text-xs text-slate-500">p.{src.page}</span>
                      )}
                    </div>
                    <p className="text-xs text-slate-400 whitespace-pre-wrap leading-relaxed line-clamp-4">
                      {src.content}
                    </p>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        <p className="text-xs mt-2 opacity-50 text-right">{formatTimestamp(msg.timestamp)}</p>
      </div>
    </div>
  );
};

/** Document chip with indexing status */
const DocChip = ({ file, onRemove }) => {
  const isIndexing = file.status === 'pending' || file.status === 'indexing';
  const isError    = file.status === 'error';

  return (
    <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs border transition-all ${
      isError    ? 'bg-red-500/20   text-red-400   border-red-500/30'   :
      isIndexing ? 'bg-amber-500/20 text-amber-400 border-amber-500/30' :
      fileColor(file.file_type)
    }`}>
      {isIndexing && <SpinnerIcon />}
      {!isIndexing && !isError && <span className="text-emerald-400"><CheckIcon /></span>}
      {isError    && <span className="text-red-400">!</span>}
      <span className="truncate max-w-[8rem]">{file.filename}</span>
      {isIndexing && <span className="text-[10px] opacity-70">indexing…</span>}
      <button
        onClick={() => onRemove(file.id)}
        className="hover:bg-black/20 rounded-full p-0.5 ml-0.5"
        disabled={isIndexing}
      >
        <XIcon />
      </button>
    </div>
  );
};

/** Toast notification */
const Toast = ({ toast }) => {
  if (!toast) return null;
  const cls = toast.type === 'error'   ? 'bg-red-500/95 text-white'      :
              toast.type === 'success' ? 'bg-emerald-500/95 text-white'  :
                                         'bg-slate-700/95 text-slate-100';
  return (
    <div className={`absolute bottom-full left-6 mb-2 px-4 py-2.5 rounded-xl shadow-xl z-50 text-sm font-medium ${cls}`}>
      {toast.message}
    </div>
  );
};

// ─── Main App ─────────────────────────────────────────────────────────────────

export default function App() {
  const [chats,          setChats]          = useState([]);
  const [currentChatId,  setCurrentChatId]  = useState(null);
  const [messages,       setMessages]       = useState([]);
  const [question,       setQuestion]       = useState('');
  const [loading,        setLoading]        = useState(false);
  const [error,          setError]          = useState(null);
  const [showSidebar,    setShowSidebar]    = useState(true);
  const [collapsed,      setCollapsed]      = useState(false);
  const [isDragging,     setIsDragging]     = useState(false);
  const [uploadedFiles,  setUploadedFiles]  = useState([]);   // {id, filename, file_type, status}
  const [expandedSources,setExpandedSources]= useState({});
  const [toast,          setToast]          = useState(null);
  const [isUploading,    setIsUploading]    = useState(false);

  const messagesEndRef = useRef(null);
  const fileInputRef   = useRef(null);
  const inputAreaRef   = useRef(null);

  // ── Initialise ──────────────────────────────────────────────────────────────
  useEffect(() => {
    const savedId = localStorage.getItem('lastChatId');
    const ROUTE_FIX = { document: 'documents', Document: 'documents' };
    const normalizeMsg = (m, i) => {
      const r = ROUTE_FIX[m.route] || m.route || m.sourceType || 'general';
      return { ...m, id: `m${i}`, route: r, sourceType: r };
    };
    listChats().then(list => {
      setChats(list);
      const target = savedId ? list.find(c => c.id === savedId) : list[0];
      if (target) {
        setCurrentChatId(target.id);
        setMessages(target.messages.map(normalizeMsg));
      }
    }).catch(console.error);
  }, []);

  useEffect(() => {
    if (currentChatId) localStorage.setItem('lastChatId', currentChatId);
  }, [currentChatId]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  useEffect(() => {
    const handle = () => setShowSidebar(window.innerWidth >= 1024);
    handle();
    window.addEventListener('resize', handle);
    return () => window.removeEventListener('resize', handle);
  }, []);

  // ── Document status polling ─────────────────────────────────────────────────
  useEffect(() => {
    const pending = uploadedFiles.filter(f => f.status === 'pending' || f.status === 'indexing');
    if (!pending.length) return;

    const interval = setInterval(async () => {
      let allDone = true;
      const updated = await Promise.all(
        uploadedFiles.map(async (f) => {
          if (f.status !== 'pending' && f.status !== 'indexing') return f;
          try {
            const res  = await fetch(`/api/documents/${f.id}/status`);
            const data = await res.json();
            if (data.status === 'pending' || data.status === 'indexing') allDone = false;
            if (data.status === 'ready' && f.status !== 'ready') {
              showToast(`✅ ${f.filename} ready!`, 'success');
            }
            return { ...f, status: data.status };
          } catch {
            return f;
          }
        })
      );
      setUploadedFiles(updated);
      if (allDone) clearInterval(interval);
    }, 2000);

    return () => clearInterval(interval);
  }, [uploadedFiles]);

  // ── Helpers ─────────────────────────────────────────────────────────────────
  const showToast = (message, type = 'info') => {
    setToast({ message, type });
    setTimeout(() => setToast(null), 3000);
  };

  // ── Upload ──────────────────────────────────────────────────────────────────
  const handleFileUpload = useCallback(async (files) => {
    if (!files?.length) return;
    setIsUploading(true);

    for (const file of files) {
      const ext = file.name.split('.').pop().toLowerCase();
      if (!['pdf', 'docx', 'txt'].includes(ext)) {
        showToast(`❌ Unsupported: .${ext}`, 'error');
        continue;
      }
      try {
        showToast(`📤 Uploading ${file.name}…`, 'loading');
        const doc = await uploadDocument(file);
        setUploadedFiles(prev => [...prev, {
          id:        doc.id,
          filename:  doc.filename,
          file_type: doc.file_type,
          status:    'pending',
        }]);
        showToast(`📚 Indexing ${file.name}…`, 'loading');
      } catch (err) {
        showToast(`❌ ${err.message}`, 'error');
        setError(err.message);
      }
    }
    setIsUploading(false);
  }, []);

  const handleRemoveFile = useCallback((docId) => {
    setUploadedFiles(prev => prev.filter(f => f.id !== docId));
    deleteDocument(docId).catch(console.error);
  }, []);

  // ── Drag & drop ─────────────────────────────────────────────────────────────
  const handleDragOver  = useCallback((e) => { e.preventDefault(); setIsDragging(true); }, []);
  const handleDragLeave = useCallback((e) => {
    if (!inputAreaRef.current?.contains(e.relatedTarget)) setIsDragging(false);
  }, []);
  const handleDrop = useCallback((e) => {
    e.preventDefault(); setIsDragging(false);
    handleFileUpload(Array.from(e.dataTransfer.files));
  }, [handleFileUpload]);

  // ── Chat management ─────────────────────────────────────────────────────────
  const handleNewChat = useCallback(async () => {
    try {
      const chat = await createChat();
      setChats(prev => [chat, ...prev]);
      setCurrentChatId(chat.id);
      setMessages([]);
      setUploadedFiles([]);
      setError(null);
    } catch (err) { setError(err.message); }
  }, []);

  const handleSelectChat = useCallback(async (chatId) => {
    try {
      const chat = await getChat(chatId);
      setCurrentChatId(chat.id);
      // Normalize route values from persisted messages — old messages may have
      // "document" (wrong) instead of "documents" (correct)
      const ROUTE_FIX = { document: 'documents', Document: 'documents' };
      setMessages(chat.messages.map((m, i) => {
        const r = ROUTE_FIX[m.route] || m.route || m.sourceType || 'general';
        return { ...m, id: `m${i}`, route: r, sourceType: r };
      }));
      setUploadedFiles([]);
      setError(null);
    } catch (err) { setError(err.message); }
  }, []);

  const handleDeleteChat = useCallback(async (chatId, e) => {
    e.stopPropagation();
    try {
      await deleteChat(chatId);
      setChats(prev => prev.filter(c => c.id !== chatId));
      if (currentChatId === chatId) { setCurrentChatId(null); setMessages([]); }
    } catch (err) { setError(err.message); }
  }, [currentChatId]);

  // ── Submit ──────────────────────────────────────────────────────────────────
  const handleSubmit = useCallback(async (e) => {
    e.preventDefault();
    if (!question.trim() || loading) return;

    const userMsg = {
      id:        `m${Date.now()}-u`,
      role:      'user',
      content:   question,
      sources:   null,
      timestamp: new Date().toISOString(),
    };

    const assistantId = `m${Date.now()}-a`;
    const assistantMsg = {
      id:         assistantId,
      role:       'assistant',
      content:    '',           // starts empty, fills token by token
      sources:    [],
      sourceType: 'general',
      route:      'general',
      timestamp:  new Date().toISOString(),
    };

    setMessages(prev => [...prev, userMsg, assistantMsg]);
    setQuestion('');
    setLoading(true);
    setError(null);

    try {
      let chatId = currentChatId;
      if (!chatId) {
        const chat = await createChat();
        chatId = chat.id;
        setCurrentChatId(chatId);
        setChats(prev => [chat, ...prev]);
      }

      await askQuestionStream(question, chatId, {
        onRoute: (route) => {
          // Set badge immediately — this is authoritative
          setMessages(prev => prev.map(m =>
            m.id === assistantId ? { ...m, route, sourceType: route } : m
          ));
        },
        onToken: (token) => {
          setMessages(prev => prev.map(m =>
            m.id === assistantId ? { ...m, content: m.content + token } : m
          ));
        },
        onDone: ({ sources, source_type }) => {
          // source_type from backend is authoritative (set by ROUTE_MAP in api.py)
          // Do not read m.route here — it may be stale due to React batching
          setMessages(prev => prev.map(m => {
            if (m.id !== assistantId) return m;
            return { ...m, sources: sources || [], route: source_type, sourceType: source_type };
          }));
          // Sync chat title
          getChat(chatId).then(updated => {
            setChats(prev => prev.map(c =>
              c.id === chatId ? { ...c, title: updated.title } : c
            ));
          }).catch(() => {});
        },
        onError: (err) => {
          setError(err.message);
          setMessages(prev => prev.filter(m => m.id !== assistantId));
        },
      });

    } catch (err) {
      setError(err.message);
      setMessages(prev => prev.filter(m => m.id !== assistantId));
    } finally {
      setLoading(false);
    }
  }, [question, currentChatId, loading]);

  const handleToggleSources = useCallback((msgId) => {
    setExpandedSources(prev => ({ ...prev, [msgId]: !prev[msgId] }));
  }, []);

  // ── Keyboard shortcut ───────────────────────────────────────────────────────
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSubmit(e); }
  };

  const anyIndexing = uploadedFiles.some(f => f.status === 'pending' || f.status === 'indexing');

  // ── Render ──────────────────────────────────────────────────────────────────
  return (
    <div className="h-screen bg-slate-900 text-slate-100 flex overflow-hidden">

      {/* Mobile overlay */}
      {showSidebar && (
        <div className="lg:hidden fixed inset-0 bg-black/50 z-30" onClick={() => setShowSidebar(false)} />
      )}

      {/* Collapsed expand button */}
      {collapsed && (
        <button
          onClick={() => setCollapsed(false)}
          className="fixed left-0 top-1/2 -translate-y-1/2 z-50 w-9 h-16 bg-slate-800 hover:bg-slate-700 border border-r-0 border-slate-600 rounded-r-xl flex items-center justify-center text-slate-400 hover:text-white transition-colors"
        >
          <ChevronIcon />
        </button>
      )}

      {/* ── Sidebar ──────────────────────────────────────────────────────────── */}
      <aside className={`
        ${collapsed ? 'lg:w-16' : 'lg:w-64'}
        ${showSidebar ? 'translate-x-0' : '-translate-x-full'}
        fixed lg:relative z-40 h-screen bg-slate-800 border-r border-slate-700/60
        flex flex-col transition-all duration-300 ease-out w-64
      `}>

        {/* Sidebar header */}
        {!collapsed && (
          <div className="flex items-center justify-between px-4 py-3 border-b border-slate-700/60">
            <span className="text-sm font-semibold text-slate-300">Hybrid RAG</span>
            <button
              onClick={() => setCollapsed(true)}
              className="p-1.5 hover:bg-slate-700 rounded-lg text-slate-400 hover:text-white transition-colors"
            >
              <MenuIcon />
            </button>
          </div>
        )}

        {/* New Chat */}
        <div className={`${collapsed ? 'px-2 py-2' : 'p-3'} border-b border-slate-700/60`}>
          <button
            onClick={handleNewChat}
            className={`${collapsed ? 'w-full p-3 justify-center' : 'w-full px-4 py-2.5 gap-2'} flex items-center bg-indigo-600 hover:bg-indigo-500 rounded-xl font-medium text-sm transition-colors`}
          >
            <PlusIcon />
            {!collapsed && <span>New Chat</span>}
          </button>
        </div>

        {/* Chat list */}
        <div className="flex-1 overflow-y-auto py-2 space-y-0.5">
          {!collapsed && chats.length === 0 && (
            <div className="text-center text-slate-500 py-12 px-4">
              <p className="text-sm">No conversations yet</p>
              <p className="text-xs mt-1 opacity-70">Start a new chat</p>
            </div>
          )}

          {chats.map(chat => (
            <div key={chat.id} className="px-2">
              <button
                onClick={() => handleSelectChat(chat.id)}
                className={`w-full text-left px-3 py-2.5 rounded-xl transition-colors group flex items-center gap-2 ${
                  currentChatId === chat.id ? 'bg-slate-700' : 'hover:bg-slate-700/50'
                }`}
              >
                {collapsed ? (
                  <span title={chat.title}><ChatIcon /></span>
                ) : (
                  <>
                    <ChatIcon />
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium truncate">{chat.title}</p>
                      <p className="text-xs text-slate-500">{formatTimestamp(chat.created_at)}</p>
                    </div>
                    <button
                      onClick={(e) => handleDeleteChat(chat.id, e)}
                      className="opacity-0 group-hover:opacity-100 p-1 hover:bg-red-500/20 hover:text-red-400 rounded-lg transition-all"
                    >
                      <TrashIcon />
                    </button>
                  </>
                )}
              </button>
            </div>
          ))}
        </div>

        {/* Mobile close */}
        <button
          onClick={() => setShowSidebar(false)}
          className="lg:hidden absolute top-3 right-3 p-1.5 hover:bg-slate-700 rounded-lg text-slate-400 hover:text-white transition-colors"
        >
          <XIcon />
        </button>
      </aside>

      {/* ── Main area ──────────────────────────────────────────────────────────── */}
      <div className="flex-1 flex flex-col min-w-0 bg-slate-900">

        {/* Mobile header */}
        <header className="lg:hidden flex items-center px-4 py-3 bg-slate-800 border-b border-slate-700">
          <button onClick={() => setShowSidebar(s => !s)} className="p-2 hover:bg-slate-700 rounded-lg">
            {showSidebar ? <XIcon /> : <MenuIcon />}
          </button>
          <span className="font-semibold ml-3 text-sm">Hybrid RAG</span>
        </header>

        {/* Desktop header */}
        <header className="flex-shrink-0 hidden lg:flex items-center justify-between px-6 py-4 border-b border-slate-800">
          <div>
            <h1 className="text-lg font-bold">Hybrid RAG System</h1>
            <p className="text-xs text-slate-500 mt-0.5">
              {uploadedFiles.length > 0
                ? `${uploadedFiles.length} document(s) · ${anyIndexing ? '⏳ indexing…' : '✅ ready'}`
                : 'No documents — ask anything or upload files'}
            </p>
          </div>

          {/* Route legend */}
          <div className="hidden xl:flex items-center gap-2 text-xs">
            {Object.entries(ROUTE_CONFIG).map(([key, cfg]) => (
              <span key={key} className={`px-2 py-1 rounded-full ${cfg.cls}`}>{cfg.label}</span>
            ))}
          </div>
        </header>

        {/* ── Messages ───────────────────────────────────────────────────────── */}
        <main className="flex-1 min-h-0 overflow-y-auto px-4 lg:px-6 py-4 space-y-4">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <div className="w-14 h-14 rounded-2xl bg-indigo-500/10 flex items-center justify-center mb-4 border border-indigo-500/20">
                <ChatIcon />
              </div>
              <h2 className="text-xl font-semibold mb-2">Ask me anything</h2>
              <p className="text-slate-500 max-w-md text-sm leading-relaxed">
                Upload documents for RAG-powered answers, ask general knowledge questions, or get live web search results — all automatically routed.
              </p>
              <div className="flex flex-wrap gap-2 mt-5 justify-center">
                {Object.values(ROUTE_CONFIG).map(cfg => (
                  <span key={cfg.label} className={`text-xs px-3 py-1.5 rounded-full ${cfg.cls}`}>{cfg.label}</span>
                ))}
              </div>
            </div>
          ) : (
            messages.map(msg => (
              <MessageBubble
                key={msg.id}
                msg={msg}
                expandedSources={expandedSources}
                onToggleSources={handleToggleSources}
              />
            ))
          )}

          {/* Only show typing dots if loading and no streaming content yet */}
          {loading && messages[messages.length - 1]?.role === 'assistant' && messages[messages.length - 1]?.content === '' && (
            <TypingDots />
          )}
          <div ref={messagesEndRef} />
        </main>

        {/* ── Error banner ───────────────────────────────────────────────────── */}
        {error && (
          <div className="mx-4 lg:mx-6 mb-3 p-3 bg-red-500/10 border border-red-500/30 rounded-xl flex items-start gap-2">
            <span className="text-red-400 text-sm flex-1">{error}</span>
            <button onClick={() => setError(null)} className="text-red-400 hover:text-red-300 flex-shrink-0">
              <XIcon />
            </button>
          </div>
        )}

        {/* ── Input area ─────────────────────────────────────────────────────── */}
        <div
          ref={inputAreaRef}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={`flex-shrink-0 px-4 lg:px-6 pb-4 pt-3 border-t border-slate-800 transition-colors relative ${
            isDragging ? 'bg-indigo-900/20 border-indigo-500' : ''
          }`}
        >
          <Toast toast={toast} />

          {/* File chips */}
          {uploadedFiles.length > 0 && (
            <div className="flex flex-wrap gap-2 mb-3">
              {uploadedFiles.map(f => (
                <DocChip key={f.id} file={f} onRemove={handleRemoveFile} />
              ))}
            </div>
          )}

          {/* Drag overlay hint */}
          {isDragging && (
            <div className="absolute inset-0 flex items-center justify-center border-2 border-dashed border-indigo-500 rounded-xl bg-indigo-900/10 z-10 pointer-events-none">
              <p className="text-indigo-400 font-medium">Drop files to upload</p>
            </div>
          )}

          {/* Input row */}
          <div className="flex items-end gap-3">
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={isUploading}
              title="Attach files (PDF, DOCX, TXT)"
              className="p-3 bg-slate-800 hover:bg-slate-700 disabled:opacity-40 border border-slate-700 rounded-xl text-slate-400 hover:text-slate-200 transition-colors flex-shrink-0"
            >
              {isUploading ? <SpinnerIcon /> : <PaperclipIcon />}
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,.docx,.txt"
              multiple
              className="hidden"
              onChange={e => handleFileUpload(e.target.files)}
              disabled={isUploading}
            />

            <textarea
              value={question}
              onChange={e => setQuestion(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask anything — documents, general knowledge, or live web…"
              rows={1}
              disabled={loading}
              className="flex-1 px-4 py-3 bg-slate-800 border border-slate-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500 text-white placeholder-slate-500 text-sm resize-none leading-relaxed"
              style={{ minHeight: '48px', maxHeight: '160px' }}
              onInput={e => {
                e.target.style.height = 'auto';
                e.target.style.height = Math.min(e.target.scrollHeight, 160) + 'px';
              }}
            />

            <button
              onClick={handleSubmit}
              disabled={loading || !question.trim()}
              className="p-3 bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-700 disabled:text-slate-500 rounded-xl transition-colors flex-shrink-0"
            >
              {loading ? <SpinnerIcon /> : <SendIcon />}
            </button>
          </div>

          <p className="mt-2 text-xs text-slate-600 text-center">
            Queries are automatically routed · Documents → General AI → Web Search
          </p>
        </div>
      </div>
    </div>
  );
}