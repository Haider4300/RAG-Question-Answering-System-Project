import { useState, useEffect } from 'react';
import { checkHealth, askQuestion } from './lib/api';

function App() {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState(null);
  const [sources, setSources] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [apiStatus, setApiStatus] = useState('checking');

  useEffect(() => {
    checkHealth()
      .then(() => setApiStatus('ok'))
      .catch(() => setApiStatus('error'));
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!question.trim()) return;

    setLoading(true);
    setError(null);
    setAnswer(null);
    setSources([]);

    try {
      const result = await askQuestion(question);
      setAnswer(result.answer);
      setSources(result.sources || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setQuestion('');
    setAnswer(null);
    setSources([]);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100">
      <div className="max-w-4xl mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold mb-2">RAG Q&A Interface</h1>
          <p className="text-gray-400">Ask questions about Sarah Chen's CV</p>

          {/* API Status */}
          <div className="mt-3 flex items-center justify-center gap-2">
            <span
              className={`w-2 h-2 rounded-full ${
                apiStatus === 'ok'
                  ? 'bg-green-500'
                  : apiStatus === 'error'
                  ? 'bg-red-500'
                  : 'bg-yellow-500'
              }`}
            />
            <span className="text-sm text-gray-400">
              API {apiStatus === 'checking' ? 'checking...' : apiStatus}
            </span>
          </div>
        </div>

        {/* Query Form */}
        <form onSubmit={handleSubmit} className="mb-8">
          <div className="flex gap-3">
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder={apiStatus === 'ok' ? "Ask a question about the document..." : "Waiting for API..."}
              className="flex-1 px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-white placeholder-gray-500"
              disabled={loading}
            />
            <button
              type="submit"
              disabled={loading || !question.trim()}
              className="px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 disabled:text-gray-500 rounded-lg font-medium transition-colors"
            >
              {loading ? 'Thinking...' : 'Ask'}
            </button>
            {(answer || error) && (
              <button
                type="button"
                onClick={handleClear}
                className="px-6 py-3 bg-gray-700 hover:bg-gray-600 rounded-lg font-medium transition-colors"
              >
                Clear
              </button>
            )}
          </div>
        </form>

        {/* Error Message */}
        {error && (
          <div className="mb-6 p-4 bg-red-900/50 border border-red-700 rounded-lg">
            <p className="text-red-300">Error: {error}</p>
          </div>
        )}

        {/* Answer */}
        {answer && (
          <div className="mb-8">
            <h2 className="text-lg font-semibold text-green-400 mb-2">Answer</h2>
            <div className="p-4 bg-gray-800 border border-gray-700 rounded-lg">
              <p className="whitespace-pre-wrap">{answer}</p>
            </div>
          </div>
        )}

        {/* Sources */}
        {sources.length > 0 && (
          <div>
            <h2 className="text-lg font-semibold text-blue-400 mb-3">
              Source Documents ({sources.length})
            </h2>
            <div className="space-y-3">
              {sources.map((source, idx) => (
                <div
                  key={idx}
                  className="p-4 bg-gray-800 border border-gray-700 rounded-lg"
                >
                  <p className="text-sm text-gray-400 mb-1">
                    Source {idx + 1}
                  </p>
                  <p className="text-gray-300 whitespace-pre-wrap">
                    {source.content}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Loading State */}
        {loading && (
          <div className="flex items-center justify-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500" />
          </div>
        )}
      </div>
    </div>
  );
}

export default App;