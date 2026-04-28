import { useEffect, useState } from 'react';

export default function GroqInsightPanel({
  title = 'AI Analysis',
  insight,
  loading = false,
  error = null,
  onRegenerate = null,
  defaultExpanded = true
}) {
  const [expanded, setExpanded] = useState(defaultExpanded);
  const [typedText, setTypedText] = useState('');

  useEffect(() => {
    if (!expanded || loading) {
      return;
    }
    const fullText = insight || '';
    setTypedText('');
    if (!fullText) {
      return;
    }

    let idx = 0;
    const timer = setInterval(() => {
      idx += 1;
      setTypedText(fullText.slice(0, idx));
      if (idx >= fullText.length) {
        clearInterval(timer);
      }
    }, 20);

    return () => clearInterval(timer);
  }, [insight, loading, expanded]);

  return (
    <div className="card-industrial rounded-lg border border-gray-800 border-l-4 border-l-cyan/70 bg-gray-900/40 mb-8">
      <div className="px-5 py-3 border-b border-gray-800 flex items-center justify-between gap-3">
        <button
          type="button"
          onClick={() => setExpanded(prev => !prev)}
          className="flex items-center gap-2 text-left"
        >
          <span className="text-sm font-bold text-white uppercase tracking-widest">AI Analysis</span>
          <span className="text-[10px] text-gray-500 font-mono">{title}</span>
          <span className="text-[9px] text-gray-500 border border-gray-700 rounded px-1.5 py-0.5">Powered by Groq</span>
        </button>

        <div className="flex items-center gap-2">
          {onRegenerate && (
            <button
              type="button"
              onClick={onRegenerate}
              disabled={loading}
              className="text-gray-500 hover:text-cyan transition-colors disabled:opacity-50"
              title="Regenerate insight"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
            </button>
          )}
          <button
            type="button"
            onClick={() => setExpanded(prev => !prev)}
            className="text-gray-500 hover:text-cyan transition-colors"
            title={expanded ? 'Collapse' : 'Expand'}
          >
            <svg xmlns="http://www.w3.org/2000/svg" className={`h-4 w-4 transition-transform ${expanded ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>
        </div>
      </div>

      {expanded && (
        <div className="px-5 py-4">
          {loading && (
            <div className="space-y-2 animate-pulse">
              <div className="h-3 bg-gray-800 rounded w-full"></div>
              <div className="h-3 bg-gray-800 rounded w-11/12"></div>
              <div className="h-3 bg-gray-800 rounded w-4/5"></div>
            </div>
          )}

          {!loading && error && (
            <p className="text-[12px] text-amber-400 font-mono">{error}</p>
          )}

          {!loading && !error && (
            <p className="text-sm text-gray-300 leading-relaxed whitespace-pre-wrap">
              {typedText || insight || 'No AI insight available yet.'}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
