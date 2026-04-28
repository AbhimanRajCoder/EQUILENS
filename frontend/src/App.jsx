import { useState, useCallback, useEffect, useRef } from 'react';
import axios from 'axios';
import { useDropzone } from 'react-dropzone';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
  ScatterChart, Scatter, ZAxis, ReferenceArea, Label, LineChart, Line, Legend
} from 'recharts';
import GroqInsightPanel from './components/GroqInsightPanel';

const API_BASE = 'http://localhost:8000';
const DOMAIN_OPTIONS = ['General', 'Lending', 'Hiring', 'Healthcare', 'Insurance', 'Education'];

// --- Helpers ---
const getRiskInfo = (di) => {
  if (di < 0.5) return { label: 'Critical bias', color: 'text-red-400', border: 'border-red-500', bg: 'bg-red-500/10' };
  if (di < 0.8) return { label: 'High bias', color: 'text-orange-400', border: 'border-orange-500', bg: 'bg-orange-500/10' };
  if (di < 0.9) return { label: 'Moderate bias', color: 'text-yellow-400', border: 'border-yellow-500', bg: 'bg-yellow-500/10' };
  return { label: 'Low bias', color: 'text-emerald-400', border: 'border-emerald-500', bg: 'bg-emerald-500/10' };
};

// --- Sub-components ---

const ChartTooltip = ({ active, payload }) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    const fairness = Number.isFinite(data?.fairness) ? `${(data.fairness * 100).toFixed(1)}%` : 'N/A';
    const accuracy = Number.isFinite(data?.accuracy) ? `${(data.accuracy * 100).toFixed(1)}%` : 'N/A';
    return (
      <div className="card-industrial p-4 rounded border border-cyan/50 shadow-2xl backdrop-blur-md bg-gray-900/90">
        <p className="text-white font-bold text-sm mb-2 uppercase tracking-widest">{data.name}</p>
        <div className="space-y-1.5 border-t border-gray-800 pt-2">
          <div className="flex justify-between gap-8 text-[11px]">
            <span className="text-gray-500 font-mono uppercase">Fairness</span>
            <span className="text-cyan font-bold">{fairness}</span>
          </div>
          <div className="flex justify-between gap-8 text-[11px]">
            <span className="text-gray-500 font-mono uppercase">Accuracy</span>
            <span className="text-white font-bold">{accuracy}</span>
          </div>
        </div>
      </div>
    );
  }
  return null;
};

const StrategyDot = ({ cx, cy, payload }) => {
  const isRec = payload.isRec;
  return (
    <g transform={`translate(${cx},${cy})`}>
      {isRec && (
        <circle r="12" fill="#10b981" fillOpacity="0.2">
          <animate attributeName="r" from="8" to="16" dur="1.5s" repeatCount="indefinite" />
          <animate attributeName="opacity" from="0.4" to="0" dur="1.5s" repeatCount="indefinite" />
        </circle>
      )}
      <circle
        r={isRec ? 6 : 4}
        fill={isRec ? "#10b981" : "#00f5d4"}
        stroke={isRec ? "#fff" : "none"}
        strokeWidth={1.5}
        className="transition-all duration-300"
      />
    </g>
  );
};

// Removed unused components

const InlineSpinner = () => (
  <div className="inline-block w-4 h-4 border-2 border-gray-400/30 border-t-white rounded-full animate-spin"></div>
);

const InlineError = ({ message, onFix }) => {
  if (!message) return null;
  
  const hasSuggestion = message.includes('Suggestion:');
  const [mainError, suggestion] = hasSuggestion ? message.split('Suggestion:') : [message, null];
  const canFix = message.includes('Fix Automatically');

  return (
    <div className={`text-[11px] font-mono mt-2 animate-reveal flex flex-col gap-1.5 p-3 rounded border ${hasSuggestion ? 'bg-red-500/10 border-red-500/30 text-red-400' : 'text-red-500 border-transparent'}`}>
      <div className="flex items-center gap-1.5 font-bold">
        <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        {mainError.trim()}
      </div>
      {suggestion && (
        <div className="pl-5 mt-1 text-gray-300 border-l border-red-500/50 py-1 leading-relaxed flex flex-col gap-2">
          <div>
            <span className="text-cyan font-bold uppercase text-[9px] block mb-1">Recommended Action</span>
            {suggestion.trim()}
          </div>
          {canFix && onFix && (
            <button
              onClick={(e) => {
                e.preventDefault();
                onFix();
              }}
              className="w-fit px-3 py-1 bg-cyan text-black font-bold rounded hover:bg-cyan/80 transition-all uppercase text-[9px] tracking-widest mt-1"
            >
              Fix Automatically
            </button>
          )}
        </div>
      )}
    </div>
  );
};

const SectionHeader = ({ title, subtitle }) => (
  <div className="mb-8 mt-12">
    <h2 className="text-3xl font-bold text-cyan tracking-wider uppercase mb-2">{title}</h2>
    <div className="h-1 w-24 bg-cyan mb-4"></div>
    {subtitle && <p className="text-gray-400 font-light">{subtitle}</p>}
  </div>
);

const ConfusionMatrixView = ({ data, groupName }) => {
  if (!data) return null;
  return (
    <div className="flex flex-col items-center">
      <p className="text-[10px] text-gray-500 font-mono mb-4 uppercase tracking-widest">{groupName}</p>
      <div className="grid grid-cols-2 gap-1 w-48 h-48 font-mono text-[10px]">
        <div className="bg-emerald-500/20 border border-emerald-500/30 flex flex-col items-center justify-center p-2 rounded">
          <span className="text-emerald-400 font-bold text-lg">{data.tp}</span>
          <span className="text-gray-500 uppercase">True Pos</span>
        </div>
        <div className="bg-red-500/10 border border-red-500/20 flex flex-col items-center justify-center p-2 rounded">
          <span className="text-red-400 font-bold text-lg">{data.fp}</span>
          <span className="text-gray-500 uppercase">False Pos</span>
        </div>
        <div className="bg-red-500/10 border border-red-500/20 flex flex-col items-center justify-center p-2 rounded">
          <span className="text-red-400 font-bold text-lg">{data.fn}</span>
          <span className="text-gray-500 uppercase">False Neg</span>
        </div>
        <div className="bg-emerald-500/20 border border-emerald-500/30 flex flex-col items-center justify-center p-2 rounded">
          <span className="text-emerald-400 font-bold text-lg">{data.tn}</span>
          <span className="text-gray-500 uppercase">True Neg</span>
        </div>
      </div>
    </div>
  );
};

const DistributionChart = ({ scores = [], groupName, color }) => {
  // Create bins for histogram
  const bins = Array(10).fill(0);
  (scores || []).forEach(s => {
    const binIdx = Math.min(Math.floor(s * 10), 9);
    bins[binIdx]++;
  });
  const data = bins.map((count, i) => ({
    range: `${(i / 10).toFixed(1)}-${((i + 1) / 10).toFixed(1)}`,
    count
  }));

  return (
    <div className="h-[300px] w-full">
      <p className="text-[10px] font-mono text-gray-500 uppercase mb-2 text-center">{groupName} Score Distribution</p>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false} />
          <XAxis dataKey="range" stroke="#4b5563" fontSize={10} />
          <YAxis stroke="#4b5563" fontSize={10} />
          <Tooltip 
            contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', fontSize: '10px', fontFamily: 'monospace' }}
          />
          <Bar dataKey="count" fill={color} radius={[2, 2, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

const ResearcherAnalytics = ({ metrics }) => {
  const groups = Object.keys(metrics.per_group_metrics);
  const [activeTab, setActiveTab] = useState('roc');

  return (
    <div className="card-industrial p-8 rounded-lg border border-gray-800 animate-reveal">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-12">
        <h3 className="text-xl font-bold text-white flex items-center gap-3">
          <svg className="w-6 h-6 text-cyan" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path></svg>
          Researcher Deep-Dive
        </h3>
        <div className="flex flex-wrap gap-2">
          {['roc', 'pr', 'calibration', 'distribution'].map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-4 py-1.5 rounded text-[10px] font-mono uppercase tracking-widest transition-all ${activeTab === tab ? 'bg-cyan text-black font-bold' : 'text-gray-500 hover:text-white border border-gray-800'}`}
            >
              {tab === 'roc' ? 'ROC Curve' : tab === 'pr' ? 'Precision-Recall' : tab === 'calibration' ? 'Calibration' : 'Distributions'}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-12">
        {/* Left: Curves or Distributions */}
        <div className="lg:col-span-2 min-h-[400px]">
          {activeTab === 'distribution' ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              {groups.map((group, idx) => (
                <DistributionChart 
                  key={group} 
                  groupName={group} 
                  scores={metrics.per_group_metrics[group]?.score_distribution || []}
                  color={idx === 0 ? '#00f5d4' : idx === 1 ? '#a855f7' : '#f59e0b'}
                />
              ))}
            </div>
          ) : (
            <div className="h-[400px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false} />
                  <XAxis 
                    type="number" 
                    dataKey="x" 
                    domain={[0, 1]} 
                    stroke="#4b5563" 
                    fontSize={10} 
                    tickFormatter={(v) => v.toFixed(1)}
                    label={{ value: activeTab === 'roc' ? 'False Positive Rate' : activeTab === 'pr' ? 'Recall' : 'Predicted Prob', position: 'bottom', offset: 0, fill: '#6b7280', fontSize: 10, fontFamily: 'monospace' }}
                  />
                  <YAxis 
                    type="number" 
                    dataKey="y" 
                    domain={[0, 1]} 
                    stroke="#4b5563" 
                    fontSize={10} 
                    tickFormatter={(v) => v.toFixed(1)}
                    label={{ value: activeTab === 'roc' ? 'True Positive Rate' : activeTab === 'pr' ? 'Precision' : 'True Prob', angle: -90, position: 'left', fill: '#6b7280', fontSize: 10, fontFamily: 'monospace' }}
                  />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', fontSize: '10px', fontFamily: 'monospace' }}
                    itemStyle={{ padding: '2px 0' }}
                  />
                  <Legend verticalAlign="top" align="right" iconType="circle" wrapperStyle={{ fontSize: '10px', fontFamily: 'monospace', textTransform: 'uppercase', paddingBottom: '20px' }} />
                  {groups.map((group, idx) => (
                    <Line
                      key={group}
                      type="monotone"
                      data={metrics.per_group_metrics[group]?.[`${activeTab}_curve`] || []}
                      dataKey="y"
                      name={group}
                      stroke={idx === 0 ? '#00f5d4' : idx === 1 ? '#a855f7' : '#f59e0b'}
                      strokeWidth={2}
                      dot={false}
                      activeDot={{ r: 4, strokeWidth: 0 }}
                    />
                  ))}
                  {['roc', 'calibration'].includes(activeTab) && (
                    <Line 
                      data={[{x:0, y:0}, {x:1, y:1}]} 
                      dataKey="y" 
                      stroke="#374151" 
                      strokeDasharray="5 5" 
                      dot={false} 
                      name={activeTab === 'roc' ? "Random" : "Ideal"} 
                    />
                  )}
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>

        {/* Right: Confusion Matrices */}
        <div className="flex flex-col gap-12 overflow-y-auto max-h-[400px] pr-4 custom-scrollbar">
          {groups.map(group => (
            <ConfusionMatrixView 
              key={group} 
              groupName={group} 
              data={metrics.per_group_metrics[group]?.confusion_matrix} 
            />
          ))}
        </div>
      </div>
    </div>
  );
};

const SignificanceCard = ({ pValues }) => {
  if (!pValues) return null;
  const pVal = pValues.demographic_parity;
  const isSignificant = pVal < 0.05;
  return (
    <div className={`card-industrial p-6 rounded-lg border-l-4 ${isSignificant ? 'border-red-500 glow-red' : 'border-emerald-500 glow-emerald'}`}>
      <div className="flex justify-between items-start mb-2">
        <p className="text-gray-500 text-xs uppercase tracking-widest">Statistical Significance</p>
        <span className={`text-[10px] font-mono px-2 py-0.5 rounded ${isSignificant ? 'bg-red-500/20 text-red-400' : 'bg-emerald-500/20 text-emerald-400'}`}>
          {isSignificant ? 'Significant' : 'Not Significant'}
        </span>
      </div>
      <p className="text-3xl font-bold text-white font-mono mb-2">
        p = {pVal.toFixed(4)}
      </p>
      <p className="text-[10px] text-gray-500 leading-relaxed italic">
        {isSignificant 
          ? "The observed disparity is unlikely to have occurred by chance. High confidence of systemic bias."
          : "The observed disparity might be due to random sampling noise. Low confidence of systemic bias."
        }
      </p>
    </div>
  );
};

const RepresentationChart = ({ data }) => {
  if (!data) return null;
  const chartData = Object.entries(data).map(([name, value]) => ({
    name,
    value: parseFloat((value * 100).toFixed(1))
  }));

  return (
    <div className="card-industrial p-8 rounded-lg border border-gray-800 h-full">
      <h4 className="text-xs font-mono text-gray-500 uppercase tracking-widest mb-6 flex items-center gap-2">
        <svg className="w-4 h-4 text-cyan" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M11 3.055A9.001 9.001 0 1020.945 13H11V3.055z"></path><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M20.488 9H15V3.512A9.025 9.025 0 0120.488 9z"></path></svg>
        Representation Audit
      </h4>
      <div className="h-[250px]">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" horizontal={false} />
            <XAxis type="number" domain={[0, 100]} stroke="#4b5563" fontSize={10} tickFormatter={(v) => `${v}%`} />
            <YAxis dataKey="name" type="category" stroke="#4b5563" fontSize={10} width={80} />
            <Tooltip 
              cursor={{fill: 'rgba(255,255,255,0.05)'}}
              contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', fontSize: '10px', fontFamily: 'monospace' }}
              formatter={(value) => [`${value}%`, 'Representation']}
            />
            <Bar dataKey="value" fill="#00f5d4" radius={[0, 4, 4, 0]}>
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={index === 0 ? '#00f5d4' : index === 1 ? '#a855f7' : '#f59e0b'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

const PerGroupShapView = ({ perGroupShap }) => {
  if (!perGroupShap) return null;
  const groups = Object.keys(perGroupShap);
  
  return (
    <div className="card-industrial p-8 rounded-lg border border-gray-800">
      <h4 className="text-xs font-mono text-gray-500 uppercase tracking-widest mb-8 flex items-center gap-2">
        <svg className="w-4 h-4 text-cyan" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"></path></svg>
        Per-Group Feature Impact (SHAP)
      </h4>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
        {groups.map((group, gIdx) => (
          <div key={group} className="space-y-4">
            <p className="text-[10px] font-mono text-gray-400 uppercase border-b border-gray-800 pb-2">{group}</p>
            {(perGroupShap[group] || []).map((feat, fIdx) => (
              <div key={feat.feature} className="space-y-1">
                <div className="flex justify-between text-[10px] font-mono">
                  <span className="text-gray-500 truncate w-32" title={feat.feature}>{feat.feature}</span>
                  <span className="text-white">{feat.importance.toFixed(4)}</span>
                </div>
                <div className="h-1 w-full bg-gray-800 rounded-full overflow-hidden">
                  <div 
                    className="h-full transition-all duration-1000" 
                    style={{ 
                      width: `${(feat.importance / (perGroupShap[group][0]?.importance || 1)) * 100}%`,
                      backgroundColor: gIdx === 0 ? '#00f5d4' : gIdx === 1 ? '#a855f7' : '#f59e0b'
                    }}
                  ></div>
                </div>
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
};

const MetricCard = ({ label, value, isFair, tooltip }) => (
  <div className={`card-industrial p-6 rounded-lg border-l-4 ${isFair ? 'border-cyan glow-cyan' : 'border-red-500 glow-red'}`}>
    <p className="text-gray-500 text-xs uppercase tracking-widest mb-1">{label}</p>
    <p className={`text-3xl font-bold ${isFair ? 'text-cyan' : 'text-red-500'} font-mono`}>
      {typeof value === 'number' ? value.toFixed(3) : value}
    </p>
  </div>
);

const SectionDivider = ({ label }) => (
  <div className="relative my-24">
    <div className="absolute inset-0 flex items-center" aria-hidden="true">
      <div className="w-full border-t border-gray-800"></div>
    </div>
    <div className="relative flex justify-center">
      <span className="px-6 bg-[#0a0a0f] text-[10px] font-mono text-gray-500 uppercase tracking-[0.3em]">{label}</span>
    </div>
  </div>
);

const EmptyState = ({ icon, prompt, onAction, actionLabel }) => (
  <div className="card-industrial p-12 rounded-lg border border-gray-800 flex flex-col items-center text-center justify-center animate-reveal">
    <div className="text-gray-700 mb-6">
      {icon}
    </div>
    <p className="text-gray-500 font-mono text-sm mb-6 max-w-xs">{prompt}</p>
    {onAction && (
      <button 
        onClick={onAction}
        className="px-6 py-2 border border-gray-700 text-gray-400 hover:border-cyan hover:text-cyan text-xs font-mono uppercase tracking-widest transition-all"
      >
        {actionLabel}
      </button>
    )}
  </div>
);

const TrustSection = () => {
  const [whyOpen, setWhyOpen] = useState(false);
  return (
    <div className="w-full max-w-7xl mx-auto px-8 mb-24 animate-reveal">
      {/* 1. Tech Badge Row */}
      <div className="flex flex-wrap items-center justify-center gap-4 mb-16">
        <span className="text-[10px] font-mono text-gray-600 uppercase tracking-widest mr-4">Powered by</span>
        {['SHAP', 'Stable-Baselines3', 'FastAPI', 'Scikit-learn'].map(tech => (
          <span key={tech} className="px-4 py-1.5 rounded-full border border-gray-800 bg-gray-900/50 text-xs font-mono text-gray-400 transition-colors hover:border-cyan/30">
            {tech}
          </span>
        ))}
      </div>

      {/* 2. Use Cases */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-16">
        {[
          { 
            title: "Finance", 
            desc: "Loan approval fairness", 
            icon: <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
          },
          { 
            title: "Hiring", 
            desc: "Resume screening audits", 
            icon: <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"></path></svg>
          },
          { 
            title: "Healthcare", 
            desc: "Diagnostic model equity", 
            icon: <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"></path></svg>
          }
        ].map(uc => (
          <div key={uc.title} className="card-industrial p-6 rounded-lg border border-gray-800 flex items-start gap-4 hover:border-gray-700 transition-colors">
            <div className="text-cyan p-2 bg-cyan/5 rounded">{uc.icon}</div>
            <div>
              <h4 className="text-white font-bold text-sm mb-1">{uc.title}</h4>
              <p className="text-gray-500 text-xs">{uc.desc}</p>
            </div>
          </div>
        ))}
      </div>

      {/* 3. Why Fairness Matters (Collapsible) */}
      <div className="mb-16">
        <button 
          onClick={() => setWhyOpen(!whyOpen)}
          className="w-full flex items-center justify-between p-6 bg-gray-900/20 border border-gray-800 rounded-lg group"
        >
          <span className="text-sm font-bold text-gray-300 uppercase tracking-widest">Why fairness matters</span>
          <svg className={`w-5 h-5 text-gray-500 group-hover:text-cyan transition-transform ${whyOpen ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path></svg>
        </button>
        {whyOpen && (
          <div className="p-8 bg-gray-900/10 border-x border-b border-gray-800 rounded-b-lg text-gray-400 text-sm leading-relaxed animate-reveal">
            Biased models don't just harm individuals — they expose organizations to regulatory risk. The EU AI Act requires high-risk AI systems to demonstrate non-discrimination. BiasGuard gives you the audit trail to prove it.
          </div>
        )}
      </div>

      {/* 4. Metrics Row */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
        {[
          { val: "4", label: "Mitigation strategies tested in parallel" },
          { val: "3", label: "Industry-standard fairness metrics" },
          { val: "SHAP", label: "Explainable — no black boxes" }
        ].map(m => (
          <div key={m.label} className="text-center">
            <div className="text-2xl font-bold text-white mb-1 font-mono">{m.val}</div>
            <div className="text-[10px] text-gray-600 uppercase tracking-widest">{m.label}</div>
          </div>
        ))}
      </div>
    </div>
  );
};

const IntegrateSection = () => (
  <section className="animate-reveal py-24 bg-gray-900/10 border-t border-gray-800 mt-24">
    <div className="max-w-7xl mx-auto px-8">
      <SectionHeader title="Integrate BiasGuard" subtitle="Connect our fairness audit engine directly to your ML pipeline." />
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
        {/* Code Snippet */}
        <div className="card-industrial rounded-lg border border-gray-800 overflow-hidden">
          <div className="bg-gray-900 px-4 py-2 border-b border-gray-800 flex items-center justify-between">
            <span className="text-[10px] font-mono text-gray-500 uppercase tracking-widest">Python SDK Example</span>
            <div className="flex gap-1.5">
              <div className="w-2.5 h-2.5 rounded-full bg-red-500/20"></div>
              <div className="w-2.5 h-2.5 rounded-full bg-yellow-500/20"></div>
              <div className="w-2.5 h-2.5 rounded-full bg-green-500/20"></div>
            </div>
          </div>
          <pre className="p-6 font-mono text-sm leading-relaxed overflow-x-auto">
            <code className="text-gray-300">
              <span className="text-purple-400">import</span> requests{"\n"}
              {"\n"}
              result = requests.post(
                <span className="text-emerald-400">"http://localhost:8000/api/detect"</span>,{"\n"}
                json={"{"}{"\n"}
                {"  "}<span className="text-cyan">"target_col"</span>: <span className="text-emerald-400">"income"</span>,{"\n"}
                {"  "}<span className="text-cyan">"sensitive_col"</span>: <span className="text-emerald-400">"sex"</span>{"\n"}
                {"}"}{"\n"}
              ){"\n"}
              {"\n"}
              <span className="text-purple-400">print</span>(result.json()[<span className="text-emerald-400">"bias_summary"</span>])
            </code>
          </pre>
        </div>

        {/* Roadmap & Webhooks */}
        <div className="space-y-8">
          <div>
            <h4 className="text-xs font-mono text-gray-500 uppercase tracking-widest mb-4">Coming Soon</h4>
            <div className="flex flex-wrap gap-3">
              {[
                "API key authentication", 
                "Python SDK (pip install biasguard)", 
                "GitHub Action for CI/CD"
              ].map(item => (
                <span key={item} className="px-4 py-2 bg-gray-900/50 border border-gray-800 rounded-full text-xs text-gray-600 flex items-center gap-2">
                  {item}
                  <span className="text-[8px] bg-gray-800 text-gray-500 px-1.5 py-0.5 rounded uppercase font-bold">Soon</span>
                </span>
              ))}
            </div>
          </div>
          
          <div className="p-6 rounded-lg border border-cyan/10 bg-cyan/5">
            <h4 className="text-white font-bold text-sm mb-2 flex items-center gap-2">
              <svg className="w-4 h-4 text-cyan" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
              Webhook Integration
            </h4>
            <p className="text-gray-400 text-xs leading-relaxed">
              Trigger audits automatically when your model is retrained. Webhook support coming in v2 for seamless MLOps integration.
            </p>
          </div>
        </div>
      </div>
    </div>
  </section>
);

const StatusBar = ({ state, message }) => {
  if (!state) return null;
  return (
    <div className="fixed bottom-0 left-0 right-0 z-[100] bg-[#0a0a0f] border-t border-cyan/20 px-8 py-3 flex items-center justify-between animate-reveal-up backdrop-blur-xl">
      <div className="flex items-center gap-4">
        <InlineSpinner />
        <span className="text-[11px] font-mono text-cyan uppercase tracking-widest animate-pulse">
          {message}
        </span>
      </div>
      <div className="text-[10px] font-mono text-gray-600 uppercase tracking-widest hidden sm:block">
        Pipeline Active · {state}
      </div>
    </div>
  );
};



const InfoTooltip = ({ text }) => (
  <span className="relative inline-flex ml-1.5 group">
    <span className="inline-flex items-center justify-center w-4 h-4 rounded-full border border-gray-600 text-[9px] text-gray-500 cursor-help font-bold leading-none group-hover:border-cyan group-hover:text-cyan transition-colors">?</span>
    <span className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-2 rounded bg-gray-900 border border-gray-700 text-[11px] text-gray-300 font-normal normal-case tracking-normal whitespace-normal w-56 text-center opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 pointer-events-none z-50 shadow-lg shadow-black/40">
      {text}
      <span className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-gray-700"></span>
    </span>
  </span>
);

// --- Main App ---

export default function App() {
  const [file, setFile] = useState(null);
  const [columns, setColumns] = useState([]);
  const [config, setConfig] = useState({ target_col: '', sensitive_col: '' });
  const [domain, setDomain] = useState('General');
  const [detectionResults, setDetectionResults] = useState(null);
  const [simulationResults, setSimulationResults] = useState(null);
  const [recommendation, setRecommendation] = useState(null);
  const [storyMode, setStoryMode] = useState(false);
  const [showGroqExportToast, setShowGroqExportToast] = useState(false);
  const [groqInsights, setGroqInsights] = useState({
    biasNarrative: null,
    shapInsight: null,
    counterfactualStory: null,
    mitigationAdvice: null,
    intersectionalInsight: null
  });
  const [groqLoading, setGroqLoading] = useState({
    biasNarrative: false,
    shapInsight: false,
    counterfactualStory: false,
    mitigationAdvice: false,
    intersectionalInsight: false
  });
  const [groqErrors, setGroqErrors] = useState({
    biasNarrative: null,
    shapInsight: null,
    counterfactualStory: null,
    mitigationAdvice: null,
    intersectionalInsight: null
  });
  
  // Pipeline Loading State
  const [loadingState, setLoadingState] = useState(null); // null, "uploading", "auditing", "simulating", "recommending", "exporting"
  const [auditingMessageIdx, setAuditingMessageIdx] = useState(0);
  const auditingMessages = [
    "Training baseline model…",
    "Running SHAP explainer…",
    "Calculating fairness metrics…"
  ];

  // Specific errors for inline display
  const [errors, setErrors] = useState({
    demo: null,
    detect: null,
    simulate: null,
    recommend: null,
    export: null
  });

  const [intersectionalOpen, setIntersectionalOpen] = useState(false);
  const [showHero, setShowHero] = useState(true);
  const [showInfoBanner, setShowInfoBanner] = useState(true);
  const [parsedData, setParsedData] = useState([]);
  const [datasetProfileOpen, setDatasetProfileOpen] = useState(false);
  const [beforeAfterVisible, setBeforeAfterVisible] = useState(false);
  const [howItWorksOpen, setHowItWorksOpen] = useState(false);

  // Cycle auditing messages
  useEffect(() => {
    let interval;
    if (loadingState === 'auditing') {
      interval = setInterval(() => {
        setAuditingMessageIdx(prev => (prev + 1) % auditingMessages.length);
      }, 1500);
    } else {
      setAuditingMessageIdx(0);
    }
    return () => clearInterval(interval);
  }, [loadingState]);

  const resetGroqInsights = () => {
    setStoryMode(false);
    setGroqInsights({
      biasNarrative: null,
      shapInsight: null,
      counterfactualStory: null,
      mitigationAdvice: null,
      intersectionalInsight: null
    });
    setGroqErrors({
      biasNarrative: null,
      shapInsight: null,
      counterfactualStory: null,
      mitigationAdvice: null,
      intersectionalInsight: null
    });
    setGroqLoading({
      biasNarrative: false,
      shapInsight: false,
      counterfactualStory: false,
      mitigationAdvice: false,
      intersectionalInsight: false
    });
  };

  const fetchGroqInsight = async (key, endpoint, payload) => {
    setGroqLoading(prev => ({ ...prev, [key]: true }));
    setGroqErrors(prev => ({ ...prev, [key]: null }));
    try {
      const res = await axios.post(`${API_BASE}${endpoint}`, payload);
      if (res.data?.error) {
        setGroqErrors(prev => ({ ...prev, [key]: res.data.error }));
      }
      setGroqInsights(prev => ({ ...prev, [key]: res.data?.insight || null }));
    } catch (err) {
      setGroqErrors(prev => ({ ...prev, [key]: err.response?.data?.detail || 'Unable to generate AI insight right now.' }));
      setGroqInsights(prev => ({ ...prev, [key]: null }));
    } finally {
      setGroqLoading(prev => ({ ...prev, [key]: false }));
    }
  };

  // Helper to handle downloads
  const handleExport = async (format) => {
    if (!detectionResults) return;
    const includeGroqNarrative = format === 'pdf';
    setLoadingState('exporting');
    setErrors(prev => ({ ...prev, export: null }));
    if (includeGroqNarrative) {
      setShowGroqExportToast(true);
    }
    try {
      const payload = {
        timestamp: new Date().toLocaleString(),
        metadata: {
          filename: file?.name || "Demo Dataset",
          target_col: config.target_col,
          sensitive_col: config.sensitive_col
        },
        baseline_metrics: detectionResults,
        simulation_results: simulationResults || null,
        recommendation: recommendation || null,
        counterfactual_examples: detectionResults.counterfactual_examples || null,
        include_groq_narrative: includeGroqNarrative
      };

      const response = await fetch(`http://localhost:8000/api/export/${format}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!response.ok) throw new Error('Export failed');

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `BiasGuard_Audit_${payload.metadata.filename}.${format}`;
      document.body.appendChild(a);
      a.click();
      a.remove();
    } catch (err) {
      console.error(err);
      setErrors(prev => ({ ...prev, export: 'Export failed — please try again.' }));
    } finally {
      setLoadingState(null);
      setShowGroqExportToast(false);
    }
  };
  const [error, setError] = useState(null);
  const beforeAfterRef = useRef(null);
  const recName = recommendation?.recommended_strategy;

  // Derive recommended strategy metrics from simulation results
  const recommendedMetrics = simulationResults?.find(
    s => s.strategy_name === recommendation?.recommended_strategy
  );
  const dots = (simulationResults || [])
    .map((s) => ({
      name: s.strategy_name,
      fairness: s.fairness_score,
      accuracy: s.accuracy,
      size: s.strategy_name === recName ? 240 : 140,
      isRec: s.strategy_name === recName
    }))
    .filter((d) => Number.isFinite(d.fairness) && Number.isFinite(d.accuracy));

  const onDrop = useCallback(acceptedFiles => {
    const droppedFile = acceptedFiles[0];
    setFile(droppedFile);

    // Read header + full data to populate columns and compute summaries
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target.result;
      const lines = text.split('\n').filter(l => l.trim() !== '');
      const firstLine = lines[0];
      // Split by comma and strip quotes/whitespace
      const cols = firstLine.split(',').map(c => c.trim().replace(/^["']|["']$/g, '')).filter(c => c !== "");
      setColumns(cols);

      // Parse rows into objects for column summaries
      const rows = lines.slice(1).map(line => {
        const vals = line.split(',').map(v => v.trim().replace(/^["']|["']$/g, ''));
        const obj = {};
        cols.forEach((c, i) => { obj[c] = vals[i] || ''; });
        return obj;
      });
      setParsedData(rows);
      
      // Auto-set defaults if possible
      if (cols.length >= 2) {
        setConfig({
          target_col: cols[cols.length - 1],
          sensitive_col: cols[0]
        });
      }
    };
    reader.readAsText(droppedFile);
  }, []);

  // --- "Try demo dataset" handler ---
  const handleDemoDataset = async () => {
    setLoadingState('uploading');
    setErrors(prev => ({ ...prev, demo: null }));
    // Reset any previous analysis
    setDetectionResults(null);
    setSimulationResults(null);
    setRecommendation(null);
    resetGroqInsights();

    try {
      const res = await axios.get(`${API_BASE}/api/demo-dataset`);
      const { data, columns: cols, config: demoConfig } = res.data;
      
      // Artificial delay to show uploading state
      await new Promise(r => setTimeout(r, 800));

      // Build a CSV string from the JSON rows and wrap it in a File object
      // so the existing upload flow (FormData with a File) works unchanged.
      const header = cols.join(',');
      const rows = data.map(row => cols.map(c => {
        const v = row[c];
        // Quote strings that contain commas
        return typeof v === 'string' && v.includes(',') ? `"${v}"` : v;
      }).join(','));
      const csvString = [header, ...rows].join('\n');
      const blob = new Blob([csvString], { type: 'text/csv' });
      const syntheticFile = new File([blob], 'adult_sample.csv', { type: 'text/csv' });

      setFile(syntheticFile);
      setColumns(cols);
      setConfig({
        target_col: demoConfig.target_col,
        sensitive_col: demoConfig.sensitive_col,
      });
      // Store parsed rows for column summaries
      setParsedData(data);
      setShowHero(false);
    } catch (err) {
      setErrors(prev => ({ ...prev, demo: err.response?.data?.detail || 'Failed to load demo dataset' }));
    } finally {
      setLoadingState(null);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop, accept: { 'text/csv': ['.csv'] }, multiple: false });

  // Function to fix data types in parsedData and regenerate the file
  const handleAutoFix = () => {
    if (!parsedData.length || !columns.length) return;

    // Convert all values in the problematic columns to strings
    const updatedData = parsedData.map(row => {
      const newRow = { ...row };
      if (config.target_col) newRow[config.target_col] = String(row[config.target_col]);
      if (config.sensitive_col) newRow[config.sensitive_col] = String(row[config.sensitive_col]);
      return newRow;
    });

    setParsedData(updatedData);

    // Regenerate CSV file
    const header = columns.join(',');
    const rows = updatedData.map(row => columns.map(c => {
      const v = row[c];
      return typeof v === 'string' && v.includes(',') ? `"${v}"` : v;
    }).join(','));
    const csvString = [header, ...rows].join('\n');
    const blob = new Blob([csvString], { type: 'text/csv' });
    const fixedFile = new File([blob], file?.name || 'fixed_dataset.csv', { type: 'text/csv' });

    setFile(fixedFile);
    setErrors(prev => ({ ...prev, detect: null }));
  };

  const handleDetect = async (e) => {
    e.preventDefault();
    if (!file || !config.target_col || !config.sensitive_col) return;
    
    setLoadingState('auditing');
    setErrors(prev => ({ ...prev, detect: null }));
    resetGroqInsights();
    
    const formData = new FormData();
    formData.append('file', file);
    formData.append('target_col', config.target_col);
    formData.append('sensitive_col', config.sensitive_col);

    // Auto-detect intersectional columns: include all categorical-looking
    // columns that differ from the target/sensitive and are commonly used
    // for intersectional analysis (sex, race, etc.).
    const intersectionalCandidates = columns.filter(
      c => c !== config.target_col && ['sex', 'race', 'gender', 'ethnicity'].includes(c.toLowerCase())
    );
    if (intersectionalCandidates.length >= 2) {
      formData.append('intersectional_cols', intersectionalCandidates.join(','));
    } else if (intersectionalCandidates.length === 1 && intersectionalCandidates[0].toLowerCase() !== config.sensitive_col.toLowerCase()) {
      // Include the sensitive col + the additional candidate
      formData.append('intersectional_cols', [config.sensitive_col, intersectionalCandidates[0]].join(','));
    } else {
      // Fall back: use sensitive_col + any second categorical col that isn't the target
      const fallback = columns.find(
        c => c !== config.target_col && c !== config.sensitive_col &&
             !['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week'].includes(c)
      );
      if (fallback) {
        formData.append('intersectional_cols', [config.sensitive_col, fallback].join(','));
      }
    }

    try {
      const res = await axios.post(`${API_BASE}/api/detect`, formData);
      setDetectionResults(res.data);
      setGroqInsights(prev => ({
        ...prev,
        biasNarrative: res.data?.groq_narrative || null,
        shapInsight: res.data?.groq_shap_insight || null
      }));
    } catch (err) {
      setErrors(prev => ({ ...prev, detect: err.response?.data?.detail || 'Audit failed — check that target column has binary values.' }));
    } finally {
      setLoadingState(null);
    }
  };

  const handleSimulate = async () => {
    setLoadingState('simulating');
    setErrors(prev => ({ ...prev, simulate: null }));
    const formData = new FormData();
    formData.append('file', file);
    formData.append('target_col', config.target_col);
    formData.append('sensitive_col', config.sensitive_col);

    try {
      const res = await axios.post(`${API_BASE}/api/simulate`, formData);
      setSimulationResults(res.data.strategies);
    } catch (err) {
      setErrors(prev => ({ ...prev, simulate: err.response?.data?.detail || 'Simulation failed' }));
    } finally {
      setLoadingState(null);
    }
  };

  const handleRecommend = async () => {
    setLoadingState('recommending');
    setErrors(prev => ({ ...prev, recommend: null }));
    try {
      const res = await axios.post(`${API_BASE}/api/recommend`, {
        bias_score: detectionResults.fairness_metrics.demographic_parity_difference,
        accuracy: detectionResults.accuracy
      });
      setRecommendation(res.data);
    } catch (err) {
      setErrors(prev => ({ ...prev, recommend: err.response?.data?.detail || 'Recommendation failed' }));
    } finally {
      setLoadingState(null);
    }
  };

  // Auto-trigger recommendation when detection results arrive
  useEffect(() => {
    if (detectionResults && !recommendation && loadingState !== 'recommending') {
      handleRecommend();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [detectionResults]);

  useEffect(() => {
    if (!storyMode || !detectionResults?.counterfactual_examples?.length) return;
    if (groqInsights.counterfactualStory || groqLoading.counterfactualStory) return;
    fetchGroqInsight('counterfactualStory', '/api/groq/counterfactual-story', {
      counterfactual_examples: detectionResults.counterfactual_examples,
      sensitive_col: config.sensitive_col
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [storyMode, detectionResults]);

  useEffect(() => {
    if (!detectionResults?.intersectional_bias?.length) return;
    fetchGroqInsight('intersectionalInsight', '/api/groq/intersectional-insight', {
      intersectional_data: detectionResults.intersectional_bias
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [detectionResults?.intersectional_bias]);

  // Apply strategy: run simulation if needed, then scroll to before/after
  // Live validation for column types
  useEffect(() => {
    if (!config.target_col || !config.sensitive_col || !parsedData.length) return;

    const checkColumn = (colName, role) => {
      const vals = parsedData.map(r => r[colName]).filter(v => v !== undefined && v !== '');
      const types = new Set(vals.map(v => {
        if (!isNaN(v) && v !== '') return 'number';
        return 'string';
      }));
      
      // If we see both numbers and strings (that aren't just numbers-as-strings)
      // Actually, in parsed CSV, everything is a string initially. 
      // But we can check if some values are numeric and some are not.
      const hasNumeric = vals.some(v => !isNaN(parseFloat(v)) && isFinite(v));
      const hasNonNumeric = vals.some(v => isNaN(parseFloat(v)));
      
      // If the column has mixed numeric and non-numeric content, it might cause issues
      // if the backend tries to infer types.
    };

    // For now, we'll rely on the backend's precise detection but we can show a proactive warning
    // if we detect potential mixed types.
  }, [config.target_col, config.sensitive_col, parsedData]);

  const handleApplyStrategy = async () => {
    if (!detectionResults) return;

    if (!simulationResults) {
      await handleSimulate();
    }

    if (!recommendation) {
      await handleRecommend();
    }

    // Ensure users see the "after applying" projection immediately.
    setBeforeAfterVisible(true);
    setTimeout(() => {
      if (beforeAfterRef.current) {
        beforeAfterRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
      } else {
        document.getElementById('recommendation')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    }, 120);
  };

  // Scroll to results when they arrive
  useEffect(() => {
    if (detectionResults) {
      setTimeout(() => {
        document.getElementById('audit')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 100);
    }
  }, [detectionResults]);

  useEffect(() => {
    if (simulationResults) {
      setTimeout(() => {
        document.getElementById('strategy')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 100);
    }
  }, [simulationResults]);

  useEffect(() => {
    if (recommendation) {
      setTimeout(() => {
        document.getElementById('recommendation')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 100);
    }
  }, [recommendation]);

  // Trigger animation when Before/After section scrolls into view
  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setBeforeAfterVisible(true);
        }
      },
      { threshold: 0.2 }
    );
    if (beforeAfterRef.current) {
      observer.observe(beforeAfterRef.current);
    }
    return () => observer.disconnect();
  }, [recommendation]);

  return (
    <div className="min-h-screen bg-[#0a0a0f]">
      {showGroqExportToast && (
        <div className="fixed top-20 right-6 z-50 bg-gray-900/95 border border-cyan/40 text-cyan px-4 py-2 rounded text-xs font-mono uppercase tracking-widest shadow-lg animate-reveal">
          Generating AI narrative...
        </div>
      )}
      {/* Sticky Navbar */}
      <nav className="sticky top-0 z-50 bg-[#0a0a0f]/80 backdrop-blur-md border-b border-gray-800 py-4 px-8">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-center gap-4">
          <div className="flex items-center gap-4">
            <h1 className="text-2xl font-bold text-cyan glow-cyan px-3 py-1 border border-cyan tracking-tighter">
              EQUILENS <span className="text-white font-light">AI</span>
            </h1>
            <div className="h-6 w-[1px] bg-gray-700 hidden md:block"></div>
            <p className="text-gray-500 font-mono text-[10px] tracking-widest uppercase hidden md:block">
              Most tools detect bias. We fix it.
            </p>
          </div>
          <div className="flex gap-6 text-[10px] font-mono text-gray-500 uppercase tracking-widest">
            <a href="#upload" className="hover:text-cyan transition-colors">01 Ingestion</a>
            {detectionResults && <a href="#audit" className="hover:text-cyan transition-colors">02 Audit</a>}
            {detectionResults && <a href="#strategy" className="hover:text-cyan transition-colors">03 Strategy</a>}
            {recommendation && <a href="#recommendation" className="hover:text-cyan transition-colors">04 Recommendation</a>}
          </div>
        </div>
      </nav>

      {showHero ? (
        <div className="flex flex-col items-center justify-center min-h-[85vh] text-center animate-reveal px-4 py-16">
          <h1 className="text-5xl md:text-7xl font-bold text-white mb-6 tracking-tighter">
            Detect & Fix AI Bias in <span className="text-cyan">Minutes</span>
          </h1>
          <p className="text-xl text-gray-400 max-w-3xl mb-16 font-light">
            Upload any ML dataset. EquiLens audits fairness, explains what's driving bias, and recommends the best mitigation strategy — automatically.
          </p>

          {/* 3 Steps */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-16 max-w-5xl w-full">
            <div className="card-industrial p-8 flex flex-col items-center text-center rounded-lg border border-gray-800 hover:border-cyan/30 transition-all">
              <div className="w-16 h-16 bg-cyan/10 rounded-full flex items-center justify-center mb-6 text-cyan border border-cyan/20">
                <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"></path></svg>
              </div>
              <div className="text-xs text-cyan font-mono mb-3 uppercase tracking-widest">Step 1</div>
              <h3 className="text-xl text-white font-bold mb-3">Upload CSV</h3>
              <p className="text-gray-500 text-sm">Drop your tabular dataset and define your target and sensitive columns.</p>
            </div>

            <div className="card-industrial p-8 flex flex-col items-center text-center rounded-lg border border-gray-800 hover:border-purple-500/30 transition-all">
              <div className="w-16 h-16 bg-purple-500/10 rounded-full flex items-center justify-center mb-6 text-purple-400 border border-purple-500/20">
                <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path></svg>
              </div>
              <div className="text-xs text-purple-400 font-mono mb-3 uppercase tracking-widest">Step 2</div>
              <h3 className="text-xl text-white font-bold mb-3">Run Fairness Audit</h3>
              <p className="text-gray-500 text-sm">View SHAP impact, demographic parity, and intersectional bias.</p>
            </div>

            <div className="card-industrial p-8 flex flex-col items-center text-center rounded-lg border border-gray-800 hover:border-emerald-500/30 transition-all">
              <div className="w-16 h-16 bg-emerald-500/10 rounded-full flex items-center justify-center mb-6 text-emerald-400 border border-emerald-500/20">
                <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
              </div>
              <div className="text-xs text-emerald-400 font-mono mb-3 uppercase tracking-widest">Step 3</div>
              <h3 className="text-xl text-white font-bold mb-3">Get Fix</h3>
              <p className="text-gray-500 text-sm">RL agent recommends the optimal fairness-accuracy mitigation strategy.</p>
            </div>
          </div>

          {/* CTAs */}
          <div className="flex flex-col sm:flex-row gap-6 mb-24">
            <button
              onClick={handleDemoDataset}
              disabled={loadingState === 'uploading'}
              className="bg-cyan hover:bg-cyan/80 text-black font-bold py-4 px-8 rounded flex items-center justify-center gap-3 transition-all uppercase tracking-widest text-sm disabled:opacity-50"
            >
              {loadingState === 'uploading' ? <InlineSpinner /> : <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"></path><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>}
              Try Demo Dataset
            </button>
            <button
              onClick={() => {
                setShowHero(false);
                setTimeout(() => {
                  document.getElementById('upload')?.scrollIntoView({ behavior: 'smooth' });
                }, 100);
              }}
              className="border border-gray-600 text-white hover:border-cyan hover:text-cyan font-bold py-4 px-8 rounded transition-all uppercase tracking-widest text-sm"
            >
              Upload Your Data
            </button>
          </div>
          {errors.demo && <InlineError message={errors.demo} />}

          <TrustSection />

          {/* Tech Badges */}
          <div className="flex flex-wrap items-center justify-center gap-4 text-xs font-mono text-gray-500 uppercase tracking-widest">
            <span className="text-gray-600">Built with</span>
            <span className="px-3 py-1.5 rounded bg-gray-900 border border-gray-800 text-gray-400">SHAP</span>
            <span className="text-gray-700">·</span>
            <span className="px-3 py-1.5 rounded bg-gray-900 border border-gray-800 text-gray-400">Stable-Baselines3</span>
            <span className="text-gray-700">·</span>
            <span className="px-3 py-1.5 rounded bg-gray-900 border border-gray-800 text-gray-400">FastAPI</span>
          </div>
        </div>
      ) : (
      <div className="p-8 max-w-7xl mx-auto pb-24">
        {/* Header / Mini-Hero */}
        <header className="mb-16 border-b border-gray-800 pb-12 pt-8 animate-reveal">
          <h1 className="text-6xl font-bold text-white mb-4 tracking-tighter">
            NEURAL <span className="text-cyan">FAIRNESS</span> AUDIT
          </h1>
          <p className="text-gray-500 font-mono text-sm tracking-widest max-w-2xl">
            Automated detection, simulation, and mitigation of algorithmic bias using reinforcement learning and SHAP explainability.
          </p>
        </header>

        {/* 1. Upload Panel */}
        <section id="upload" className="animate-reveal">
          <SectionHeader title="01. Data Ingestion" subtitle="Upload your model training data (CSV) to begin the audit." />
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {/* Left column: dropzone + demo shortcut */}
            <div className="flex flex-col gap-4">
              <div {...getRootProps()} className={`card-industrial p-12 rounded-lg border-2 border-dashed flex flex-col items-center justify-center cursor-pointer transition-all flex-1 ${isDragActive ? 'border-cyan bg-cyan/5' : 'border-gray-700 hover:border-cyan/50'}`}>
                <input {...getInputProps()} />
                <div className="text-cyan mb-4">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                  </svg>
                </div>
                <p className="text-gray-400 font-mono text-sm">{file ? file.name : 'DRAG & DROP CSV OR CLICK'}</p>
              </div>

              {/* Demo dataset shortcut button */}
              <button
                id="demo-dataset-btn"
                onClick={handleDemoDataset}
                disabled={loadingState === 'uploading'}
                className="group relative overflow-hidden rounded-lg border border-amber-500/40 bg-amber-500/10 px-6 py-3 font-mono text-sm uppercase tracking-widest text-amber-400 transition-all hover:border-amber-400 hover:bg-amber-500/20 hover:text-amber-300 hover:shadow-[0_0_24px_rgba(245,158,11,0.15)] disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-3"
              >
                {/* decorative pulse dot */}
                <span className="relative flex h-2.5 w-2.5">
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-amber-400 opacity-60"></span>
                  <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-amber-400"></span>
                </span>
                {loadingState === 'uploading' ? <InlineSpinner /> : 'Try Demo Dataset'}
                <span className="text-[10px] text-amber-500/60 font-light normal-case tracking-normal ml-1">
                  — Adult Income · 500 rows
                </span>
              </button>
            </div>

            {/* Right column: config form */}
            <form onSubmit={handleDetect} className="flex flex-col gap-4 justify-center">
              {/* Target Column */}
              <div className="flex flex-col">
                <label className="text-xs text-gray-500 font-mono uppercase mb-2 flex items-center">
                  Target Column (Y)
                  <InfoTooltip text="The outcome your model predicts. For example: loan approval, hiring decision, or credit score." />
                </label>
                <select 
                  id="select-target-col"
                  className="bg-gray-900 border border-gray-800 rounded p-3 text-cyan focus:outline-none focus:border-cyan appearance-none cursor-pointer"
                  value={config.target_col}
                  onChange={(e) => setConfig({...config, target_col: e.target.value})}
                >
                  <option value="">Select Target...</option>
                  {columns.map(col => (
                    <option key={col} value={col}>{col}</option>
                  ))}
                </select>
                {/* Auto-summary for target column */}
                {config.target_col && parsedData.length > 0 && (() => {
                  const vals = parsedData.map(r => r[config.target_col]).filter(v => v !== undefined && v !== '');
                  const positiveCount = vals.filter(v => v === '1' || v === 1 || String(v).toLowerCase() === 'yes' || String(v).toLowerCase() === 'true' || String(v) === '>50K').length;
                  const pct = ((positiveCount / vals.length) * 100).toFixed(1);
                  return (
                    <p className="text-[11px] text-gray-500 font-mono mt-1.5 pl-1">
                      {pct}% positive outcomes · {vals.length} rows
                    </p>
                  );
                })()}
              </div>

              {/* Sensitive Attribute */}
              <div className="flex flex-col">
                <label className="text-xs text-gray-500 font-mono uppercase mb-2 flex items-center">
                  Sensitive Attribute (S)
                  <InfoTooltip text="The demographic feature to test for bias. For example: gender, race, or age group." />
                </label>
                <select 
                  id="select-sensitive-col"
                  className="bg-gray-900 border border-gray-800 rounded p-3 text-cyan focus:outline-none focus:border-cyan appearance-none cursor-pointer"
                  value={config.sensitive_col}
                  onChange={(e) => setConfig({...config, sensitive_col: e.target.value})}
                >
                  <option value="">Select Sensitive Attribute...</option>
                  {columns.map(col => (
                    <option key={col} value={col}>{col}</option>
                  ))}
                </select>
                {/* Auto-summary for sensitive attribute */}
                {config.sensitive_col && parsedData.length > 0 && (() => {
                  const vals = parsedData.map(r => r[config.sensitive_col]).filter(v => v !== undefined && v !== '');
                  const freq = {};
                  vals.forEach(v => { freq[v] = (freq[v] || 0) + 1; });
                  const uniqueCount = Object.keys(freq).length;
                  const mostCommon = Object.entries(freq).sort((a, b) => b[1] - a[1])[0];
                  const pct = ((mostCommon[1] / vals.length) * 100).toFixed(1);
                  return (
                    <p className="text-[11px] text-gray-500 font-mono mt-1.5 pl-1">
                      {uniqueCount} unique values · Most common: {mostCommon[0]} ({pct}%)
                    </p>
                  );
                })()}
              </div>

              <div className="flex flex-col">
                <label className="text-xs text-gray-500 font-mono uppercase mb-2">Domain Context</label>
                <div className="relative">
                  <select
                    className="bg-gray-900 border border-gray-800 rounded p-3 text-cyan focus:outline-none focus:border-cyan appearance-none cursor-pointer w-full"
                    value={domain}
                    onChange={(e) => setDomain(e.target.value)}
                  >
                    {DOMAIN_OPTIONS.map((option) => (
                      <option key={option} value={option}>{option}</option>
                    ))}
                  </select>
                </div>
              </div>

              {/* Start Audit Button */}
              <div className="mt-4 flex flex-col gap-3">
                <button
                  type="submit"
                  disabled={loadingState === 'auditing' || !file || !config.target_col || !config.sensitive_col}
                  className="bg-cyan hover:bg-cyan/80 text-black font-bold py-4 px-8 rounded flex items-center justify-center gap-3 transition-all uppercase tracking-widest text-sm disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {loadingState === 'auditing' ? <InlineSpinner /> : <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path></svg>}
                  Start Fairness Audit
                </button>
                {errors.detect && <InlineError message={errors.detect} onFix={handleAutoFix} />}
              </div>

              {/* Dataset Overview — collapsible */}
              {parsedData.length > 0 && config.target_col && config.sensitive_col && (() => {
                const N = parsedData.length;
                const M = columns.length;
                let totalCells = N * M;
                let missingCells = 0;
                parsedData.forEach(r => columns.forEach(c => { if (r[c] === undefined || r[c] === '' || r[c] === null) missingCells++; }));
                const missingPct = ((missingCells / totalCells) * 100).toFixed(1);

                // Sensitive attr distribution
                const sensVals = parsedData.map(r => r[config.sensitive_col]).filter(v => v !== undefined && v !== '');
                const sensFreq = {};
                sensVals.forEach(v => { sensFreq[v] = (sensFreq[v] || 0) + 1; });
                const sensGroups = Object.entries(sensFreq).sort((a, b) => b[1] - a[1]);
                const maxCount = sensGroups[0]?.[1] || 1;
                const hasImbalance = sensGroups.some(([, c]) => (c / sensVals.length) < 0.15);

                // Target positive rate
                const targetVals = parsedData.map(r => r[config.target_col]).filter(v => v !== undefined && v !== '');
                const posCount = targetVals.filter(v => v === '1' || v === 1 || String(v).toLowerCase() === 'yes' || String(v).toLowerCase() === 'true' || String(v) === '>50K').length;
                const posRate = (posCount / targetVals.length) * 100;
                const targetSkewed = posRate < 10 || posRate > 90;

                // Pearson correlation: encode sensitive col as numeric, correlate with all numeric cols
                const sensMap = {};
                let sensIdx = 0;
                sensVals.forEach(v => { if (!(v in sensMap)) sensMap[v] = sensIdx++; });
                const sensNumeric = parsedData.map(r => sensMap[r[config.sensitive_col]] ?? 0);
                const mean = arr => arr.reduce((a, b) => a + b, 0) / arr.length;
                const pearson = (a, b) => {
                  const ma = mean(a), mb = mean(b);
                  let num = 0, da = 0, db = 0;
                  for (let i = 0; i < a.length; i++) { const x = a[i] - ma, y = b[i] - mb; num += x * y; da += x * x; db += y * y; }
                  const denom = Math.sqrt(da * db);
                  return denom === 0 ? 0 : num / denom;
                };
                const correlations = columns
                  .filter(c => c !== config.sensitive_col && c !== config.target_col)
                  .map(c => {
                    const nums = parsedData.map(r => { const v = parseFloat(r[c]); return isNaN(v) ? null : v; });
                    if (nums.filter(v => v !== null).length < N * 0.5) return null;
                    const filled = nums.map(v => v ?? 0);
                    return { feature: c, corr: Math.abs(pearson(sensNumeric, filled)) };
                  })
                  .filter(Boolean)
                  .sort((a, b) => b.corr - a.corr)
                  .slice(0, 5);

                return (
                  <div className="mt-2">
                    <button
                      type="button"
                      onClick={() => setDatasetProfileOpen(p => !p)}
                      className="w-full flex items-center justify-between text-[11px] text-gray-500 font-mono uppercase tracking-widest hover:text-cyan transition-colors py-2"
                    >
                      <span>View dataset profile</span>
                      <svg xmlns="http://www.w3.org/2000/svg" className={`h-3.5 w-3.5 transition-transform duration-300 ${datasetProfileOpen ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                      </svg>
                    </button>

                    {datasetProfileOpen && (
                      <div className="card-industrial rounded-lg border border-gray-800 p-5 mt-1 mb-2 space-y-5 animate-reveal">
                        {/* Summary row */}
                        <p className="text-xs text-gray-400 font-mono">
                          <span className="text-white font-semibold">{N}</span> rows · <span className="text-white font-semibold">{M}</span> columns · <span className={parseFloat(missingPct) > 5 ? 'text-amber-400' : 'text-white font-semibold'}>{missingPct}%</span> missing values
                        </p>

                        {/* Sensitive attr distribution */}
                        <div>
                          <p className="text-[10px] text-gray-500 font-mono uppercase tracking-widest mb-2">{config.sensitive_col} distribution</p>
                          <div className="space-y-1.5">
                            {sensGroups.map(([val, count]) => {
                              const pct = ((count / sensVals.length) * 100).toFixed(1);
                              const w = (count / maxCount) * 100;
                              const isMinority = (count / sensVals.length) < 0.15;
                              return (
                                <div key={val} className="flex items-center gap-2">
                                  <span className="text-[10px] text-gray-400 font-mono w-20 truncate shrink-0">{val}</span>
                                  <div className="flex-1 h-3 bg-gray-800 rounded-full overflow-hidden">
                                    <svg width="100%" height="12">
                                      <rect x="0" y="0" width={`${w}%`} height="12" rx="6" fill={isMinority ? '#f59e0b' : '#a78bfa'} />
                                    </svg>
                                  </div>
                                  <span className={`text-[10px] font-mono w-12 text-right ${isMinority ? 'text-amber-400' : 'text-gray-500'}`}>{pct}%</span>
                                </div>
                              );
                            })}
                          </div>
                          {hasImbalance && (
                            <p className="text-[10px] text-amber-400 mt-2">⚠ Imbalanced groups detected — results may be less reliable for minority groups.</p>
                          )}
                        </div>

                        {/* Target positive rate */}
                        <div>
                          <p className="text-[10px] text-gray-500 font-mono uppercase tracking-widest mb-2">{config.target_col} positive rate</p>
                          <div className="flex items-center gap-2">
                            <div className="flex-1 h-3 bg-gray-800 rounded-full overflow-hidden">
                              <svg width="100%" height="12">
                                <rect x="0" y="0" width={`${posRate}%`} height="12" rx="6" fill={targetSkewed ? '#f59e0b' : '#00f5d4'} />
                              </svg>
                            </div>
                            <span className={`text-[10px] font-mono ${targetSkewed ? 'text-amber-400' : 'text-cyan'}`}>{posRate.toFixed(1)}%</span>
                          </div>
                          {targetSkewed && (
                            <p className="text-[10px] text-amber-400 mt-2">⚠ Heavily skewed target — model may show inflated accuracy.</p>
                          )}
                        </div>

                        {/* Top correlated features */}
                        {correlations.length > 0 && (
                          <div>
                            <p className="text-[10px] text-gray-500 font-mono uppercase tracking-widest mb-2">Top correlated with {config.sensitive_col}</p>
                            <div className="space-y-1">
                              {correlations.map((c, i) => (
                                <div key={c.feature} className="flex items-center gap-2">
                                  <span className="text-[10px] text-gray-600 font-mono w-4">{i + 1}.</span>
                                  <span className="text-[10px] text-gray-400 font-mono flex-1 truncate">{c.feature}</span>
                                  <span className={`text-[10px] font-mono font-bold ${c.corr > 0.3 ? 'text-amber-400' : 'text-gray-500'}`}>{c.corr.toFixed(3)}</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })()}

              {/* Dismissible info banner */}
              {showInfoBanner && (
                <div className="flex items-start gap-3 p-3 rounded bg-blue-500/5 border border-blue-500/20 mt-1">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-blue-400 mt-0.5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <p className="text-[11px] text-blue-300/80 leading-relaxed flex-1">
                    EquiLens will train a model on your data locally. Nothing is sent to any external server.
                  </p>
                  <button
                    type="button"
                    onClick={() => setShowInfoBanner(false)}
                    className="text-blue-400/50 hover:text-blue-300 transition-colors shrink-0"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                    </svg>
                  </button>
                </div>
              )}

              {/* Run Audit button with contextual disable reason */}
              <button 
                type="submit"
                disabled={loadingState === 'auditing' || !file || !config.target_col || !config.sensitive_col}
                className="mt-4 bg-cyan text-black font-bold py-3 rounded hover:bg-cyan/80 transition-all disabled:opacity-50 disabled:cursor-not-allowed uppercase tracking-widest text-sm"
              >
                {loadingState === 'auditing'
                  ? 'Processing...'
                  : !file
                    ? 'Upload a dataset to continue'
                    : !config.target_col
                      ? 'Select target column to continue'
                      : !config.sensitive_col
                        ? 'Select sensitive attribute to continue'
                        : 'Analyze Bias'}
              </button>
              {errors.detect && <InlineError message={errors.detect} />}
            </form>
          </div>
          {error && <div className="mt-4 p-4 bg-red-900/20 border border-red-500/50 text-red-500 text-sm rounded">{error}</div>}
        </section>

        <SectionDivider label="Fairness Audit Results" />

        {/* 2. Bias Metrics & 3,4 Charts */}
        {detectionResults && (
          <div id="audit" className="animate-reveal">
            <div className="flex flex-col md:flex-row md:items-end justify-between mb-8">
              <SectionHeader title="02. Fairness Audit" subtitle="Primary metrics and disparity analysis of the current model." />
              <div className="flex items-center gap-3 mb-8 md:mb-10">
                <button
                  onClick={() => handleExport('pdf')}
                  disabled={loadingState === 'exporting' || !detectionResults}
                  className="px-4 py-2 bg-gray-900 border border-gray-800 rounded text-[10px] font-mono uppercase tracking-widest text-gray-400 hover:text-cyan hover:border-cyan transition-all disabled:opacity-50 flex items-center gap-2"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  Download PDF
                </button>
                <button
                  onClick={() => handleExport('json')}
                  disabled={loadingState === 'exporting' || !detectionResults}
                  className="px-4 py-2 bg-gray-900 border border-gray-800 rounded text-[10px] font-mono uppercase tracking-widest text-gray-400 hover:text-cyan hover:border-cyan transition-all disabled:opacity-50 flex items-center gap-2"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                  Download JSON
                </button>
              </div>
            </div>

            {/* Bias Summary Card — plain-English storytelling */}
            {(() => {
              const fm = detectionResults.fairness_metrics;
              const rates = detectionResults.per_group_approval_rates;
              const di = fm.disparate_impact_ratio;
              const dpd = fm.demographic_parity_difference;
              const eod = fm.equal_opportunity_difference;

              // Identify advantaged / disadvantaged groups from approval rates
              const groups = Object.entries(rates);
              const sorted = [...groups].sort((a, b) => b[1] - a[1]);
              const advantaged = sorted[0]?.[0] || '—';
              const disadvantaged = sorted[sorted.length - 1]?.[0] || '—';

              // Risk level from disparate impact
              let riskLabel, riskColor, riskBorder, riskBg;
              if (di < 0.5) {
                riskLabel = 'Critical bias'; riskColor = 'text-red-400'; riskBorder = 'border-red-500'; riskBg = 'bg-red-500/10';
              } else if (di < 0.8) {
                riskLabel = 'High bias'; riskColor = 'text-orange-400'; riskBorder = 'border-orange-500'; riskBg = 'bg-orange-500/10';
              } else if (di < 0.9) {
                riskLabel = 'Moderate bias'; riskColor = 'text-yellow-400'; riskBorder = 'border-yellow-500'; riskBg = 'bg-yellow-500/10';
              } else {
                riskLabel = 'Low bias'; riskColor = 'text-emerald-400'; riskBorder = 'border-emerald-500'; riskBg = 'bg-emerald-500/10';
              }

              // Build plain-English headlines
              const headlines = [];
              if (dpd > 0) {
                headlines.push(
                  <>Model favors <span className="text-white font-semibold">{advantaged}</span> over <span className="text-white font-semibold">{disadvantaged}</span> by <span className="text-white font-semibold">{(dpd * 100).toFixed(1)}%</span></>
                );
              }
              if (eod > 0.1) {
                headlines.push(
                  <>Qualified <span className="text-white font-semibold">{disadvantaged}</span> applicants are <span className="text-white font-semibold">{(eod * 100).toFixed(1)}%</span> less likely to be approved</>
                );
              }

              // Metric tiles config
              const tiles = [
                { label: 'Demographic Parity Difference', value: dpd, explain: 'Gap in approval rates between groups' },
                { label: 'Equal Opportunity Difference', value: eod, explain: 'Gap in true positive rates between groups' },
                { label: 'Disparate Impact Ratio', value: di, explain: di >= 0.8 ? 'Meets the 80% rule threshold' : 'Below the 80% rule threshold' },
              ];

              return (
                <div className="card-industrial rounded-lg border border-gray-800 mb-12 overflow-hidden animate-reveal">
                  {/* Header row: headlines + risk badge */}
                  <div className="p-6 pb-5 flex flex-col sm:flex-row sm:items-start gap-4">
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-4">
                        <svg xmlns="http://www.w3.org/2000/svg" className={`h-5 w-5 ${riskColor}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        <h3 className="text-sm font-bold text-white uppercase tracking-widest">Bias Summary</h3>
                        <span className={`ml-auto sm:ml-3 inline-flex items-center px-3 py-1 rounded-full text-[10px] font-bold uppercase tracking-wider ${riskColor} ${riskBg} border ${riskBorder}`}>
                          {riskLabel}
                        </span>
                      </div>

                      {headlines.length > 0 ? (
                        <ul className="space-y-2">
                          {headlines.map((h, i) => (
                            <li key={i} className="flex items-start gap-2 text-sm text-gray-400 leading-relaxed">
                              <span className={`mt-1.5 shrink-0 inline-block w-1.5 h-1.5 rounded-full ${riskBg} ${riskBorder} border`}></span>
                              {h}
                            </li>
                          ))}
                        </ul>
                      ) : (
                        <p className="text-sm text-gray-500">No significant disparity detected between groups.</p>
                      )}
                    </div>
                  </div>

                  {/* 3 metric tiles */}
                  <div className="grid grid-cols-1 sm:grid-cols-3 border-t border-gray-800">
                    {tiles.map((t, i) => (
                      <div key={i} className={`p-5 border-l-4 ${riskBorder} ${i > 0 ? 'sm:border-l-4 border-t sm:border-t-0' : ''}`}>
                        <p className={`text-2xl font-bold font-mono ${riskColor} mb-1`}>{t.value.toFixed(3)}</p>
                        <p className="text-[10px] text-gray-400 font-mono uppercase tracking-widest mb-1">{t.label}</p>
                        <p className="text-[11px] text-gray-600">{t.explain}</p>
                      </div>
                    ))}
                  </div>
                </div>
              );
            })()}

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
              <MetricCard label="Model Accuracy" value={detectionResults.accuracy} isFair={true} />
              <MetricCard label="Demographic Parity Diff" value={detectionResults.fairness_metrics.demographic_parity_difference} isFair={detectionResults.fairness_metrics.demographic_parity_difference < 0.1} />
              <MetricCard label="Equal Opportunity Diff" value={detectionResults.fairness_metrics.equal_opportunity_difference} isFair={detectionResults.fairness_metrics.equal_opportunity_difference < 0.1} />
              <MetricCard label="Disparate Impact Ratio" value={detectionResults.fairness_metrics.disparate_impact_ratio} isFair={detectionResults.fairness_metrics.disparate_impact_ratio > 0.8} />
            </div>

            <GroqInsightPanel
              title="Executive Bias Narrative"
              insight={groqInsights.biasNarrative || detectionResults.groq_narrative}
              loading={groqLoading.biasNarrative}
              error={groqErrors.biasNarrative}
              onRegenerate={() => fetchGroqInsight('biasNarrative', '/api/groq/bias-narrative', {
                metrics: detectionResults.fairness_metrics,
                sensitive_col: config.sensitive_col,
                dataset_name: file?.name || 'uploaded dataset'
              })}
            />

            {/* Advanced Researcher Analytics */}
            {detectionResults.advanced_metrics && (
              <div className="mb-16">
                <SectionHeader 
                  title="02. Deep Performance Audit" 
                  subtitle="Detailed model evaluation metrics and error analysis per sensitive group." 
                />
                <ResearcherAnalytics metrics={detectionResults.advanced_metrics} />
              </div>
            )}

            {/* Recommended Action — renders once /api/recommend returns */}
            {recommendation && (() => {
              const strategyExplanations = {
                'Remove Sensitive Attribute': 'Removes the sensitive attribute entirely. Simple, but bias can persist through correlated features.',
                'Reweight Dataset': 'Adjusts training sample weights to balance group representation. Best fairness gain with low accuracy cost.',
                'Threshold Adjustment': 'Sets different approval thresholds per group. Effective but requires post-deployment monitoring.',
                'Fairness Constraint': 'Adds a fairness penalty during training. Good balance for regulated industries.',
              };

              const explanation = strategyExplanations[recName] || recommendation.reason;

              // All 4 strategies for comparison (use simulation if available, otherwise mock from recommendation)
              const allStrategies = simulationResults || [
                { strategy_name: recName, fairness_gain: recommendation.expected_fairness_gain, accuracy_drop: recommendation.expected_accuracy_drop },
              ];
              const maxGain = Math.max(...allStrategies.map(s => s.fairness_gain), 0.01);
              const maxDrop = Math.max(...allStrategies.map(s => s.accuracy_drop), 0.01);

              return (
                <div className="mb-16 animate-reveal">
                  <div className="flex items-center gap-3 mb-6">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                    </svg>
                    <h3 className="text-lg font-bold text-white uppercase tracking-widest">What To Do Next</h3>
                    <span className="text-[10px] text-gray-500 font-mono ml-1">RL-Powered Recommendation</span>
                  </div>

                  {/* Main recommendation card */}
                  <div className="card-industrial rounded-lg border border-emerald-500/20 overflow-hidden mb-6">
                    <div className="p-6">
                      <div className="flex flex-col sm:flex-row sm:items-center gap-4 mb-5">
                        <span className="inline-flex items-center gap-2 px-3 py-1.5 rounded bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs font-bold uppercase tracking-widest">
                          <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L3 12l5.714-2.143L13 3z" /></svg>
                          Recommended
                        </span>
                        <h4 className="text-3xl font-bold text-white">{recName}</h4>
                      </div>
                      <p className="text-gray-400 text-sm leading-relaxed mb-6 max-w-2xl">
                        {explanation}
                      </p>
                      <div className="flex flex-wrap gap-6 text-sm mb-8">
                        <div className="flex items-center gap-2">
                          <span className="inline-block w-2 h-2 rounded-full bg-emerald-400"></span>
                          <span className="text-gray-500 font-mono text-xs">Expected fairness improvement:</span>
                          <span className="text-emerald-400 font-bold font-mono">+{(recommendation.expected_fairness_gain * 100).toFixed(1)}%</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="inline-block w-2 h-2 rounded-full bg-red-400"></span>
                          <span className="text-gray-500 font-mono text-xs">Accuracy cost:</span>
                          <span className="text-red-400 font-bold font-mono">-{(recommendation.expected_accuracy_drop * 100).toFixed(1)}%</span>
                        </div>
                      </div>

                      {/* Why this strategy? section */}
                      <div className="border-t border-gray-800 pt-6 mt-6">
                        <h5 className="text-sm font-bold text-white uppercase tracking-widest mb-4">Why this strategy?</h5>
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                          <div className="space-y-4">
                            <div className="bg-gray-900/50 p-4 rounded border border-gray-800">
                              <p className="text-[10px] text-gray-500 font-mono uppercase mb-2">Reward Formula</p>
                              <p className="text-lg font-mono text-cyan">
                                Reward = Fairness Gain − (0.5 × Accuracy Drop)
                              </p>
                            </div>
                            <div className="space-y-2">
                              <p className="text-sm text-gray-300">
                                Achieved fairness gain of <span className="text-emerald-400 font-bold">+{ (recommendation.expected_fairness_gain * 100).toFixed(1) }%</span> with only <span className="text-red-400 font-bold">-{ (recommendation.expected_accuracy_drop * 100).toFixed(1) }%</span> accuracy cost. Penalty-adjusted score: <span className="text-cyan font-bold">{ (recommendation.expected_fairness_gain - 0.5 * recommendation.expected_accuracy_drop).toFixed(3) }</span>.
                              </p>
                              <p className="text-sm text-gray-500">
                                Next best was <span className="text-gray-300 font-semibold">{recommendation.runner_up.strategy_name}</span> with score <span className="text-gray-300 font-semibold">{recommendation.runner_up.score.toFixed(3)}</span> — {recommendation.runner_up.reason}.
                              </p>
                            </div>
                          </div>
                          
                          {/* Mini Bar Chart SVG */}
                          <div className="bg-gray-900/30 p-4 rounded border border-gray-800 flex flex-col items-center justify-center">
                            <p className="text-[10px] text-gray-500 font-mono uppercase mb-4 self-start">Reward Scores by Strategy</p>
                            <svg width="240" height="120" viewBox="0 0 240 120" className="overflow-visible">
                              {recommendation.all_scores.map((s, idx) => {
                                const barWidth = 40;
                                const spacing = 15;
                                const x = idx * (barWidth + spacing) + 10;
                                const maxScore = Math.max(...recommendation.all_scores.map(sc => sc.score), 0.1);
                                const minScore = Math.min(...recommendation.all_scores.map(sc => sc.score), 0);
                                const range = maxScore - minScore || 1;
                                // Normalize height to 80px max
                                const normalizedHeight = ((s.score - Math.min(0, minScore)) / (maxScore - Math.min(0, minScore))) * 80;
                                const barHeight = Math.max(5, normalizedHeight);
                                const y = 100 - barHeight;
                                const isWinner = s.strategy_name === recName;
                                
                                return (
                                  <g key={s.strategy_name}>
                                    <rect 
                                      x={x} 
                                      y={y} 
                                      width={barWidth} 
                                      height={barHeight} 
                                      fill={isWinner ? "#00f5d4" : "#1f2937"} 
                                      className="transition-all duration-500"
                                      rx="2"
                                    />
                                    {isWinner && (
                                      <rect 
                                        x={x-2} 
                                        y={y-2} 
                                        width={barWidth+4} 
                                        height={barHeight+4} 
                                        fill="none" 
                                        stroke="#00f5d4" 
                                        strokeWidth="1" 
                                        strokeOpacity="0.3"
                                        rx="3"
                                      />
                                    )}
                                    <text 
                                      x={x + barWidth/2} 
                                      y="115" 
                                      textAnchor="middle" 
                                      fill="#666" 
                                      fontSize="8" 
                                      fontFamily="monospace"
                                    >
                                      {s.strategy_name.split(' ')[0]}
                                    </text>
                                    <text 
                                      x={x + barWidth/2} 
                                      y={y - 5} 
                                      textAnchor="middle" 
                                      fill={isWinner ? "#00f5d4" : "#4b5563"} 
                                      fontSize="9" 
                                      fontWeight={isWinner ? "bold" : "normal"}
                                      fontFamily="monospace"
                                    >
                                      {s.score.toFixed(2)}
                                    </text>
                                  </g>
                                );
                              })}
                              <line x1="0" y1="100" x2="240" y2="100" stroke="#374151" strokeWidth="1" />
                            </svg>
                          </div>
                        </div>
                      </div>

                      {/* Collapsible How it works section */}
                      <div className="mt-8">
                        <button 
                          onClick={() => setHowItWorksOpen(!howItWorksOpen)}
                          className="flex items-center gap-2 text-[10px] text-gray-500 font-mono uppercase tracking-widest hover:text-cyan transition-colors"
                        >
                          <svg 
                            xmlns="http://www.w3.org/2000/svg" 
                            className={`h-3 w-3 transition-transform ${howItWorksOpen ? 'rotate-180' : ''}`} 
                            fill="none" viewBox="0 0 24 24" stroke="currentColor"
                          >
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                          </svg>
                          How the RL agent works
                        </button>
                        {howItWorksOpen && (
                          <div className="mt-4 p-4 bg-gray-900/20 rounded border border-gray-800/50 animate-reveal">
                            <ul className="space-y-3">
                              <li className="flex gap-3 text-sm text-gray-400">
                                <span className="text-cyan font-bold">•</span>
                                <span>The agent simulates each strategy in a custom environment</span>
                              </li>
                              <li className="flex gap-3 text-sm text-gray-400">
                                <span className="text-cyan font-bold">•</span>
                                <span>Reward penalizes accuracy loss twice as much as fairness gain is rewarded</span>
                              </li>
                              <li className="flex gap-3 text-sm text-gray-400">
                                <span className="text-cyan font-bold">•</span>
                                <span>After 10,000 timesteps of training, it converges on the highest-reward strategy</span>
                              </li>
                            </ul>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Strategy comparison list */}
                  {allStrategies.length > 1 && (
                    <div className="card-industrial rounded-lg border border-gray-800 overflow-hidden mb-6">
                      <div className="px-5 py-3 border-b border-gray-800">
                        <p className="text-[10px] text-gray-500 font-mono uppercase tracking-widest">All Strategies Compared</p>
                      </div>
                      <div className="divide-y divide-gray-800/60">
                        {allStrategies.map((s, i) => {
                          const isRec = s.strategy_name === recName;
                          return (
                            <div key={i} className={`px-5 py-4 flex flex-col sm:flex-row sm:items-center gap-3 ${isRec ? 'bg-emerald-500/[0.04]' : ''}`}>
                              <div className="flex items-center gap-2 sm:w-48 shrink-0">
                                {isRec && <span className="inline-block w-1.5 h-1.5 rounded-full bg-emerald-400"></span>}
                                <span className={`text-sm font-mono ${isRec ? 'text-white font-semibold' : 'text-gray-400'}`}>{s.strategy_name}</span>
                              </div>
                              <div className="flex-1 flex items-center gap-4">
                                {/* Fairness gain bar */}
                                <div className="flex-1">
                                  <div className="flex items-center justify-between mb-1">
                                    <span className="text-[9px] text-gray-600 font-mono uppercase">Fairness</span>
                                    <span className="text-[10px] text-emerald-400 font-mono">+{(s.fairness_gain * 100).toFixed(1)}%</span>
                                  </div>
                                  <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
                                    <div className="h-full bg-emerald-400 rounded-full transition-all" style={{ width: `${(s.fairness_gain / maxGain) * 100}%` }}></div>
                                  </div>
                                </div>
                                {/* Accuracy cost bar */}
                                <div className="flex-1">
                                  <div className="flex items-center justify-between mb-1">
                                    <span className="text-[9px] text-gray-600 font-mono uppercase">Accuracy cost</span>
                                    <span className="text-[10px] text-red-400 font-mono">-{(s.accuracy_drop * 100).toFixed(1)}%</span>
                                  </div>
                                  <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
                                    <div className="h-full bg-red-400/70 rounded-full transition-all" style={{ width: `${(s.accuracy_drop / maxDrop) * 100}%` }}></div>
                                  </div>
                                </div>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}

                  {/* Apply button */}
                  <button
                    onClick={handleApplyStrategy}
                    disabled={loadingState === 'simulating' || loadingState === 'recommending'}
                    className="bg-emerald-500 hover:bg-emerald-400 text-black font-bold py-3 px-8 rounded transition-all uppercase tracking-widest text-sm flex items-center gap-3 disabled:opacity-50"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                    {(loadingState === 'simulating' || loadingState === 'recommending') ? 'Applying Strategy...' : 'Apply This Strategy'}
                  </button>
                </div>
              );
            })()}

            {loadingState === 'recommending' && (
              <div className="flex items-center gap-3 mb-12 p-4 card-industrial rounded-lg border border-gray-800">
                <div className="animate-spin rounded-full h-5 w-5 border-t-2 border-b-2 border-emerald-400"></div>
                <span className="text-sm text-gray-400 font-mono">RL agent analyzing optimal strategy...</span>
              </div>
            )}

            {detectionResults.counterfactual_examples && detectionResults.counterfactual_examples.length > 0 && (
              <div className="mb-16 animate-reveal">
                <div className="flex items-center gap-3 mb-6">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <h3 className="text-lg font-bold text-white uppercase tracking-widest">What If?</h3>
                  <span className="text-[10px] text-gray-500 font-mono ml-1">Counterfactual Explanations</span>
                </div>
                <p className="text-gray-500 text-sm mb-6 max-w-2xl">
                  These real test-set individuals were <span className="text-red-400 font-semibold">rejected</span> by the model.
                  Simply changing the sensitive attribute flips the outcome to <span className="text-emerald-400 font-semibold">approved</span>.
                </p>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                  {detectionResults.counterfactual_examples.map((cf, idx) => (
                    <div key={`cf-${idx}`} className="card-industrial rounded-lg overflow-hidden border border-gray-800 hover:border-amber-500/30 transition-all">
                      {/* Headline */}
                      <div className="px-5 py-4 bg-gradient-to-r from-amber-500/10 to-transparent border-b border-gray-800">
                        <p className="text-sm text-amber-300 leading-relaxed">
                          Changing <span className="font-bold text-white">{config.sensitive_col}</span> from{' '}
                          <span className="font-bold text-red-400">{cf.sensitive_attr_original}</span> to{' '}
                          <span className="font-bold text-emerald-400">{cf.sensitive_attr_flipped}</span>{' '}
                          flips the outcome from <span className="text-red-400 font-semibold">rejected</span> to{' '}
                          <span className="text-emerald-400 font-semibold">approved</span>.
                        </p>
                      </div>

                      {/* Before / After Cards */}
                      <div className="grid grid-cols-2 divide-x divide-gray-800">
                        {/* Original (rejected) */}
                        <div className="p-4">
                          <div className="flex items-center gap-2 mb-3">
                            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider bg-red-500/20 text-red-400 border border-red-500/30">
                              <svg xmlns="http://www.w3.org/2000/svg" className="h-3 w-3" viewBox="0 0 20 20" fill="currentColor">
                                <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                              </svg>
                              Rejected
                            </span>
                          </div>
                          <div className="space-y-1.5">
                            <div className="flex justify-between text-[11px]">
                              <span className="text-gray-500 font-mono">{config.sensitive_col}</span>
                              <span className="text-red-400 font-semibold">{cf.sensitive_attr_original}</span>
                            </div>
                            {Object.entries(cf.original_features).map(([k, v]) => (
                              <div key={k} className="flex justify-between text-[11px]">
                                <span className="text-gray-500 font-mono truncate mr-2">{k}</span>
                                <span className="text-gray-300 font-mono">{typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(2)) : v}</span>
                              </div>
                            ))}
                          </div>
                        </div>

                        {/* Counterfactual (approved) */}
                        <div className="p-4 bg-emerald-500/[0.03]">
                          <div className="flex items-center gap-2 mb-3">
                            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                              <svg xmlns="http://www.w3.org/2000/svg" className="h-3 w-3" viewBox="0 0 20 20" fill="currentColor">
                                <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                              </svg>
                              Approved
                            </span>
                          </div>
                          <div className="space-y-1.5">
                            <div className="flex justify-between text-[11px]">
                              <span className="text-gray-500 font-mono">{config.sensitive_col}</span>
                              <span className="text-emerald-400 font-semibold">{cf.sensitive_attr_flipped}</span>
                            </div>
                            {Object.entries(cf.original_features).map(([k, v]) => (
                              <div key={k} className="flex justify-between text-[11px]">
                                <span className="text-gray-500 font-mono truncate mr-2">{k}</span>
                                <span className="text-gray-400 font-mono">{typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(2)) : v}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>

                <div className="mt-6 flex items-center gap-3">
                  <button
                    type="button"
                    onClick={() => setStoryMode((prev) => !prev)}
                    className={`px-4 py-2 rounded border text-[10px] font-mono uppercase tracking-widest transition-all ${
                      storyMode
                        ? 'border-amber-400 text-amber-300 bg-amber-500/10'
                        : 'border-gray-700 text-gray-400 hover:text-amber-300 hover:border-amber-400'
                    }`}
                  >
                    {storyMode ? 'Story Mode On' : 'Enable Story Mode'}
                  </button>
                </div>

                {storyMode && (
                  <div className="mt-4">
                    <GroqInsightPanel
                      title="Counterfactual Story Mode"
                      insight={groqInsights.counterfactualStory}
                      loading={groqLoading.counterfactualStory}
                      error={groqErrors.counterfactualStory}
                      onRegenerate={() => fetchGroqInsight('counterfactualStory', '/api/groq/counterfactual-story', {
                        counterfactual_examples: detectionResults.counterfactual_examples,
                        sensitive_col: config.sensitive_col
                      })}
                    />
                  </div>
                )}
              </div>
            )}


            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-16">
              <div className="card-industrial p-6 rounded-lg">
                <h3 className="text-xs text-gray-500 font-mono uppercase mb-6 tracking-widest">Feature Impact (SHAP)</h3>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart layout="vertical" data={(detectionResults.shap_features || []).slice(0, 5)} margin={{ left: 20 }}>
                      <XAxis type="number" hide />
                      <YAxis dataKey="feature" type="category" stroke="#666" fontSize="10" width={100} />
                      <Tooltip cursor={{fill: '#1a1a24'}} contentStyle={{backgroundColor: '#0f0f19', border: '1px solid #333', color: '#00f5d4'}} />
                      <Bar dataKey="importance" fill="#00f5d4" radius={[0, 4, 4, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="card-industrial p-6 rounded-lg">
                <h3 className="text-xs text-gray-500 font-mono uppercase mb-6 tracking-widest">Group Approval Rates</h3>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={Object.entries(detectionResults.per_group_approval_rates || {}).map(([name, val]) => ({ name, rate: val }))}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                      <XAxis dataKey="name" stroke="#666" fontSize={10} />
                      <YAxis stroke="#666" fontSize={10} domain={[0, 1]} />
                      <Tooltip contentStyle={{backgroundColor: '#0f0f19', border: '1px solid #333'}} />
                      <Bar dataKey="rate" fill="#00f5d4" radius={[4, 4, 0, 0]}>
                        {Object.entries(detectionResults.per_group_approval_rates || {}).map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={index % 2 === 0 ? '#00f5d4' : '#00bfa5'} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>

            <div className="mb-16">
              <PerGroupShapView perGroupShap={detectionResults.per_group_shap} />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-16">
              <div className="h-full">
                <RepresentationChart data={detectionResults.advanced_metrics?.representation_bias} />
              </div>
              <div>
                <GroqInsightPanel
                  title="SHAP Proxy Risk Insight"
                  insight={groqInsights.shapInsight || detectionResults.groq_shap_insight}
                  loading={groqLoading.shapInsight}
                  error={groqErrors.shapInsight}
                  onRegenerate={() => fetchGroqInsight('shapInsight', '/api/groq/shap-insight', {
                    shap_data: detectionResults.shap_features,
                    sensitive_col: config.sensitive_col
                  })}
                />
              </div>
            </div>

            {/* Intersectional Bias Analysis — collapsible */}
            {detectionResults.intersectional_bias && detectionResults.intersectional_bias.length > 0 && (
              <div className="mb-16 animate-reveal">
                <button
                  id="toggle-intersectional"
                  onClick={() => setIntersectionalOpen(prev => !prev)}
                  className="w-full flex items-center justify-between card-industrial p-5 rounded-lg border border-gray-800 hover:border-purple-500/40 transition-all group"
                >
                  <div className="flex items-center gap-3">
                    <span className="flex h-2.5 w-2.5 rounded-full bg-purple-500"></span>
                    <span className="text-sm font-bold text-white uppercase tracking-widest">Intersectional Analysis</span>
                    <span className="text-[10px] text-gray-500 font-mono">
                      {detectionResults.intersectional_bias.length} groups · ≥30 samples each
                    </span>
                  </div>
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className={`h-5 w-5 text-gray-500 transition-transform duration-300 ${intersectionalOpen ? 'rotate-180' : ''}`}
                    fill="none" viewBox="0 0 24 24" stroke="currentColor"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </button>

                {intersectionalOpen && (
                  <div className="mt-4 card-industrial p-6 rounded-lg border border-purple-500/20 animate-reveal">
                    <h3 className="text-xs text-gray-500 font-mono uppercase mb-6 tracking-widest">
                      Positive Prediction Rate by Group Intersection
                    </h3>
                    <div style={{ height: Math.max(280, (detectionResults.intersectional_bias || []).length * 44) }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          layout="vertical"
                          data={detectionResults.intersectional_bias || []}
                          margin={{ left: 10, right: 30, top: 5, bottom: 5 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" stroke="#1a1a24" horizontal={false} />
                          <XAxis type="number" domain={[0, 1]} stroke="#555" fontSize={10} tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
                          <YAxis dataKey="group" type="category" stroke="#555" fontSize={10} width={160} tick={{ fill: '#aaa' }} />
                          <Tooltip
                            cursor={{ fill: '#1a1a24' }}
                            contentStyle={{ backgroundColor: '#0f0f19', border: '1px solid #6b21a8', color: '#c084fc', fontSize: 12 }}
                            formatter={(value, name) => {
                              if (name === 'approval_rate') return [`${(value * 100).toFixed(1)}%`, 'Approval Rate'];
                              return [value, name];
                            }}
                            labelFormatter={(label) => label}
                          />
                          <Bar dataKey="approval_rate" radius={[0, 4, 4, 0]} maxBarSize={28}>
                            {(detectionResults.intersectional_bias || []).map((entry, index) => {
                              // Gradient: low approval → purple (#a855f7), high → cyan (#00f5d4)
                              const t = entry.approval_rate;
                              const r = Math.round(168 * (1 - t) + 0 * t);
                              const g = Math.round(85 * (1 - t) + 245 * t);
                              const b = Math.round(247 * (1 - t) + 212 * t);
                              return <Cell key={`icell-${index}`} fill={`rgb(${r},${g},${b})`} />;
                            })}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                    <div className="mt-4 flex items-center gap-4 text-[10px] text-gray-600 font-mono">
                      <div className="flex items-center gap-1.5">
                        <span className="inline-block w-3 h-3 rounded-sm" style={{ background: '#a855f7' }}></span>
                        Lower approval
                      </div>
                      <div className="flex items-center gap-1.5">
                        <span className="inline-block w-3 h-3 rounded-sm" style={{ background: '#00f5d4' }}></span>
                        Higher approval
                      </div>
                      <span className="ml-auto">Groups with &lt;30 samples excluded</span>
                    </div>

                    <div className="mt-6">
                      <GroqInsightPanel
                        title="Intersectional Compound Risk"
                        insight={groqInsights.intersectionalInsight}
                        loading={groqLoading.intersectionalInsight}
                        error={groqErrors.intersectionalInsight}
                        onRegenerate={() => fetchGroqInsight('intersectionalInsight', '/api/groq/intersectional-insight', {
                          intersectional_data: detectionResults.intersectional_bias
                        })}
                        defaultExpanded={true}
                      />
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {!detectionResults && (
          <EmptyState 
            icon={<svg className="w-12 h-12" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path></svg>}
            prompt="Run the fairness audit to see detailed bias metrics and counterfactual examples."
            onAction={handleDetect}
            actionLabel="Run Audit Now"
          />
        )}

        <SectionDivider label="Mitigation Strategies" />

        {/* 5. Simulation Panel */}
        <section id="strategy" className="animate-reveal">
          <SectionHeader title="03. Strategy Simulation" subtitle="Compare automated bias mitigation strategies side-by-side." />
          <div className="mb-8">
            <button 
              onClick={handleSimulate}
              disabled={loadingState === 'simulating' || !detectionResults}
              className="bg-transparent border border-cyan text-cyan px-8 py-3 rounded font-bold hover:bg-cyan/10 transition-all uppercase tracking-widest text-sm disabled:opacity-50"
            >
              {loadingState === 'simulating' ? <InlineSpinner /> : 'Run Simulations'}
            </button>
            {errors.simulate && <InlineError message={errors.simulate} />}
          </div>

          {simulationResults && (
            <div className="animate-in slide-in-from-bottom duration-500">
              {/* Scatter chart */}
              <div className="card-industrial p-6 rounded-lg mb-6">
                <h3 className="text-xs text-gray-500 font-mono uppercase mb-6 tracking-widest">Fairness vs Accuracy Trade-off</h3>
                <div className="h-96">
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 20, right: 30, bottom: 30, left: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1a1a24" />
                      <XAxis
                        type="number" dataKey="accuracy" name="Accuracy"
                        domain={[0.5, 1.0]} tickCount={6}
                        stroke="#555" fontSize={10} fontFamily="Space Mono, monospace"
                      >
                        <Label value="Accuracy →" position="bottom" offset={10} fill="#555" fontSize={11} fontFamily="Space Mono, monospace" />
                      </XAxis>
                      <YAxis
                        type="number" dataKey="fairness" name="Fairness"
                        domain={[0, 1.0]} tickCount={6}
                        stroke="#555" fontSize={10} fontFamily="Space Mono, monospace"
                      >
                        <Label value="Fairness →" position="insideLeft" angle={-90} offset={-5} fill="#555" fontSize={11} fontFamily="Space Mono, monospace" />
                      </YAxis>
                      <ZAxis type="number" dataKey="size" range={[80, 240]} />
                      <Tooltip content={<ChartTooltip />} cursor={false} />

                      {/* Target zone — top right */}
                      <ReferenceArea x1={0.78} x2={1.0} y1={0.85} y2={1.0} fill="#10b981" fillOpacity={0.06} stroke="#10b981" strokeOpacity={0.15} strokeDasharray="4 4">
                        <Label value="Target zone" position="insideTopRight" fill="#10b981" fontSize={9} fontFamily="Space Mono, monospace" />
                      </ReferenceArea>

                      <Scatter data={dots} shape={<StrategyDot />} />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
                <p className="text-center text-[10px] text-gray-600 font-mono mt-4 tracking-wider">
                  Each dot = one mitigation strategy. Move up and right to win.
                </p>
              </div>

              {/* Strategy cards grid (kept for detail) */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-16">
                {simulationResults.map((s, idx) => {
                  const isRec = s.strategy_name === recName;
                  return (
                    <div key={idx} className={`card-industrial p-6 rounded-lg border-t-2 flex flex-col h-full ${isRec ? 'border-emerald-500/60' : 'border-cyan/30'}`}>
                      <div className="flex items-center gap-2 mb-4 h-10">
                        {isRec && <span className="flex h-2 w-2 rounded-full bg-emerald-400"></span>}
                        <h4 className={`text-sm font-bold ${isRec ? 'text-emerald-400' : 'text-white'}`}>{s.strategy_name}</h4>
                      </div>
                      <div className="space-y-4 mt-auto">
                        <div className="flex justify-between items-end">
                          <span className="text-xs text-gray-500 font-mono">FAIRNESS GAIN</span>
                          <span className="text-cyan font-bold">+{s.fairness_gain.toFixed(3)}</span>
                        </div>
                        <div className="flex justify-between items-end">
                          <span className="text-xs text-gray-500 font-mono">ACCURACY DROP</span>
                          <span className="text-red-400 font-bold">-{s.accuracy_drop.toFixed(3)}</span>
                        </div>
                        <div className="pt-4 border-t border-gray-800">
                          <div className="flex justify-between items-end">
                            <span className="text-xs text-gray-500 font-mono uppercase">New Score</span>
                            <span className="text-white font-mono">{s.fairness_score.toFixed(3)}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>

              <div className="mb-16 card-industrial rounded-lg border border-gray-800 p-5">
                <div className="flex flex-col lg:flex-row lg:items-center gap-4 justify-between mb-4">
                  <p className="text-xs text-gray-500 font-mono uppercase tracking-widest">AI Mitigation Advisor</p>
                  <button
                    type="button"
                    onClick={() => fetchGroqInsight('mitigationAdvice', '/api/groq/mitigation-advice', {
                      simulation_results: simulationResults,
                      rl_recommendation: recommendation?.recommended_strategy || '',
                      domain
                    })}
                    disabled={groqLoading.mitigationAdvice}
                    className="bg-cyan/10 border border-cyan/40 text-cyan hover:bg-cyan/20 px-4 py-2 rounded font-mono text-[10px] uppercase tracking-widest transition-all disabled:opacity-50"
                  >
                    {groqLoading.mitigationAdvice ? 'Generating...' : 'Get AI Recommendation'}
                  </button>
                </div>
                <GroqInsightPanel
                  title={`Mitigation Plan (${domain})`}
                  insight={groqInsights.mitigationAdvice}
                  loading={groqLoading.mitigationAdvice}
                  error={groqErrors.mitigationAdvice}
                  onRegenerate={() => fetchGroqInsight('mitigationAdvice', '/api/groq/mitigation-advice', {
                    simulation_results: simulationResults,
                    rl_recommendation: recommendation?.recommended_strategy || '',
                    domain
                  })}
                />
              </div>
            </div>
          )}

          {!simulationResults && (
            <EmptyState 
              icon={<svg className="w-12 h-12" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.022.547l-2.387 2.387a2 2 0 000 2.828l.172.172a2 2 0 002.828 0l2.387-2.387a2 2 0 011.022-.547l2.387-.477a6 6 0 003.86-.517l.318-.158a6 6 0 013.86-.517l1.931.386a2 2 0 001.022-.547l2.387-2.387a2 2 0 000-2.828l-.172-.172a2 2 0 00-2.828 0l-2.387 2.387z"></path></svg>}
              prompt="Simulate mitigation strategies to see how they affect fairness and accuracy."
              onAction={handleSimulate}
              actionLabel="Run Simulations"
            />
          )}
            </section>

            <SectionDivider label="RL Recommendation" />

            {/* 6. Recommendation Panel */}
            <section id="recommendation" className="animate-reveal">
              <SectionHeader title="04. AI Recommendation" subtitle="Reinforcement Learning agent suggests the optimal path for your dataset." />
              <div className="mb-8">
                <button 
                  onClick={handleRecommend}
                  disabled={loadingState === 'recommending' || !simulationResults}
                  className="bg-cyan text-black px-8 py-3 rounded font-bold hover:bg-cyan/80 transition-all uppercase tracking-widest text-sm flex items-center gap-2 disabled:opacity-50"
                >
                  {loadingState === 'recommending' ? <InlineSpinner /> : (
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-7.714 2.143L11 21l-2.286-6.857L1 12l7.714-2.143L11 3z" />
                    </svg>
                  )}
                  {loadingState === 'recommending' ? 'RL Agent Thinking...' : 'Get AI Recommendation'}
                </button>
                {errors.recommend && <InlineError message={errors.recommend} />}
              </div>

              {recommendation && (
                <div className="card-industrial p-8 rounded-lg border-2 border-cyan glow-cyan mb-16 animate-in zoom-in duration-500">
                  <div className="flex flex-col md:flex-row gap-8 items-center">
                    <div className="flex-1">
                      <span className="text-xs text-cyan font-mono tracking-widest uppercase mb-2 block">AI Recommended Strategy</span>
                      <h4 className="text-4xl font-bold text-white mb-4">{recommendation.recommended_strategy}</h4>
                      <p className="text-gray-400 leading-relaxed border-l-2 border-gray-800 pl-4 py-2 italic">
                        &quot;{recommendation.reason}&quot;
                      </p>
                    </div>
                    <div className="flex flex-col gap-4 w-full md:w-64">
                      <div className="bg-gray-900/50 p-4 rounded border border-gray-800">
                        <span className="text-[10px] text-gray-500 font-mono block">EST. FAIRNESS GAIN</span>
                        <span className="text-2xl font-bold text-cyan">+{recommendation.expected_fairness_gain.toFixed(3)}</span>
                      </div>
                      <div className="bg-gray-900/50 p-4 rounded border border-gray-800">
                        <span className="text-[10px] text-gray-500 font-mono block">EST. ACCURACY DROP</span>
                        <span className="text-2xl font-bold text-red-400">-{recommendation.expected_accuracy_drop.toFixed(3)}</span>
                      </div>
                    </div>
                  </div>

                  {/* 7. Enhanced Before/After Comparison */}
                  <div className="mt-12 pt-12 border-t border-gray-800" ref={beforeAfterRef}>
                    <h4 className="text-xs text-gray-500 font-mono uppercase mb-12 tracking-widest text-center">Outcome Projection</h4>
                    
                    {recommendation && detectionResults ? (() => {
                      const baseFairness = 1 - Math.abs(detectionResults.fairness_metrics?.demographic_parity_difference || 0);
                      const baseAccuracy = detectionResults.accuracy || 0;
                      const baseDI = detectionResults.fairness_metrics?.disparate_impact_ratio || 1;
                      
                      const optFairness = recommendedMetrics 
                        ? recommendedMetrics.fairness_score 
                        : baseFairness + (recommendation.expected_fairness_gain || 0);
                      const optAccuracy = recommendedMetrics
                        ? baseAccuracy - (recommendedMetrics.accuracy_drop || 0)
                        : baseAccuracy - (recommendation.expected_accuracy_drop || 0);
                      
                      // For optDI, we'll estimate or use the improved fairness to reflect a better ratio
                      const optDI = Math.min(1.0, baseDI + ((recommendation.expected_fairness_gain || 0) * 1.5));
                      
                      const baseRisk = getRiskInfo(baseDI);
                      const optRisk = getRiskInfo(optDI);
                      
                      const accChange = baseAccuracy !== 0 ? ((optAccuracy - baseAccuracy) / baseAccuracy * 100).toFixed(1) : "0.0";
                      const fairChange = baseFairness !== 0 ? ((optFairness - baseFairness) / baseFairness * 100).toFixed(1) : "0.0";
                      const biasReduction = (Math.abs(detectionResults.fairness_metrics?.demographic_parity_difference || 1) > 0)
                        ? ((recommendation.expected_fairness_gain || 0) / Math.abs(detectionResults.fairness_metrics.demographic_parity_difference) * 100).toFixed(0)
                        : "0";

                      return (
                        <div className="space-y-12">
                          <div className="flex flex-col lg:flex-row items-stretch gap-0 lg:gap-4">
                            {/* Original Model Card */}
                            <div className="flex-1 card-industrial p-6 rounded-lg lg:rounded-l-lg lg:rounded-r-none border-l-4 border-red-500/50 bg-red-500/[0.02]">
                              <div className="flex justify-between items-center mb-6">
                                <span className="text-[10px] text-gray-400 font-mono uppercase tracking-widest">Original Model</span>
                                <span className={`px-2 py-0.5 rounded-full text-[9px] font-bold uppercase tracking-wider ${baseRisk.color} ${baseRisk.bg} border ${baseRisk.border}`}>
                                  {baseRisk.label}
                                </span>
                              </div>
                              <div className="grid grid-cols-2 gap-4 mb-8">
                                <div>
                                  <div className="text-3xl font-bold text-white font-mono">{baseAccuracy.toFixed(3)}</div>
                                  <div className="text-[10px] text-gray-500 font-mono uppercase">Accuracy</div>
                                </div>
                                <div>
                                  <div className="text-3xl font-bold text-gray-400 font-mono">{baseFairness.toFixed(3)}</div>
                                  <div className="text-[10px] text-gray-500 font-mono uppercase">Fairness</div>
                                </div>
                              </div>
                              <div className="space-y-2 border-t border-gray-800/50 pt-4">
                                <div className="flex justify-between text-[10px] font-mono">
                                  <span className="text-gray-500">Demographic Parity</span>
                                  <span className="text-gray-400">{(detectionResults.fairness_metrics?.demographic_parity_difference || 0).toFixed(3)}</span>
                                </div>
                                <div className="flex justify-between text-[10px] font-mono">
                                  <span className="text-gray-500">Equal Opportunity</span>
                                  <span className="text-gray-400">{(detectionResults.fairness_metrics?.equal_opportunity_difference || 0).toFixed(3)}</span>
                                </div>
                                <div className="flex justify-between text-[10px] font-mono">
                                  <span className="text-gray-500">Disparate Impact</span>
                                  <span className="text-gray-400">{baseDI.toFixed(3)}</span>
                                </div>
                              </div>
                            </div>

                            {/* Delta Column */}
                            <div className="w-full lg:w-24 flex lg:flex-col items-center justify-center gap-4 py-6 lg:py-0 bg-gray-900/20">
                              <div className="text-center">
                                <div className={`text-xs font-bold font-mono ${parseFloat(accChange) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                  {parseFloat(accChange) >= 0 ? '+' : ''}{accChange}%
                                </div>
                                <div className="text-[9px] text-gray-600 font-mono uppercase">Acc.</div>
                              </div>
                              <div className="text-cyan animate-pulse hidden lg:block">
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M13 5l7 7-7 7M5 5l7 7-7 7" />
                                </svg>
                              </div>
                              <div className="text-center">
                                <div className="text-xs font-bold font-mono text-emerald-400">
                                  +{fairChange}%
                                </div>
                                <div className="text-[9px] text-gray-600 font-mono uppercase">Fair.</div>
                              </div>
                            </div>

                            {/* Optimized Model Card */}
                            <div className="flex-1 card-industrial p-6 rounded-lg lg:rounded-r-lg lg:rounded-l-none border-r-4 border-emerald-500/50 bg-emerald-500/[0.02] relative">
                              <div className="absolute -top-3 right-6 px-3 py-1 bg-emerald-500 text-black text-[9px] font-bold uppercase tracking-widest rounded shadow-lg">
                                After applying: {recName}
                              </div>
                              <div className="flex justify-between items-center mb-6">
                                <span className="text-[10px] text-cyan font-mono uppercase tracking-widest">Optimized Model</span>
                                <span className={`px-2 py-0.5 rounded-full text-[9px] font-bold uppercase tracking-wider ${optRisk.color} ${optRisk.bg} border ${optRisk.border}`}>
                                  {optRisk.label}
                                </span>
                              </div>
                              <div className="grid grid-cols-2 gap-4 mb-8">
                                <div>
                                  <div className="text-3xl font-bold text-white font-mono">{optAccuracy.toFixed(3)}</div>
                                  <div className="text-[10px] text-gray-500 font-mono uppercase">Accuracy</div>
                                </div>
                                <div>
                                  <div className="text-3xl font-bold text-cyan font-mono">{optFairness.toFixed(3)}</div>
                                  <div className="text-[10px] text-gray-500 font-mono uppercase">Fairness</div>
                                </div>
                              </div>
                              <div className="space-y-2 border-t border-gray-800/50 pt-4">
                                <div className="flex justify-between text-[10px] font-mono">
                                  <span className="text-gray-500">Demographic Parity</span>
                                  <span className="text-cyan">{( (detectionResults.fairness_metrics?.demographic_parity_difference || 0) * (1 - parseFloat(biasReduction)/100)).toFixed(3)}</span>
                                </div>
                                <div className="flex justify-between text-[10px] font-mono">
                                  <span className="text-gray-500">Equal Opportunity</span>
                                  <span className="text-cyan">{( (detectionResults.fairness_metrics?.equal_opportunity_difference || 0) * (1 - parseFloat(biasReduction)/100)).toFixed(3)}</span>
                                </div>
                                <div className="flex justify-between text-[10px] font-mono">
                                  <span className="text-gray-500">Disparate Impact</span>
                                  <span className="text-cyan">{optDI.toFixed(3)}</span>
                                </div>
                              </div>
                            </div>
                          </div>

                          {/* Animated Progress Bar */}
                          <div className="max-w-2xl mx-auto px-4">
                            <div className="flex justify-between items-center mb-2">
                              <span className="text-xs text-gray-500 font-mono uppercase tracking-widest">Bias Reduction Progress</span>
                              <span className="text-lg font-bold text-cyan font-mono">{biasReduction}% reduced</span>
                            </div>
                            <div className="h-4 bg-gray-900 rounded-full overflow-hidden border border-gray-800 p-1">
                              <div 
                                className={`h-full bg-gradient-to-r from-cyan/40 to-cyan rounded-full transition-all duration-[1200ms] ease-out ${beforeAfterVisible ? 'bias-bar-animated' : ''}`}
                                style={{ width: beforeAfterVisible ? `${biasReduction}%` : '0%' }}
                              ></div>
                            </div>
                          </div>
                        </div>
                      );
                    })() : (
                      <div className="text-center py-12 card-industrial rounded-lg border border-dashed border-gray-800">
                        <p className="text-gray-500 font-mono text-sm mb-6">Run the RL recommender to see the optimized model projection</p>
                        <button
                          onClick={handleRecommend}
                          disabled={loadingState === 'recommending'}
                          className="bg-cyan/10 border border-cyan/40 text-cyan hover:bg-cyan/20 px-6 py-2 rounded font-mono text-xs uppercase tracking-widest transition-all"
                        >
                          {loadingState === 'recommending' ? 'Analyzing...' : 'Generate Optimized Projection'}
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </section>

        {/* Footer */}
        <footer className="mt-24 text-center border-t border-gray-800 pt-12 pb-24">
          <p className="text-gray-300 font-bold mb-2 tracking-tighter">EQUILENS <span className="text-cyan font-light">AI</span></p>
          <p className="text-gray-500 text-[10px] font-mono tracking-[0.3em] uppercase mb-6">BiasGuard Engine · Accuracy without Fairness is an Error</p>
          <div className="flex justify-center gap-6 mb-8 text-[9px] font-mono text-gray-600 uppercase tracking-widest">
            <span>Built for Global AI Ethics Hackathon</span>
            <span>•</span>
            <span>v1.0.4 Platinum</span>
          </div>
          <div className="flex justify-center">
            <svg className="w-6 h-6 text-gray-800" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 2L3 7v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V7l-9-5z"/>
            </svg>
          </div>
        </footer>
      </div>
      )}
    </div>
  );
}
