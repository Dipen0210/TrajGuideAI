import { useState } from 'react';
import { predictTrajectory, runDriverProfile } from '../../services/api';

function DriverProfiler({ sequence, prediction, result, onResult }) {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleProfile = async () => {
        setLoading(true);
        setError(null);
        try {
            let predTraj = prediction?.trajectory || [];
            if (predTraj.length === 0) {
                const pred = await predictTrajectory(sequence);
                predTraj = pred.trajectory || [];
            }

            const profileResult = await runDriverProfile(sequence, predTraj);
            onResult(profileResult);
        } catch (err) {
            setError(err.message || 'Driver profiling failed');
        } finally {
            setLoading(false);
        }
    };

    const classification = result?.classification;
    const confidence = result?.confidence || 0;
    const recommendations = result?.recommendations || [];

    const classStyles = {
        Aggressive: { emoji: '🔴', color: '#ef4444', class: 'aggressive' },
        Defensive: { emoji: '🟢', color: '#22c55e', class: 'defensive' },
        Normal: { emoji: '🔵', color: '#3b82f6', class: 'normal' },
        Distracted: { emoji: '🟡', color: '#f59e0b', class: 'distracted' },
    };

    const style = classStyles[classification] || { emoji: '⚪', color: '#94a3b8', class: '' };

    return (
        <div className="card" style={{ borderTop: '3px solid var(--color-primary)' }}>
            <div className="card-header">
                <span className="card-icon">🏎️</span>
                <div>
                    <div className="card-title">Driver Profiler</div>
                    <div className="card-description">
                        Behavioral analysis and style classification with personalized insights
                    </div>
                </div>
            </div>

            <button
                className="btn btn-primary"
                onClick={handleProfile}
                disabled={loading || !sequence.length}
                style={{ width: '100%' }}
            >
                {loading ? (
                    <>
                        <span className="spinner" style={{ width: 16, height: 16, borderWidth: 2 }}></span>
                        Analyzing...
                    </>
                ) : (
                    '📊 Analyze Driver Style'
                )}
            </button>

            {error && (
                <p style={{ color: 'var(--color-danger)', marginTop: '1rem', fontSize: '0.9rem' }}>
                    ❌ {error}
                </p>
            )}

            {result && (
                <div style={{ marginTop: '1.5rem' }}>
                    <div className={`classification-badge ${style.class}`}>
                        <span style={{ fontSize: '2.5rem', display: 'block' }}>{style.emoji}</span>
                        <h3 style={{ color: style.color, margin: '0.75rem 0 0.25rem', fontSize: '1.5rem', fontWeight: 700 }}>
                            {classification}
                        </h3>
                        <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                            {confidence}% Confidence
                        </p>
                    </div>

                    <div className="progress-bar">
                        <div className="progress-fill" style={{ width: `${confidence}%` }}></div>
                    </div>

                    {recommendations.length > 0 && (
                        <div style={{ marginTop: '1.5rem' }}>
                            <p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', marginBottom: '0.75rem', textTransform: 'uppercase', letterSpacing: '1px' }}>
                                💡 Recommendations
                            </p>
                            {recommendations.slice(0, 3).map((rec, i) => (
                                <div key={i} className="recommendation-item">
                                    {i + 1}. {rec}
                                </div>
                            ))}
                        </div>
                    )}

                    {result.report && (
                        <details style={{ marginTop: '1.25rem' }}>
                            <summary style={{ cursor: 'pointer', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                                📋 View Full Report
                            </summary>
                            <div style={{
                                marginTop: '0.75rem',
                                padding: '1rem',
                                background: 'var(--bg-glass)',
                                border: '1px solid var(--border-color)',
                                borderRadius: '12px',
                                fontSize: '0.85rem',
                                whiteSpace: 'pre-wrap',
                                color: 'var(--text-secondary)'
                            }}>
                                {result.report}
                            </div>
                        </details>
                    )}
                </div>
            )}
        </div>
    );
}

export default DriverProfiler;
