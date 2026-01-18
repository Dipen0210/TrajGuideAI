import { useState } from 'react';
import { predictTrajectory, runSafetyAudit } from '../../services/api';

function SafetyAuditor({ sequence, prediction, result, onResult }) {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleAudit = async () => {
        setLoading(true);
        setError(null);
        try {
            let predTraj = prediction?.trajectory || [];
            if (predTraj.length === 0) {
                const pred = await predictTrajectory(sequence);
                predTraj = pred.trajectory || [];
            }

            const auditResult = await runSafetyAudit(sequence, predTraj);
            onResult(auditResult);
        } catch (err) {
            setError(err.message || 'Safety audit failed');
        } finally {
            setLoading(false);
        }
    };

    const status = result?.status?.toUpperCase();
    const violations = result?.violations?.filter(v => v && v.length > 2) || [];

    return (
        <div className="card" style={{ borderTop: '3px solid var(--color-success)' }}>
            <div className="card-header">
                <span className="card-icon">🛡️</span>
                <div>
                    <div className="card-title">Safety Auditor</div>
                    <div className="card-description">
                        Real-time trajectory analysis using Chain-of-Thought reasoning
                    </div>
                </div>
            </div>

            <button
                className="btn btn-primary"
                onClick={handleAudit}
                disabled={loading || !sequence.length}
                style={{ width: '100%' }}
            >
                {loading ? (
                    <>
                        <span className="spinner" style={{ width: 16, height: 16, borderWidth: 2 }}></span>
                        Analyzing...
                    </>
                ) : (
                    '🔍 Run Safety Audit'
                )}
            </button>

            {error && (
                <p style={{ color: 'var(--color-danger)', marginTop: '1rem', fontSize: '0.9rem' }}>
                    ❌ {error}
                </p>
            )}

            {result && (
                <div style={{ marginTop: '1.5rem' }}>
                    {status === 'SAFE' && (
                        <div className="status-badge status-safe">
                            ✅ SAFE — No violations detected
                        </div>
                    )}
                    {status === 'WARNING' && (
                        <div className="status-badge status-warning">
                            ⚠️ WARNING — Potential issues detected
                        </div>
                    )}
                    {status === 'CRITICAL' && (
                        <div className="status-badge status-critical">
                            🚨 CRITICAL — Immediate action required
                        </div>
                    )}

                    {status !== 'SAFE' && violations.length > 0 && (
                        <div style={{ marginTop: '1.25rem' }}>
                            <p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', marginBottom: '0.75rem', textTransform: 'uppercase', letterSpacing: '1px' }}>
                                Detected Violations
                            </p>
                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                                {violations.map((v, i) => (
                                    <span key={i} className="violation-tag">❌ {v}</span>
                                ))}
                            </div>
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

export default SafetyAuditor;
