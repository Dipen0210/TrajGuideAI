function Sidebar({ sequence, onReset }) {
    const avgVelocity = sequence.length > 0
        ? (sequence.reduce((sum, s) => sum + (s.v_Vel || 0), 0) / sequence.length).toFixed(2)
        : '0.00';

    return (
        <aside className="sidebar">
            <div className="sidebar-logo">
                <h1>🚗 TrajAI</h1>
                <p>Agentic Vehicle Intelligence</p>
            </div>

            <div className="sidebar-section">
                <h3>Quick Stats</h3>
                {sequence.length > 0 ? (
                    <>
                        <div className="stat-card">
                            <div className="stat-label">Data Points</div>
                            <div className="stat-value">{sequence.length}</div>
                        </div>
                        <div className="stat-card">
                            <div className="stat-label">Avg Velocity</div>
                            <div className="stat-value">{avgVelocity}</div>
                        </div>
                    </>
                ) : (
                    <div className="stat-card">
                        <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                            No data loaded yet
                        </p>
                    </div>
                )}
            </div>

            <div style={{ marginTop: 'auto', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                <button className="btn btn-reset" onClick={onReset} style={{ width: '100%' }}>
                    🔄 Reset All
                </button>

                <div style={{ textAlign: 'center' }}>
                    <p style={{ color: 'var(--text-muted)', fontSize: '0.7rem', lineHeight: 1.6 }}>
                        Powered by<br />
                        <span style={{ color: 'var(--color-primary)' }}>LSTM</span> + <span style={{ color: 'var(--color-secondary)' }}>Llama 3</span>
                    </p>
                </div>
            </div>
        </aside>
    );
}

export default Sidebar;
