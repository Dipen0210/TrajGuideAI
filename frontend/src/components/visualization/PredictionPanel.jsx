import { useState } from 'react';
import { predictTrajectory } from '../../services/api';

function PredictionPanel({ sequence, prediction, onPrediction }) {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handlePredict = async () => {
        setLoading(true);
        setError(null);
        try {
            const result = await predictTrajectory(sequence);
            onPrediction(result);
        } catch (err) {
            setError(err.message || 'Prediction failed');
        } finally {
            setLoading(false);
        }
    };

    const nextPoint = prediction?.trajectory?.[0];

    return (
        <div className="card">
            <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', flexWrap: 'wrap' }}>
                <button
                    className="btn btn-primary"
                    onClick={handlePredict}
                    disabled={loading || !sequence.length}
                    style={{ flex: '0 0 200px' }}
                >
                    {loading ? '⏳ Predicting...' : '🔮 Predict Next Steps'}
                </button>

                {prediction && (
                    <>
                        <div className="stat-card" style={{ flex: 1, minWidth: '150px' }}>
                            <div className="stat-label">Next X Position</div>
                            <div className="stat-value">{nextPoint?.predicted_local_x?.toFixed(2) || '-'}</div>
                        </div>
                        <div className="stat-card" style={{ flex: 1, minWidth: '150px' }}>
                            <div className="stat-label">Next Y Position</div>
                            <div className="stat-value">{nextPoint?.predicted_local_y?.toFixed(2) || '-'}</div>
                        </div>
                        <div className="stat-card" style={{ flex: 1, minWidth: '150px' }}>
                            <div className="stat-label">Prediction Steps</div>
                            <div className="stat-value">{prediction.trajectory?.length || 0}</div>
                        </div>
                    </>
                )}
            </div>

            {error && (
                <p style={{ color: 'var(--color-danger)', marginTop: '1rem' }}>
                    ❌ {error}
                </p>
            )}
        </div>
    );
}

export default PredictionPanel;
