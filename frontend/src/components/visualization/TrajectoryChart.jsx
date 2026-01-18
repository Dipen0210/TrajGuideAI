import { Line } from 'react-chartjs-2';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
} from 'chart.js';

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend
);

function TrajectoryChart({ sequence, prediction }) {
    const historyX = sequence.map(s => s.Local_X);
    const historyY = sequence.map(s => s.Local_Y);

    const predX = prediction?.trajectory?.map(p => p.predicted_local_x) || [];
    const predY = prediction?.trajectory?.map(p => p.predicted_local_y) || [];

    const data = {
        datasets: [
            {
                label: 'Trajectory',
                data: historyX.map((x, i) => ({ x, y: historyY[i] })),
                borderColor: '#4299e1',
                backgroundColor: '#4299e1',
                pointRadius: 4,
                pointHoverRadius: 6,
                showLine: true,
                tension: 0.1,
            },
            {
                label: 'Predicted Path',
                data: predX.map((x, i) => ({ x, y: predY[i] })),
                borderColor: '#f56565',
                backgroundColor: '#f56565',
                borderDash: [5, 5],
                pointRadius: 6,
                pointStyle: 'crossRot',
                showLine: true,
                tension: 0.1,
            },
        ],
    };

    const options = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: 'top',
                labels: {
                    color: '#a0aec0',
                    font: { size: 12 },
                },
            },
            title: {
                display: true,
                text: 'Vehicle Trajectory',
                color: '#f7fafc',
                font: { size: 18, weight: 'bold' },
            },
        },
        scales: {
            x: {
                type: 'linear',
                title: {
                    display: true,
                    text: 'Local_X',
                    color: '#a0aec0',
                },
                grid: { color: 'rgba(255, 255, 255, 0.1)' },
                ticks: { color: '#a0aec0' },
            },
            y: {
                type: 'linear',
                title: {
                    display: true,
                    text: 'Local_Y',
                    color: '#a0aec0',
                },
                grid: { color: 'rgba(255, 255, 255, 0.1)' },
                ticks: { color: '#a0aec0' },
            },
        },
    };

    return (
        <div className="card">
            <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '1rem' }}>
                <div className="chart-container" style={{ height: '400px' }}>
                    <Line data={data} options={options} />
                </div>

                <div>
                    <h4 style={{ marginBottom: '1rem' }}>Forecast Data</h4>
                    <table className="data-table">
                        <thead>
                            <tr>
                                <th>Step</th>
                                <th>X</th>
                                <th>Y</th>
                            </tr>
                        </thead>
                        <tbody>
                            {prediction?.trajectory?.map((p, i) => (
                                <tr key={i}>
                                    <td>+{i + 1}</td>
                                    <td>{p.predicted_local_x?.toFixed(3)}</td>
                                    <td>{p.predicted_local_y?.toFixed(3)}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
}

export default TrajectoryChart;
