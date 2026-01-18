import './styles/global.css';
import { useState } from 'react';
import Sidebar from './components/common/Sidebar';
import Header from './components/common/Header';
import FileUpload from './components/data/FileUpload';
import DataPreview from './components/data/DataPreview';
import TrajectoryChart from './components/visualization/TrajectoryChart';
import PredictionPanel from './components/visualization/PredictionPanel';
import SafetyAuditor from './components/analysis/SafetyAuditor';
import DriverProfiler from './components/analysis/DriverProfiler';

function App() {
  const [sequence, setSequence] = useState([]);
  const [prediction, setPrediction] = useState(null);
  const [safetyAudit, setSafetyAudit] = useState(null);
  const [driverProfile, setDriverProfile] = useState(null);

  const handleReset = () => {
    setSequence([]);
    setPrediction(null);
    setSafetyAudit(null);
    setDriverProfile(null);
  };

  return (
    <div className="app-container">
      <Sidebar
        sequence={sequence}
        onReset={handleReset}
      />

      <main className="main-content">
        <Header />

        {/* Data Upload Section */}
        <div className="section-header">
          <h2>📁 Data Upload</h2>
        </div>

        <div className="grid-2">
          <FileUpload onDataLoaded={setSequence} />
          <DataPreview data={sequence} />
        </div>

        {/* Trajectory Prediction Section */}
        {sequence.length > 0 && (
          <>
            <div className="section-header">
              <h2>🎯 Trajectory Prediction</h2>
            </div>

            <PredictionPanel
              sequence={sequence}
              prediction={prediction}
              onPrediction={setPrediction}
            />

            {prediction && (
              <TrajectoryChart
                sequence={sequence}
                prediction={prediction}
              />
            )}
          </>
        )}

        {/* AI Agent Analysis Section */}
        {sequence.length > 0 && (
          <>
            <div className="section-header">
              <h2>🤖 AI Agent Analysis</h2>
            </div>

            <div className="grid-2">
              <SafetyAuditor
                sequence={sequence}
                prediction={prediction}
                result={safetyAudit}
                onResult={setSafetyAudit}
              />
              <DriverProfiler
                sequence={sequence}
                prediction={prediction}
                result={driverProfile}
                onResult={setDriverProfile}
              />
            </div>
          </>
        )}
      </main>
    </div>
  );
}

export default App;
