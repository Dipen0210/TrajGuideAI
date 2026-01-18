import { useRef, useState } from 'react';
import Papa from 'papaparse';

function FileUpload({ onDataLoaded }) {
    const [isDragging, setIsDragging] = useState(false);
    const [fileName, setFileName] = useState(null);
    const fileInputRef = useRef(null);

    const handleFile = (file) => {
        if (!file) return;

        Papa.parse(file, {
            header: true,
            dynamicTyping: true,
            complete: (results) => {
                if (results.data && results.data.length > 0) {
                    // Filter out empty rows
                    const validData = results.data.filter(row =>
                        row.Local_X !== undefined && row.Local_X !== null
                    );
                    onDataLoaded(validData);
                    setFileName(file.name);
                }
            },
            error: (error) => {
                console.error('CSV parsing error:', error);
                alert('Failed to parse CSV file');
            }
        });
    };

    const handleDrop = (e) => {
        e.preventDefault();
        setIsDragging(false);
        const file = e.dataTransfer.files[0];
        if (file && file.name.endsWith('.csv')) {
            handleFile(file);
        }
    };

    const handleDragOver = (e) => {
        e.preventDefault();
        setIsDragging(true);
    };

    const handleDragLeave = () => {
        setIsDragging(false);
    };

    const handleClick = () => {
        fileInputRef.current?.click();
    };

    const handleInputChange = (e) => {
        const file = e.target.files[0];
        if (file) {
            handleFile(file);
        }
    };

    return (
        <div className="card">
            <div
                className={`file-upload ${isDragging ? 'dragging' : ''}`}
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onClick={handleClick}
            >
                <input
                    ref={fileInputRef}
                    type="file"
                    accept=".csv"
                    onChange={handleInputChange}
                    style={{ display: 'none' }}
                />
                <div className="file-upload-icon">📤</div>
                {fileName ? (
                    <>
                        <p style={{ color: 'var(--color-success)', fontWeight: '600' }}>
                            ✅ {fileName} loaded
                        </p>
                        <p style={{ color: 'var(--color-text-secondary)', fontSize: '0.9rem' }}>
                            Click to upload a different file
                        </p>
                    </>
                ) : (
                    <>
                        <p>Drag and drop a CSV file here</p>
                        <p style={{ color: 'var(--color-text-secondary)', fontSize: '0.9rem' }}>
                            or click to browse
                        </p>
                    </>
                )}
            </div>
        </div>
    );
}

export default FileUpload;
