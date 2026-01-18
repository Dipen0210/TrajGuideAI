function DataPreview({ data }) {
    if (!data || data.length === 0) {
        return (
            <div className="card">
                <div style={{
                    textAlign: 'center',
                    padding: '3rem',
                    border: '2px dashed rgba(102, 126, 234, 0.3)',
                    borderRadius: '12px'
                }}>
                    <p style={{ color: 'var(--color-text-secondary)' }}>
                        Upload a CSV file to see data preview
                    </p>
                </div>
            </div>
        );
    }

    const columns = Object.keys(data[0]);
    const previewData = data.slice(0, 10);

    return (
        <div className="card">
            <h3 style={{ marginBottom: '1rem' }}>Data Preview</h3>
            <div style={{ overflowX: 'auto' }}>
                <table className="data-table">
                    <thead>
                        <tr>
                            {columns.slice(0, 6).map((col) => (
                                <th key={col}>{col}</th>
                            ))}
                            {columns.length > 6 && <th>...</th>}
                        </tr>
                    </thead>
                    <tbody>
                        {previewData.map((row, idx) => (
                            <tr key={idx}>
                                {columns.slice(0, 6).map((col) => (
                                    <td key={col}>
                                        {typeof row[col] === 'number' ? row[col].toFixed(2) : row[col]}
                                    </td>
                                ))}
                                {columns.length > 6 && <td>...</td>}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
            {data.length > 10 && (
                <p style={{
                    color: 'var(--color-text-secondary)',
                    fontSize: '0.85rem',
                    marginTop: '1rem',
                    textAlign: 'center'
                }}>
                    Showing 10 of {data.length} rows
                </p>
            )}
        </div>
    );
}

export default DataPreview;
