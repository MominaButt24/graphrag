export default function DocsLibrary(props) {
  const documents = props.documents || [];

  const formatDate = (unixSeconds) => {
    const d = new Date(unixSeconds * 1000);
    return d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
  };

  if (documents.length === 0) {
    return (
      <div style={{ padding: "16px", opacity: 0.7, fontStyle: "italic" }}>
        No documents ingested yet — upload a file in the chat to see it here.
      </div>
    );
  }

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))",
        gap: "12px",
        padding: "8px",
      }}
    >
      {documents.map((doc, i) => (
        <div
          key={i}
          style={{
            border: "1px solid rgba(128,128,128,0.3)",
            borderRadius: "10px",
            padding: "14px",
            background: "rgba(128,128,128,0.06)",
          }}
        >
          <div style={{ fontSize: "20px", marginBottom: "6px" }}>📄</div>
          <div
            style={{
              fontWeight: 600,
              fontSize: "14px",
              marginBottom: "8px",
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
            }}
            title={doc.filename}
          >
            {doc.filename}
          </div>
          <div style={{ fontSize: "12px", opacity: 0.75, display: "flex", justifyContent: "space-between" }}>
            <span>{doc.chunk_count} chunks</span>
            <span>{formatDate(doc.uploaded_at)}</span>
          </div>
        </div>
      ))}
    </div>
  );
}