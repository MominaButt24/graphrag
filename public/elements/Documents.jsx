import React, { useRef, useState } from "react";

export default function Documents(props) {
  const documents = props?.documents || [];
  const fileInputRef = useRef(null);
  const [uploading, setUploading] = useState(false);

  const handleSelectFile = () => {
    if (!uploading) {
      fileInputRef.current?.click();
    }
  };

  const handleFileChange = async (event) => {
    const file = event.target.files?.[0];

    if (!file) {
      return;
    }

    setUploading(true);

    try {
      // Send the selected file through Chainlit.
      // Chainlit will turn this into a message element on the backend.
      await props.sendUserMessage("", [file]);

    } catch (error) {
      console.error("Upload failed:", error);
      setUploading(false);
    }

    // Clear the input so selecting the same file again works.
    event.target.value = "";
  };

  return (
    <div className="w-full rounded-xl border border-border bg-background p-6">

      <h2 className="mb-5 text-xl font-semibold">
        -- Documents --
      </h2>

      {/* Upload area */}
      <div
        onClick={handleSelectFile}
        className={`mb-6 rounded-xl border-2 border-dashed border-border p-8 text-center transition ${
          uploading
            ? "cursor-wait opacity-70"
            : "cursor-pointer hover:bg-muted/30"
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          className="hidden"
          onChange={handleFileChange}
          disabled={uploading}
        />

        <div className="mb-3 text-3xl">
          {uploading ? "⏳" : "↑"}
        </div>

        <p className="font-medium">
          {uploading ? "Uploading..." : "Upload documents"}
        </p>

        <p className="mt-1 text-sm text-muted-foreground">
          {uploading
            ? "Processing your document..."
            : "Click here to select a document"}
        </p>
      </div>

      {/* Document list */}
      <h3 className="mb-3 font-semibold">
        Your Documents
      </h3>

      {documents.length === 0 ? (
        <div className="rounded-lg border border-border p-6 text-center text-muted-foreground">
          No documents yet.
        </div>
      ) : (
        <div className="space-y-2">
          {documents.map((doc, index) => (
            <div
              key={doc.id ?? index}
              className="flex items-center justify-between rounded-lg border border-border p-4"
            >
              <div>
                <div className="font-medium">
                  {doc.filename}
                </div>

                <div className="text-sm text-muted-foreground">
                  {doc.chunk_count ?? 0} chunks
                </div>
              </div>

              <span className="text-sm">
                {doc.status === "done"
                  ? "✅ Ready"
                  : doc.status === "failed"
                  ? "❌ Failed"
                  : "⏳ Processing"}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

