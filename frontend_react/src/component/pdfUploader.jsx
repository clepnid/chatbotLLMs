import React, { useState } from 'react';
import './fileUploader.css'; // Importa el archivo CSS aquí

const FileUploader = () => {
  const [files, setFiles] = useState(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [folderName, setFolderName] = useState('');
  const [dragOver, setDragOver] = useState(false);

  const handleFileChange = (e) => {
    setFiles(e.target.files);
    setDragOver(false);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = () => {
    setDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const droppedFiles = Array.from(e.dataTransfer.files);
    setFiles(droppedFiles);
    setDragOver(false);
  };

  const uploadFiles = async () => {
    if (!files) return;

    setLoading(true);

    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append('pdf', files[i]);
    }

    try {
      const response = await fetch('http://localhost:8080/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Error al subir los archivos.');
      }

      const data = await response.json();

      setMessage(data.message);
      setFolderName(data.foldername);
    } catch (error) {
      console.error('Error uploading files:', error);
      setMessage('Error al subir los archivos.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={`file-uploader ${dragOver ? 'drag-over' : ''}`} onDragOver={handleDragOver} onDragLeave={handleDragLeave} onDrop={handleDrop}>
      <input type="file" multiple onChange={handleFileChange} />
      <div className="drag-drop-area">
        {files ? (
          <ul>
            {Array.from(files).map((file, index) => (
              <li key={index}>{file.name}</li>
            ))}
          </ul>
        ) : (
          <p>Arrastra y suelta archivos aquí</p>
        )}
      </div>
      <button onClick={uploadFiles} disabled={loading}>
        {loading ? 'Subiendo archivos...' : 'Subir archivos'}
      </button>
      {loading && <div className="loading">Cargando...</div>}
      {message && <div className="message">{message}</div>}
      {folderName && <div className="folder-name">Carpeta: {folderName}</div>}
    </div>
  );
};

export default FileUploader;