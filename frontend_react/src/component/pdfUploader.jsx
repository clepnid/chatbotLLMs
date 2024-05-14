import React, { useState } from 'react';

const FileUploader = () => {
  const [files, setFiles] = useState(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [folderName, setFolderName] = useState('');

  const handleFileChange = (e) => {
    setFiles(e.target.files);
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
    <div>
      <input type="file" multiple onChange={handleFileChange} />
      <br />
      <button onClick={uploadFiles}>Subir archivos</button>
      <br />
      {loading && <div>Loading...</div>}
      {message && <div>{message}</div>}
      {folderName && <div>Carpeta: {folderName}</div>}
    </div>
  );
};

export default FileUploader;
