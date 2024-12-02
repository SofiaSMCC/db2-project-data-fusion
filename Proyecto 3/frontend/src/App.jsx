import { useState } from "react";
import { useDropzone } from "react-dropzone";
import "./App.css";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [algorithm, setAlgorithm] = useState("knn-sequential");
  const [radius, setRadius] = useState("");
  const [k, setK] = useState("");
  const [results, setResults] = useState([]);

  // Configuración del drag-and-drop
  const onDrop = (acceptedFiles) => {
    const file = acceptedFiles[0];
    setSelectedFile(file);

    // Generar vista previa
    const reader = new FileReader();
    reader.onload = () => setPreview(reader.result);
    reader.readAsDataURL(file);
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: "image/*",
    maxFiles: 1,
  });

  const handleAlgorithmChange = (event) => {
    setAlgorithm(event.target.value);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    if (!selectedFile) {
      alert("Please select an image.");
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    if (algorithm === "knn-sequential" || algorithm === "knn-rtree" || algorithm === "knn-faiss") {
      formData.append("k", k);
    } else if (algorithm === "range-search") {
      formData.append("radius", radius);
    }

    try {
      const response = await fetch(`http://localhost:8000/${algorithm}`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log("Response data:", data);

      const formattedResults = data.results
        .map(([distance, image]) => {
          const filename = image.split(/[/\\]/).pop();
          return { distance: parseFloat(distance), image: filename };
        })
        .sort((a, b) => a.distance - b.distance);

      setResults(formattedResults);
    } catch (error) {
      console.error("Error while fetching results:", error);
      alert("An error occurred. Please try again.");
    }
  };

  return (
    <div className="container">
      <h1>Image Search Application</h1>

      <form onSubmit={handleSubmit}>
        <div
          {...getRootProps()}
          className={`dropzone ${isDragActive ? "dropzone-active" : ""}`}
        >
          <input {...getInputProps()} />
          {isDragActive ? (
            <p>Drop the image here...</p>
          ) : (
            <p>Drag and drop an image here, or click to select one</p>
          )}
        </div>
        {preview && <img src={preview} alt="Preview" className="preview" />}

        <div>
          <label>Select Algorithm:</label>
          <select value={algorithm} onChange={handleAlgorithmChange}>
            <option value="knn-sequential">KNN Sequential</option>
            <option value="range-search">Range Search</option>
            <option value="knn-rtree">KNN with R-tree</option>
            <option value="knn-faiss">KNN with FAISS</option>
          </select>
        </div>

        {algorithm === "range-search" && (
          <div>
            <label>Radius:</label>
            <input
              type="number"
              step="0.1"
              value={radius}
              onChange={(e) => setRadius(e.target.value)}
            />
          </div>
        )}

        {(algorithm === "knn-sequential" || algorithm === "knn-rtree" || algorithm === "knn-faiss") && (
          <div>
            <label>K:</label>
            <input
              type="number"
              value={k}
              onChange={(e) => setK(e.target.value)}
            />
          </div>
        )}

        <button type="submit">Search</button>
      </form>

      <div className="results">
        <h2>Results:</h2>
        {results.length > 0 ? (
          <div className="results-grid">
            {results.map((result, index) => (
              <div key={index} className="result-item">
                <img
                  src={`http://localhost:8000/poke2?img=${result.image}`}
                  alt={`Result ${index + 1}`}
                />
                <p><strong>Image:</strong> {result.image}</p>
                <p><strong>Distance:</strong> {result.distance.toFixed(2)}</p>
              </div>
            ))}
          </div>
        ) : (
          <p>No results yet.</p>
        )}
      </div>
    </div>
  );
}

export default App;
