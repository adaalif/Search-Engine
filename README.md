# Search Engine FastAPI Simulation

A web-based search engine simulation built with FastAPI that implements BM25 ranking with Reciprocal Rank Fusion (RRF) and automatic typo correction.

## Features

- 🔍 **Advanced Search**: BM25-based ranking across multiple document fields
- ✨ **Typo Correction**: Automatic spelling correction using Levenshtein distance
- 🎯 **RRF Fusion**: Combines scores from title, abstract, and keyphrases
- 🌐 **Simple Web Interface**: Clean, functional web UI
- 📊 **Statistics**: Real-time search engine statistics
- 📄 **Document Viewer**: Full document details with HTML template
- 🔌 **API Endpoints**: RESTful API for programmatic access

## Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   # Simple way
   python start.py
   
   # Or directly
   python main.py
   ```

4. **Open your browser** and go to `http://localhost:8000`

## Usage

### Web Interface

1. **Search**: Enter your query in the search box
2. **View Results**: Browse through ranked search results
3. **View Documents**: Click "View Full Document" to see complete content
4. **Statistics**: Check the search engine stats on the homepage

### API Endpoints

- `GET /` - Home page with search interface
- `POST /search` - Search with HTML results
- `GET /api/search?query=your_query&top_n=10` - JSON search results
- `GET /document/{doc_id}` - Get full document by ID
- `GET /stats` - Get search engine statistics

### Example API Usage

```bash
# Search for documents
curl "http://localhost:8000/api/search?query=computer%20vision&top_n=5"

# Get document details
curl "http://localhost:8000/document/1103"

# Get statistics
curl "http://localhost:8000/stats"
```

## How It Works

### Search Algorithm

1. **Preprocessing**: 
   - Tokenization using NLTK
   - Stop word removal
   - Stemming with Porter Stemmer

2. **Typo Correction**:
   - Uses Levenshtein distance to find closest vocabulary words
   - Corrects words with distance ≤ 3

3. **BM25 Scoring**:
   - Separate BM25 indices for title, abstract, and keyphrases
   - Calculates relevance scores for each field

4. **Reciprocal Rank Fusion**:
   - Combines scores from all fields using RRF formula
   - Ranks documents by combined relevance

### Dataset

The search engine uses the INSPEC dataset from Hugging Face, which contains:
- Academic papers with titles, abstracts, and keyphrases
- Over 1,000 documents for testing
- Computer science and engineering topics

## Project Structure

```
Information retrieaval/
├── main.py                 # FastAPI application
├── start.py               # Simple startup script
├── requirements.txt       # Python dependencies
├── README.md             # Documentation
├── templates/            # HTML templates
│   ├── index.html        # Home page
│   ├── search_results.html # Search results page
│   └── document.html     # Document viewer page
└── static/               # Static files
    ├── css/
    │   └── style.css     # All CSS styles
    └── js/
        └── main.js       # JavaScript functionality
```

## Technical Details

- **Framework**: FastAPI with Jinja2 templates
- **Search Algorithm**: BM25 with RRF
- **Typo Correction**: Levenshtein distance
- **UI**: Clean HTML/CSS/JS (no external frameworks)
- **Data Source**: Hugging Face datasets

## Configuration

You can modify search parameters in the code:

- `top_n`: Number of results to return (default: 10)
- `k`: RRF parameter for score combination (default: 60)
- Typo correction threshold (default: distance ≤ 3)

## Troubleshooting

1. **Dataset Loading Issues**: The app downloads the dataset on first run. Ensure you have internet connection.

2. **Memory Issues**: The dataset is loaded into memory. For larger datasets, consider implementing lazy loading.

3. **Port Conflicts**: If port 8000 is busy, modify the port in `main.py`:
   ```python
   uvicorn.run(app, host="0.0.0.0", port=8001)
   ```

## License

This project is for educational purposes. The INSPEC dataset is used under its respective license.
