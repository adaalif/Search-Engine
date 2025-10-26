from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import string
import numpy as np
from collections import defaultdict
from datasets import load_dataset, concatenate_datasets
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi
from Levenshtein import distance as levenshtein_distance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import nltk
import os
import re
from pathlib import Path
import tarfile

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize FastAPI app
app = FastAPI(
    title="Cluster-then-Search Information Retrieval Engine", 
    description="Advanced search engine implementing cluster-then-search architecture with typo correction, cluster filtering, and RRF",
    version="3.0.0"
)

# Templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global search engine instance
search_engine = None
current_dataset = None

# Preprocessing functions
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    """Preprocess text: tokenize, lowercase, remove stopwords, stem"""
    if not isinstance(text, str):
        return []
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    return tokens

def extract_tar(archive_path, dest_dir=None, verbose=True):
    """Extract .tar.gz files safely"""
    archive_path = Path(archive_path)
    if dest_dir is None:
        dest_dir = archive_path.parent
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    # Check magic bytes (0x1F 0x8B = gzip)
    try:
        with open(archive_path, "rb") as fh:
            magic = fh.read(2)
        is_gzip = magic == b"\x1f\x8b"
    except Exception:
        is_gzip = False

    def _is_within_directory(base: Path, target: Path) -> bool:
        return str(target.resolve()).startswith(str(base.resolve()))

    def _extract_with_mode(mode):
        extracted = []
        with tarfile.open(archive_path, mode) as tf:
            for m in tf.getmembers():
                target = dest_dir / m.name
                if not _is_within_directory(dest_dir, target):
                    if verbose:
                        print(f"[skip] {m.name} (path traversal)")
                    continue
                tf.extract(m, path=dest_dir)
                extracted.append(m.name)
                if verbose:
                    print(f"[ok]  {m.name}")
        return extracted

    # Try appropriate mode first
    try:
        if is_gzip:
            if verbose: print("[detect] gzip-compressed tar → mode r:gz")
            extracted = _extract_with_mode("r:gz")
        else:
            if verbose: print("[detect] plain tar (not gzip) → mode r:")
            extracted = _extract_with_mode("r:")
    except tarfile.ReadError:
        # fallback: try opposite mode
        fallback_mode = "r:" if is_gzip else "r:gz"
        if verbose: print(f"[retry] trying fallback mode {fallback_mode} ...")
        extracted = _extract_with_mode(fallback_mode)

    if verbose:
        print(f"\nDone. Extracted to: {dest_dir.resolve()}")
    return extracted

def parse_cacm_file(filepath):
    """Parse CACM dataset"""
    with open(filepath, 'r', encoding='latin-1') as f:
        content = f.read()
    
    docs = []
    chunks = re.split(r'\.I (\d+)', content)
    
    for i in range(1, len(chunks), 2):
        doc_id = chunks[i]
        doc_content = chunks[i+1] if i+1 < len(chunks) else ""
        
        # Extract title
        title_match = re.search(r'\.T\s+(.*?)(?=\.A|\.B|\.W|\.N|\.X|\.I|$)', doc_content, re.DOTALL)
        title = title_match.group(1).strip() if title_match else ""
        
        # Extract abstract (.W section)
        abstract_match = re.search(r'\.W\s+(.*?)(?=\.X|\.I|$)', doc_content, re.DOTALL)
        abstract = abstract_match.group(1).strip() if abstract_match else ""
        
        # If no abstract, try .N section
        if not abstract:
            note_match = re.search(r'\.N\s+(.*?)(?=\.X|\.I|$)', doc_content, re.DOTALL)
            abstract = note_match.group(1).strip() if note_match else ""
        
        docs.append({
            'id': int(doc_id),
            'title': title,
            'abstract': abstract,
            'text': f"{title} {abstract}".strip()
        })
    
    return docs

def parse_cisi_file(filepath):
    """Parse CISI dataset"""
    with open(filepath, 'r', encoding='latin-1') as f:
        content = f.read()
    
    docs = []
    chunks = re.split(r'\.I (\d+)', content)
    
    for i in range(1, len(chunks), 2):
        doc_id = chunks[i]
        doc_content = chunks[i+1] if i+1 < len(chunks) else ""
        
        # Extract title (.T)
        title_match = re.search(r'\.T\s+(.*?)(?=\.A|\.W|\.I|$)', doc_content, re.DOTALL)
        title = title_match.group(1).strip() if title_match else ""
        
        # Extract abstract (.W)
        abstract_match = re.search(r'\.W\s+(.*?)(?=\.X|\.I|$)', doc_content, re.DOTALL)
        abstract = abstract_match.group(1).strip() if abstract_match else ""
        
        docs.append({
            'id': int(doc_id),
            'title': title,
            'abstract': abstract,
            'text': f"{title} {abstract}".strip()
        })
    
    return docs

class ClusterThenSearchEngine:
    """
    Cluster-then-Search Engine Implementation
    
    Phase 1 (Offline): Initialize clustering, BM25 models, vocabulary
    Phase 2 (Online): Search with cluster filtering and RRF
    """
    
    def __init__(self, dataset, n_clusters=10):
        """
        Phase 1: Offline Initialization
        """
        print("="*60)
        print("🔄 PHASE 1: OFFLINE INITIALIZATION")
        print("="*60)
        
        self.dataset = list(dataset)
        self.n_docs = len(self.dataset)
        self.n_clusters = min(n_clusters, self.n_docs)
        
        # Step 1: Load and preprocess dataset
        print("\n📝 Step 1: Loading and preprocessing dataset...")
        self._prepare_documents()
        
        # Step 2: Build vocabulary for typo correction
        print("\n📚 Step 2: Building vocabulary for typo correction...")
        self._build_vocabulary()
        
        # Step 3: TF-IDF Vectorization
        print("\n📊 Step 3: Creating TF-IDF vectors...")
        self._create_tfidf_vectors()
        
        # Step 4: K-Means Clustering
        print("\n🎯 Step 4: Performing K-Means clustering...")
        self._perform_clustering()
        
        # Step 5: Initialize BM25 models
        print("\n🔍 Step 5: Initializing BM25 models...")
        self._initialize_bm25_models()
        
        print("\n✅ Phase 1 completed successfully!")
        print(f"   - Total documents: {self.n_docs}")
        print(f"   - Vocabulary size: {len(self.vocabulary)}")
        print(f"   - Number of clusters: {len(set(self.doc_cluster_labels))}")
        print("="*60)
    
    def _prepare_documents(self):
        """Prepare documents: preprocess title and abstract separately"""
        self.processed_titles = []
        self.processed_abstracts = []
        self.combined_texts = []  # For clustering
        
        for doc in self.dataset:
            # Preprocess each field separately
            title = doc.get('title', '')
            abstract = doc.get('abstract', '')
            
            self.processed_titles.append(preprocess(title))
            self.processed_abstracts.append(preprocess(abstract))
            
            # Combine all text for clustering
            combined = f"{title} {abstract}"
            self.combined_texts.append(combined)
        
        print(f"   ✓ Processed {len(self.processed_titles)} documents")
    
    def _build_vocabulary(self):
        """Build vocabulary from entire corpus for typo correction"""
        self.vocabulary = set()
        for tokens in self.processed_titles:
            self.vocabulary.update(tokens)
        for tokens in self.processed_abstracts:
            self.vocabulary.update(tokens)
        
        print(f"   ✓ Vocabulary size: {len(self.vocabulary)}")
    
    def _create_tfidf_vectors(self):
        """Create TF-IDF vectors for clustering"""
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.combined_texts)
        print(f"   ✓ TF-IDF matrix shape: {self.tfidf_matrix.shape}")
    
    def _perform_clustering(self):
        """Perform K-Means clustering and store centroids"""
        self.kmeans_model = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10
        )
        self.doc_cluster_labels = self.kmeans_model.fit_predict(self.tfidf_matrix)
        self.cluster_centroids = self.kmeans_model.cluster_centers_
        
        # Create cluster-to-documents mapping
        self.cluster_to_docs = defaultdict(list)
        for doc_idx, cluster_id in enumerate(self.doc_cluster_labels):
            self.cluster_to_docs[cluster_id].append(doc_idx)
        
        print(f"   ✓ Clustering completed: {self.n_clusters} clusters")
        cluster_counts = defaultdict(int)
        for label in self.doc_cluster_labels:
            cluster_counts[label] += 1
        print(f"   ✓ Cluster sizes: {dict(cluster_counts)}")
    
    def _initialize_bm25_models(self):
        """Initialize two separate BM25 models for title and abstract"""
        self.bm25_title = BM25Okapi(self.processed_titles) if any(self.processed_titles) else None
        self.bm25_abstract = BM25Okapi(self.processed_abstracts) if any(self.processed_abstracts) else None
        
        if self.bm25_title: print("   ✓ BM25 model for title: initialized")
        if self.bm25_abstract: print("   ✓ BM25 model for abstract: initialized")
    
    def search(self, query, top_n=10, k=60):
        """
        Phase 2: Online Search
        """
        print("\n" + "="*60)
        print("🔍 PHASE 2: ONLINE SEARCH")
        print("="*60)
        print(f"\n📌 Query: '{query}'")
        
        # Step 1: Typo correction
        corrected_query = self._correct_typos(query)
        
        # Step 2: Preprocess query
        query_tokens = preprocess(corrected_query)
        print(f"📝 Preprocessed query: {query_tokens}")
        
        # Step 3: Filter clusters (find winning cluster)
        winning_cluster = self._find_winning_cluster(corrected_query)
        print(f"🎯 Winning cluster: {winning_cluster}")
        print(f"   Cluster contains {len(self.cluster_to_docs[winning_cluster])} documents")
        
        # Step 4: Get sub-corpus (documents in winning cluster only)
        sub_corpus_indices = self.cluster_to_docs[winning_cluster]
        print(f"📚 Searching within sub-corpus of {len(sub_corpus_indices)} documents")
        
        # Step 5: BM25 ranking within the winning cluster
        results = self._rank_within_cluster(query_tokens, sub_corpus_indices, k=k)
        
        # Step 6: Return top N results
        print(f"\n✅ Returning top {min(top_n, len(results))} results")
        print("="*60)
        
        return results[:top_n]
    
    def _correct_typos(self, query):
        """Step 1: Typo Correction using Levenshtein distance"""
        query_tokens = preprocess(query)
        corrected_tokens = []
        corrections = []
        
        for token in query_tokens:
            if token in self.vocabulary:
                corrected_tokens.append(token)
            else:
                # Find closest word in vocabulary
                min_dist = float('inf')
                closest_word = token
                
                for vocab_word in self.vocabulary:
                    dist = levenshtein_distance(token, vocab_word)
                    if dist < min_dist:
                        min_dist = dist
                        closest_word = vocab_word
                
                # Only correct if distance <= 2
                if min_dist <= 2:
                    corrected_tokens.append(closest_word)
                    if closest_word != token:
                        corrections.append(f"'{token}' -> '{closest_word}'")
                else:
                    corrected_tokens.append(token)
        
        if corrections:
            print(f"🔧 Typo corrections: {', '.join(corrections)}")
        
        return ' '.join(corrected_tokens)
    
    def _find_winning_cluster(self, query):
        """Step 3: Filter clusters using cosine similarity"""
        # Transform query to TF-IDF vector
        query_vector = self.tfidf_vectorizer.transform([query])
        
        # Calculate cosine similarity with all cluster centroids
        similarities = cosine_similarity(query_vector, self.cluster_centroids)[0]
        
        # Find cluster with highest similarity
        winning_cluster_id = np.argmax(similarities)
        similarity_score = similarities[winning_cluster_id]
        
        print(f"   Cluster {winning_cluster_id} has similarity: {similarity_score:.4f}")
        
        return int(winning_cluster_id)
    
    def _rank_within_cluster(self, query_tokens, sub_corpus_indices, k=60):
        """Step 5: BM25 Ranking within Winning Cluster"""
        # Get rankings from each BM25 model
        rank_maps = self._get_bm25_rankings(query_tokens, sub_corpus_indices)
        
        # Apply Reciprocal Rank Fusion (RRF)
        rrf_scores = self._apply_rrf(rank_maps, k=k)
        
        # Sort by score and create result documents
        sorted_scores = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for doc_idx, score in sorted_scores:
            doc = self.dataset[doc_idx]
            results.append({
                'id': doc['id'],
                'title': doc.get('title', 'N/A'),
                'score': score,
                'abstract': doc.get('abstract', 'N/A'),
                'cluster_id': int(self.doc_cluster_labels[doc_idx])
            })
        
        return results
    
    def _get_bm25_rankings(self, query_tokens, sub_corpus_indices):
        """Get rankings from both BM25 models"""
        rank_maps = {}
        
        if self.bm25_title:
            # Get scores for all documents
            all_title_scores = self.bm25_title.get_scores(query_tokens)
            # Filter to sub-corpus only and rank
            sub_title_scores = [(doc_idx, all_title_scores[doc_idx]) for doc_idx in sub_corpus_indices]
            sub_title_scores.sort(key=lambda x: x[1], reverse=True)
            rank_maps['title'] = {doc_idx: rank+1 
                                   for rank, (doc_idx, _) in enumerate(sub_title_scores)}
        
        if self.bm25_abstract:
            all_abstract_scores = self.bm25_abstract.get_scores(query_tokens)
            sub_abstract_scores = [(doc_idx, all_abstract_scores[doc_idx]) for doc_idx in sub_corpus_indices]
            sub_abstract_scores.sort(key=lambda x: x[1], reverse=True)
            rank_maps['abstract'] = {doc_idx: rank+1 
                                     for rank, (doc_idx, _) in enumerate(sub_abstract_scores)}
        
        return rank_maps
    
    def _apply_rrf(self, rank_maps, k=60):
        """Apply Reciprocal Rank Fusion (RRF)"""
        rrf_scores = defaultdict(float)
        
        all_docs = set()
        for rank_map in rank_maps.values():
            all_docs.update(rank_map.keys())
        
        for doc_idx in all_docs:
            score = 0.0
            for rank_map in rank_maps.values():
                if doc_idx in rank_map:
                    rank = rank_map[doc_idx]
                    score += 1 / (k + rank)
            rrf_scores[doc_idx] = score
        
        return rrf_scores

def load_dataset_by_name(dataset_name):
    """Load dataset by name (CACM, CISI, or Inspec)"""
    global search_engine, current_dataset
    
    if dataset_name == "CACM":
        print("🔄 Loading CACM dataset...")
        
        # Extract if needed
        if not os.path.exists("data/cacm/cacm.all"):
            if os.path.exists("data/cacm/cacm.tar.gz"):
                extract_tar("data/cacm/cacm.tar.gz", dest_dir="data/cacm")
            else:
                raise FileNotFoundError("CACM dataset not found. Please ensure data/cacm/cacm.tar.gz exists.")
        
        docs = parse_cacm_file('data/cacm/cacm.all')
        print(f"✅ Loaded {len(docs)} CACM documents")
        
    elif dataset_name == "CISI":
        print("🔄 Loading CISI dataset...")
        
        # Extract if needed
        if not os.path.exists("data/cisi/CISI.ALL"):
            if os.path.exists("data/cisi/cisi.tar.gz"):
                extract_tar("data/cisi/cisi.tar.gz", dest_dir="data/cisi")
            else:
                raise FileNotFoundError("CISI dataset not found. Please ensure data/cisi/cisi.tar.gz exists.")
        
        docs = parse_cisi_file('data/cisi/CISI.ALL')
        print(f"✅ Loaded {len(docs)} CISI documents")
        
    elif dataset_name == "Inspec":
        print("🔄 Loading Inspec dataset from Hugging Face...")
        full_dataset_dict = load_dataset("taln-ls2n/inspec")
        dataset_to_eval = concatenate_datasets([
            full_dataset_dict['train'], 
            full_dataset_dict['validation'], 
            full_dataset_dict['test']
        ])
        docs = list(dataset_to_eval)
        print(f"✅ Loaded {len(docs)} Inspec documents")
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Initialize search engine
    search_engine = ClusterThenSearchEngine(docs, n_clusters=10)
    current_dataset = dataset_name
    
    return search_engine

@app.on_event("startup")
async def startup_event():
    """Initialize the search engine on startup"""
    global search_engine, current_dataset
    print("🔄 Initializing search engine...")
    
    try:
        # Try to load CACM first, fallback to Inspec
        if os.path.exists("data/cacm/cacm.tar.gz") or os.path.exists("data/cacm/cacm.all"):
            search_engine = load_dataset_by_name("CACM")
        else:
            search_engine = load_dataset_by_name("Inspec")
        
        print(f"✅ Search engine initialized with {len(search_engine.dataset)} documents")
        print(f"📚 Vocabulary size: {len(search_engine.vocabulary)} words")
        print(f"🎯 Using dataset: {current_dataset}")
        
    except Exception as e:
        print(f"❌ Error initializing search engine: {e}")
        search_engine = None

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with search interface"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "current_dataset": current_dataset,
        "available_datasets": ["CACM", "CISI", "Inspec"]
    })

@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, query: str = Form(...), top_n: int = Form(10)):
    """Search endpoint that returns HTML results"""
    if not search_engine:
        raise HTTPException(status_code=500, detail="Search engine not initialized")
    
    try:
        results = search_engine.search(query, top_n=top_n)
        return templates.TemplateResponse("search_results.html", {
            "request": request,
            "query": query,
            "results": results,
            "total_results": len(results),
            "current_dataset": current_dataset
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.post("/switch_dataset")
async def switch_dataset(request: Request, dataset: str = Form(...)):
    """Switch to a different dataset"""
    global search_engine, current_dataset
    
    try:
        search_engine = load_dataset_by_name(dataset)
        return JSONResponse({
            "success": True,
            "message": f"Switched to {dataset} dataset",
            "dataset": dataset,
            "document_count": len(search_engine.dataset)
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"Error switching dataset: {str(e)}"
        }, status_code=500)

@app.get("/api/search")
async def api_search(query: str, top_n: int = 10):
    """API endpoint for JSON search results"""
    if not search_engine:
        raise HTTPException(status_code=500, detail="Search engine not initialized")
    
    try:
        results = search_engine.search(query, top_n=top_n)
        return {
            "query": query,
            "total_results": len(results),
            "results": results,
            "dataset": current_dataset
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.get("/document/{doc_id}", response_class=HTMLResponse)
async def get_document(request: Request, doc_id: str):
    """Get full document details by ID with HTML template"""
    if not search_engine:
        raise HTTPException(status_code=500, detail="Search engine not initialized")
    
    # Find document by ID
    doc = None
    for document in search_engine.dataset:
        if str(document['id']) == str(doc_id):
            doc = document
            break
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return templates.TemplateResponse("document.html", {
        "request": request,
        "doc": {
            "id": doc['id'],
            "title": doc.get('title', 'N/A'),
            "abstract": doc.get('abstract', 'N/A')
        },
        "current_dataset": current_dataset
    })

@app.get("/api/document/{doc_id}")
async def get_document_api(doc_id: str):
    """Get full document details by ID as JSON API"""
    if not search_engine:
        raise HTTPException(status_code=500, detail="Search engine not initialized")
    
    # Find document by ID
    doc = None
    for document in search_engine.dataset:
        if str(document['id']) == str(doc_id):
            doc = document
            break
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {
        "id": doc['id'],
        "title": doc.get('title', 'N/A'),
        "abstract": doc.get('abstract', 'N/A'),
        "dataset": current_dataset
    }

@app.get("/stats")
async def get_stats():
    """Get search engine statistics"""
    if not search_engine:
        raise HTTPException(status_code=500, detail="Search engine not initialized")
    
    return {
        "total_documents": len(search_engine.dataset),
        "vocabulary_size": len(search_engine.vocabulary),
        "has_title_index": search_engine.bm25_title is not None,
        "has_abstract_index": search_engine.bm25_abstract is not None,
        "clusters_count": len(set(search_engine.doc_cluster_labels)),
        "clustering_enabled": True,
        "current_dataset": current_dataset
    }

@app.get("/clusters")
async def get_clusters():
    """Get clustering information"""
    if not search_engine:
        raise HTTPException(status_code=500, detail="Search engine not initialized")
    
    cluster_info = {}
    for cluster_id in range(len(set(search_engine.doc_cluster_labels))):
        docs_in_cluster = search_engine.cluster_to_docs[cluster_id]
        cluster_info[cluster_id] = {
            "size": len(docs_in_cluster),
            "documents": [search_engine.dataset[i]['id'] for i in docs_in_cluster[:10]]  # Top 10 docs
        }
    
    return {
        "total_clusters": len(set(search_engine.doc_cluster_labels)),
        "clusters": cluster_info,
        "dataset": current_dataset
    }

@app.get("/document/{doc_id}/similar", response_class=HTMLResponse)
async def get_similar_documents(request: Request, doc_id: str):
    """Get similar documents based on clustering with HTML template"""
    if not search_engine:
        raise HTTPException(status_code=500, detail="Search engine not initialized")
    
    # Find document and its cluster
    doc_idx = None
    for i, doc in enumerate(search_engine.dataset):
        if str(doc['id']) == str(doc_id):
            doc_idx = i
            break
    
    if doc_idx is None:
        raise HTTPException(status_code=404, detail="Document not found")
    
    cluster_id = search_engine.doc_cluster_labels[doc_idx]
    cluster_docs = search_engine.cluster_to_docs[cluster_id]
    
    similar_docs = []
    for similar_doc_idx in cluster_docs[:10]:  # Top 10 similar docs
        if similar_doc_idx != doc_idx:  # Exclude the original document
            doc = search_engine.dataset[similar_doc_idx]
            similar_docs.append({
                'id': doc['id'],
                'title': doc.get('title', 'N/A'),
                'abstract': doc.get('abstract', 'N/A')[:200] + '...' if len(doc.get('abstract', '')) > 200 else doc.get('abstract', 'N/A')
            })
    
    return templates.TemplateResponse("similar_documents.html", {
        "request": request,
        "original_doc_id": doc_id,
        "cluster_id": cluster_id,
        "cluster_size": len(cluster_docs),
        "similar_documents": similar_docs,
        "current_dataset": current_dataset
    })

@app.get("/api/document/{doc_id}/similar")
async def get_similar_documents_api(doc_id: str):
    """Get similar documents based on clustering as JSON API"""
    if not search_engine:
        raise HTTPException(status_code=500, detail="Search engine not initialized")
    
    # Find document and its cluster
    doc_idx = None
    for i, doc in enumerate(search_engine.dataset):
        if str(doc['id']) == str(doc_id):
            doc_idx = i
            break
    
    if doc_idx is None:
        raise HTTPException(status_code=404, detail="Document not found")
    
    cluster_id = search_engine.doc_cluster_labels[doc_idx]
    cluster_docs = search_engine.cluster_to_docs[cluster_id]
    
    similar_docs = []
    for similar_doc_idx in cluster_docs[:10]:  # Top 10 similar docs
        if similar_doc_idx != doc_idx:  # Exclude the original document
            doc = search_engine.dataset[similar_doc_idx]
            similar_docs.append({
                'id': doc['id'],
                'title': doc.get('title', 'N/A'),
                'abstract': doc.get('abstract', 'N/A')[:200] + '...' if len(doc.get('abstract', '')) > 200 else doc.get('abstract', 'N/A')
            })
    
    return {
        "original_doc_id": doc_id,
        "cluster_id": cluster_id,
        "cluster_size": len(cluster_docs),
        "similar_documents": similar_docs,
        "dataset": current_dataset
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)