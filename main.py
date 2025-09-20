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

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize FastAPI app
app = FastAPI(
    title="Advanced Information Retrieval Search Engine", 
    description="A comprehensive search engine implementing BM25 ranking, document clustering, typo correction, and multi-field search with Reciprocal Rank Fusion (RRF)",
    version="2.0.0"
)

# Templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global search engine instance
search_engine = None

# Preprocessing functions (from your notebook)
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    if not isinstance(text, str):
        return []
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    return tokens

class BM25_RRF_Clustered_SearchEngine:
    """
    Advanced Information Retrieval Search Engine
    
    This class implements a comprehensive search engine that combines:
    1. Multi-field BM25 ranking (title, abstract, keyphrases)
    2. Document clustering using K-Means with TF-IDF
    3. Reciprocal Rank Fusion (RRF) for score combination
    4. Automatic typo correction using Levenshtein distance
    5. Clustering-based relevance boosting
    
    Pipeline Overview:
    - Document Preprocessing: Tokenization, stopword removal, stemming
    - Vocabulary Building: Create searchable term dictionary
    - Index Construction: Build BM25 indices for each field
    - Document Clustering: Group similar documents using K-Means
    - Query Processing: Preprocess and correct user queries
    - Multi-field Search: Search across title, abstract, and keyphrases
    - Score Fusion: Combine scores using RRF with clustering boost
    - Result Ranking: Return ranked results with cluster information
    """
    def __init__(self, dataset, n_clusters=10):
        self.dataset = list(dataset)
        self.doc_map = {doc['id']: doc for doc in self.dataset}
        self.n_clusters = n_clusters

        tokenized_titles = [preprocess(doc.get('title', '')) for doc in self.dataset]
        tokenized_keyphrases = [preprocess(' '.join(doc.get('keyphrases', []))) for doc in self.dataset]
        tokenized_abstracts = [preprocess(doc.get('abstract', '')) for doc in self.dataset]
        
        self.vocabulary = set()
        for doc_tokens in tokenized_titles:
            self.vocabulary.update(doc_tokens)
        if any(tokenized_keyphrases):
            for doc_tokens in tokenized_keyphrases:
                self.vocabulary.update(doc_tokens)
        for doc_tokens in tokenized_abstracts:
            self.vocabulary.update(doc_tokens)
        
        if any(tokenized_titles): 
            self.bm25_title = BM25Okapi(tokenized_titles)
        else: 
            self.bm25_title = None
            
        if any(tokenized_keyphrases): 
            self.bm25_keyphrases = BM25Okapi(tokenized_keyphrases)
        else: 
            self.bm25_keyphrases = None
            
        if any(tokenized_abstracts): 
            self.bm25_abstract = BM25Okapi(tokenized_abstracts)
        else: 
            self.bm25_abstract = None
        
        # Initialize clustering
        self._setup_clustering()
    
    def _setup_clustering(self):
        """Setup document clustering using K-Means"""
        print("🔄 Setting up document clustering...")
        
        # Combine all text for clustering
        combined_texts = []
        for doc in self.dataset:
            title = doc.get('title', '')
            abstract = doc.get('abstract', '')
            keyphrases = ' '.join(doc.get('keyphrases', []))
            combined_text = f"{title} {abstract} {keyphrases}"
            combined_texts.append(combined_text)
        
        # Create TF-IDF vectors for clustering
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(combined_texts)
        
        # Perform K-Means clustering
        self.kmeans = KMeans(
            n_clusters=min(self.n_clusters, len(self.dataset)),
            random_state=42,
            n_init=10
        )
        
        self.doc_clusters = self.kmeans.fit_predict(tfidf_matrix)
        self.cluster_centers = self.kmeans.cluster_centers_
        
        # Create cluster to documents mapping
        self.cluster_to_docs = defaultdict(list)
        for i, cluster_id in enumerate(self.doc_clusters):
            self.cluster_to_docs[cluster_id].append(i)
        
        print(f"✅ Clustering completed: {len(set(self.doc_clusters))} clusters created")
    
    def get_cluster_info(self, doc_id):
        """Get cluster information for a document"""
        for i, doc in enumerate(self.dataset):
            if str(doc['id']) == str(doc_id):
                cluster_id = self.doc_clusters[i]
                cluster_docs = self.cluster_to_docs[cluster_id]
                return {
                    'cluster_id': int(cluster_id),
                    'cluster_size': len(cluster_docs),
                    'similar_docs': [self.dataset[j]['id'] for j in cluster_docs[:5]]  # Top 5 similar docs
                }
        return None

    def _correct_query_word(self, word):
        if word in self.vocabulary:
            return word
        min_dist = float('inf')
        corrected_word = word
        for vocab_word in self.vocabulary:
            dist = levenshtein_distance(word, vocab_word)
            if dist < min_dist:
                min_dist = dist
                corrected_word = vocab_word
        return corrected_word if min_dist <= 3 else word

    def search(self, query, top_n=10, k=60, use_clustering=True):
        query_tokens = preprocess(query)
        
        corrected_query_tokens = []
        corrections_made = []
        typo_found = False

        for token in query_tokens:
            corrected_word = self._correct_query_word(token)
            if corrected_word != token:
                typo_found = True
                corrections_made.append(f"'{token}' -> '{corrected_word}'")
            corrected_query_tokens.append(corrected_word)

        # If using clustering, find relevant clusters first
        if use_clustering:
            relevant_clusters = self._find_relevant_clusters(query)
            print(f"🎯 Found {len(relevant_clusters)} relevant clusters")
        else:
            relevant_clusters = None

        title_rank_map, keyphrase_rank_map, abstract_rank_map = {}, {}, {}
        all_doc_indices = set()
        
        if self.bm25_title is not None:
            title_scores = self.bm25_title.get_scores(corrected_query_tokens)
            title_ranks_indices = np.argsort(title_scores)[::-1]
            title_rank_map = {doc_idx: rank + 1 for rank, doc_idx in enumerate(title_ranks_indices)}
            all_doc_indices.update(title_ranks_indices)

        if self.bm25_keyphrases is not None:
            keyphrase_scores = self.bm25_keyphrases.get_scores(corrected_query_tokens)
            keyphrase_ranks_indices = np.argsort(keyphrase_scores)[::-1]
            keyphrase_rank_map = {doc_idx: rank + 1 for rank, doc_idx in enumerate(keyphrase_ranks_indices)}
            all_doc_indices.update(keyphrase_ranks_indices)
            
        if self.bm25_abstract is not None:
            abstract_scores = self.bm25_abstract.get_scores(corrected_query_tokens)
            abstract_ranks_indices = np.argsort(abstract_scores)[::-1]
            abstract_rank_map = {doc_idx: rank + 1 for rank, doc_idx in enumerate(abstract_ranks_indices)}
            all_doc_indices.update(abstract_ranks_indices)
        
        rrf_scores = defaultdict(float)
        for doc_idx in all_doc_indices:
            # Apply clustering boost if document is in relevant clusters
            cluster_boost = 1.0
            if use_clustering and relevant_clusters is not None:
                doc_cluster = self.doc_clusters[doc_idx]
                if doc_cluster in relevant_clusters:
                    cluster_boost = 1.2  # 20% boost for relevant cluster documents
            
            score = 0.0
            if doc_idx in title_rank_map:
                score += 1 / (k + title_rank_map[doc_idx])
            if doc_idx in keyphrase_rank_map:
                score += 1 / (k + keyphrase_rank_map[doc_idx])
            if doc_idx in abstract_rank_map:
                score += 1 / (k + abstract_rank_map[doc_idx])
            
            rrf_scores[doc_idx] = score * cluster_boost
            
        sorted_docs = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        
        results = []
        for doc_idx, score in sorted_docs[:top_n]:
            original_doc = self.dataset[doc_idx]
            cluster_info = self.get_cluster_info(original_doc['id'])
            results.append({
                'id': original_doc['id'],
                'title': original_doc.get('title', 'N/A'),
                'score': score,
                'abstract': original_doc.get('abstract', 'N/A'),
                'keyphrases': original_doc.get('keyphrases', 'N/A'),
                'corrections': corrections_made if typo_found else [],
                'cluster_id': cluster_info['cluster_id'] if cluster_info else None,
                'cluster_size': cluster_info['cluster_size'] if cluster_info else None
            })
            
        return results
    
    def _find_relevant_clusters(self, query):
        """Find clusters most relevant to the query"""
        # Transform query to TF-IDF vector
        query_vector = self.tfidf_vectorizer.transform([query])
        
        # Calculate similarity between query and cluster centers
        similarities = cosine_similarity(query_vector, self.cluster_centers)[0]
        
        # Get top clusters (above threshold)
        threshold = 0.1
        relevant_clusters = []
        for i, sim in enumerate(similarities):
            if sim > threshold:
                relevant_clusters.append(i)
        
        # If no clusters above threshold, return top 3
        if not relevant_clusters:
            top_indices = np.argsort(similarities)[-3:]
            relevant_clusters = top_indices.tolist()
        
        return relevant_clusters

@app.on_event("startup")
async def startup_event():
    """Initialize the search engine on startup"""
    global search_engine
    print("🔄 Loading dataset and initializing search engine...")
    
    try:
        # Load the dataset
        full_dataset_dict = load_dataset("taln-ls2n/inspec")
        dataset_to_eval = concatenate_datasets([
            full_dataset_dict['train'], 
            full_dataset_dict['validation'], 
            full_dataset_dict['test']
        ])
        
        # Initialize search engine
        search_engine = BM25_RRF_Clustered_SearchEngine(dataset_to_eval)
        print(f"✅ Search engine initialized with {len(search_engine.dataset)} documents")
        print(f"📚 Vocabulary size: {len(search_engine.vocabulary)} words")
        
    except Exception as e:
        print(f"❌ Error initializing search engine: {e}")
        search_engine = None

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with search interface"""
    return templates.TemplateResponse("index.html", {"request": request})

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
            "total_results": len(results)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

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
            "results": results
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
            "abstract": doc.get('abstract', 'N/A'),
            "keyphrases": doc.get('keyphrases', 'N/A')
        }
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
        "keyphrases": doc.get('keyphrases', 'N/A')
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
        "has_keyphrase_index": search_engine.bm25_keyphrases is not None,
        "has_abstract_index": search_engine.bm25_abstract is not None,
        "clusters_count": len(set(search_engine.doc_clusters)),
        "clustering_enabled": True
    }

@app.get("/clusters")
async def get_clusters():
    """Get clustering information"""
    if not search_engine:
        raise HTTPException(status_code=500, detail="Search engine not initialized")
    
    cluster_info = {}
    for cluster_id in range(len(set(search_engine.doc_clusters))):
        docs_in_cluster = search_engine.cluster_to_docs[cluster_id]
        cluster_info[cluster_id] = {
            "size": len(docs_in_cluster),
            "documents": [search_engine.dataset[i]['id'] for i in docs_in_cluster[:10]]  # Top 10 docs
        }
    
    return {
        "total_clusters": len(set(search_engine.doc_clusters)),
        "clusters": cluster_info
    }

@app.get("/document/{doc_id}/similar", response_class=HTMLResponse)
async def get_similar_documents(request: Request, doc_id: str):
    """Get similar documents based on clustering with HTML template"""
    if not search_engine:
        raise HTTPException(status_code=500, detail="Search engine not initialized")
    
    cluster_info = search_engine.get_cluster_info(doc_id)
    if not cluster_info:
        raise HTTPException(status_code=404, detail="Document not found")
    
    similar_docs = []
    for similar_doc_id in cluster_info['similar_docs']:
        if str(similar_doc_id) != str(doc_id):  # Exclude the original document
            for doc in search_engine.dataset:
                if str(doc['id']) == str(similar_doc_id):
                    similar_docs.append({
                        'id': doc['id'],
                        'title': doc.get('title', 'N/A'),
                        'abstract': doc.get('abstract', 'N/A')[:200] + '...' if len(doc.get('abstract', '')) > 200 else doc.get('abstract', 'N/A')
                    })
                    break
    
    return templates.TemplateResponse("similar_documents.html", {
        "request": request,
        "original_doc_id": doc_id,
        "cluster_id": cluster_info['cluster_id'],
        "cluster_size": cluster_info['cluster_size'],
        "similar_documents": similar_docs
    })

@app.get("/api/document/{doc_id}/similar")
async def get_similar_documents_api(doc_id: str):
    """Get similar documents based on clustering as JSON API"""
    if not search_engine:
        raise HTTPException(status_code=500, detail="Search engine not initialized")
    
    cluster_info = search_engine.get_cluster_info(doc_id)
    if not cluster_info:
        raise HTTPException(status_code=404, detail="Document not found")
    
    similar_docs = []
    for similar_doc_id in cluster_info['similar_docs']:
        if str(similar_doc_id) != str(doc_id):  # Exclude the original document
            for doc in search_engine.dataset:
                if str(doc['id']) == str(similar_doc_id):
                    similar_docs.append({
                        'id': doc['id'],
                        'title': doc.get('title', 'N/A'),
                        'abstract': doc.get('abstract', 'N/A')[:200] + '...' if len(doc.get('abstract', '')) > 200 else doc.get('abstract', 'N/A')
                    })
                    break
    
    return {
        "original_doc_id": doc_id,
        "cluster_id": cluster_info['cluster_id'],
        "cluster_size": cluster_info['cluster_size'],
        "similar_documents": similar_docs
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
