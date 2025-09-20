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
import nltk
import os

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize FastAPI app
app = FastAPI(title="Search Engine API", description="A BM25-based search engine with typo correction")

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

class BM25_RRF_SearchEngine:
    def __init__(self, dataset):
        self.dataset = list(dataset)
        self.doc_map = {doc['id']: doc for doc in self.dataset}

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

    def search(self, query, top_n=10, k=60):
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
            score = 0.0
            if doc_idx in title_rank_map:
                score += 1 / (k + title_rank_map[doc_idx])
            if doc_idx in keyphrase_rank_map:
                score += 1 / (k + keyphrase_rank_map[doc_idx])
            if doc_idx in abstract_rank_map:
                score += 1 / (k + abstract_rank_map[doc_idx])
            rrf_scores[doc_idx] = score
            
        sorted_docs = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        
        results = []
        for doc_idx, score in sorted_docs[:top_n]:
            original_doc = self.dataset[doc_idx]
            results.append({
                'id': original_doc['id'],
                'title': original_doc.get('title', 'N/A'),
                'score': score,
                'abstract': original_doc.get('abstract', 'N/A'),
                'keyphrases': original_doc.get('keyphrases', 'N/A'),
                'corrections': corrections_made if typo_found else []
            })
            
        return results

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
        search_engine = BM25_RRF_SearchEngine(dataset_to_eval)
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
        "has_abstract_index": search_engine.bm25_abstract is not None
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
