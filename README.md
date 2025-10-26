# Information Retrieval Search Engine with Clustering

## 📖 Overview

This project implements a sophisticated **Information Retrieval (IR) Search Engine** that uses clustering to improve search performance. Think of it like a smart library system that not only finds books but also groups similar books together to make searching faster and more accurate.

### What is Information Retrieval?
Information Retrieval is the science of finding relevant information from large collections of documents. When you search on Google, Bing, or any search engine, you're using IR technology. This project demonstrates how to build such a system from scratch.

## 🎯 What This Project Does

### The Problem
Imagine you have thousands of research papers and you want to find the most relevant ones for your query. A simple approach would be to search through every single document, but this is slow and often returns irrelevant results.

### Our Solution: Cluster-then-Search
Our system uses a two-phase approach:

1. **Phase 1 (Offline)**: Organize documents into groups (clusters) of similar content
2. **Phase 2 (Online)**: When someone searches, only look in the most relevant cluster(s)

This is like organizing a library by subject areas - when you want books about "machine learning," you go directly to the computer science section instead of searching the entire library.

## 🏗️ System Architecture

### Phase 1: Offline Initialization (Setup)
```
Documents → Preprocessing → Clustering → BM25 Models → Ready for Search
```

**What happens here:**
- **Preprocessing**: Clean and standardize text (remove punctuation, convert to lowercase, etc.)
- **Clustering**: Group similar documents together using K-Means algorithm
- **BM25 Models**: Create search models for different document parts (title, abstract, keyphrases)
- **Vocabulary Building**: Create a word dictionary for typo correction

### Phase 2: Online Search (When User Searches)
```
Query → Typo Correction → Find Best Cluster → Search in Cluster → Rank Results
```

**What happens here:**
- **Typo Correction**: Fix spelling mistakes automatically
- **Cluster Selection**: Find which cluster is most relevant to the query
- **Focused Search**: Only search within the selected cluster (much faster!)
- **Result Ranking**: Combine scores from different document parts using RRF

## 🔧 Key Technologies Used

### 1. **Text Preprocessing**
- **Tokenization**: Split text into individual words
- **Stemming**: Reduce words to their root form (e.g., "running" → "run")
- **Stop Word Removal**: Remove common words like "the", "and", "is"

### 2. **Clustering (K-Means)**
- Groups documents with similar content together
- Uses TF-IDF vectors to represent document content
- Creates cluster centroids (center points) for each group

### 3. **BM25 Ranking**
- A sophisticated algorithm for ranking search results
- Considers term frequency and document length
- Separate models for title, abstract, and keyphrases

### 4. **Reciprocal Rank Fusion (RRF)**
- Combines rankings from multiple sources
- Formula: `RRF(d) = Σ 1/(k + rank_i(d))`
- Ensures balanced consideration of all document parts

### 5. **Typo Correction**
- Uses Levenshtein distance to find similar words
- Automatically corrects spelling mistakes in queries
- Only corrects if the distance is ≤ 2 characters

## 📊 Datasets Used

### 1. **CACM Dataset**
- **Size**: 3,204 documents
- **Content**: Computer science research papers
- **Queries**: 64 test queries
- **Relevance Judgments**: 52 queries with known relevant documents

### 2. **CISI Dataset**
- **Size**: 1,460 documents
- **Content**: Information science research papers
- **Queries**: 112 test queries
- **Relevance Judgments**: 76 queries with known relevant documents

### 3. **Inspec Dataset (Hugging Face)**
- **Size**: 2,000 documents
- **Content**: Computer science abstracts with keyphrases
- **Purpose**: Initial testing and development

## 📈 Evaluation Metrics

We measure our system's performance using standard IR metrics:

### 1. **Precision@10**
- **What it measures**: Of the top 10 results, how many are actually relevant?
- **Example**: If 7 out of 10 results are relevant, Precision@10 = 0.7

### 2. **Recall@10**
- **What it measures**: Of all relevant documents, how many did we find in the top 10?
- **Example**: If there are 20 relevant documents and we found 8 in top 10, Recall@10 = 0.4

### 3. **F1@10**
- **What it measures**: Harmonic mean of Precision and Recall
- **Formula**: F1 = 2 × (Precision × Recall) / (Precision + Recall)
- **Why useful**: Balances both precision and recall

### 4. **Mean Average Precision (MAP)**
- **What it measures**: Average precision across all queries
- **Why important**: Considers the ranking order of results

### 5. **Mean Reciprocal Rank (MRR)**
- **What it measures**: How quickly we find the first relevant result
- **Example**: If the first relevant result is at position 3, MRR = 1/3 = 0.33

## 🚀 How to Use This Project

### Prerequisites
```bash
pip install nltk scikit-learn numpy python-Levenshtein rank-bm25 datasets matplotlib seaborn
```

### Running the Notebook
1. Open `cluster-search-clean.ipynb` in Jupyter Notebook
2. Run cells sequentially from top to bottom
3. The notebook will:
   - Load and preprocess datasets
   - Build the search engine
   - Run evaluations
   - Generate visualizations

### Example Search
```python
# Initialize the search engine
engine = ClusterThenSearchEngine(dataset, n_clusters=10)

# Perform a search
results = engine.search("machine learning algorithms", top_n=5)

# View results
for doc in results:
    print(f"Title: {doc['title']}")
    print(f"Score: {doc['score']}")
    print(f"Cluster: {doc['cluster_id']}")
```

## 📊 Experimental Results

### CACM Dataset Results
- **Best Performance**: K=2 clusters
- **MRR**: 0.7694
- **F1@10**: 0.2494
- **MAP**: 0.2234

### CISI Dataset Results
- **Best Performance**: Varies by metric
- **MRR**: ~0.5850
- **F1@10**: ~0.1240
- **MAP**: ~0.0651

### Key Findings
1. **Optimal Cluster Count**: Different datasets perform best with different numbers of clusters
2. **CACM vs CISI**: CACM generally performs better, possibly due to dataset characteristics
3. **Cluster Impact**: Clustering significantly improves search efficiency and often improves relevance

## 🔍 Understanding the Code Structure

### Main Classes

#### `ClusterThenSearchEngine`
The main search engine class that implements the two-phase approach.

**Key Methods:**
- `__init__()`: Phase 1 - Builds all models offline
- `search()`: Phase 2 - Performs online search
- `_correct_typos()`: Handles spelling mistakes
- `_find_winning_cluster()`: Selects most relevant cluster
- `_rank_within_cluster()`: Ranks documents in selected cluster

### Key Functions

#### `preprocess(text)`
Cleans and standardizes text for processing.

#### `evaluate_search()`
Runs comprehensive evaluation using multiple metrics.

#### `visualize_clusters()`
Creates visualizations of cluster distributions and performance.

## 🎓 Learning Outcomes

After studying this project, you'll understand:

1. **Information Retrieval Fundamentals**
   - How search engines work
   - Document preprocessing techniques
   - Ranking algorithms (BM25, RRF)

2. **Machine Learning Applications**
   - Clustering algorithms (K-Means)
   - Feature extraction (TF-IDF)
   - Similarity measures (cosine similarity)

3. **Evaluation Methods**
   - Standard IR metrics
   - Experimental design
   - Performance analysis

4. **System Design**
   - Two-phase architecture
   - Efficiency vs. accuracy trade-offs
   - Scalability considerations

## 🔬 Advanced Features

### 1. **Typo Correction**
- Automatically fixes spelling mistakes
- Uses edit distance algorithms
- Improves user experience

### 2. **Multi-field Search**
- Searches title, abstract, and keyphrases separately
- Combines results using RRF
- More comprehensive than single-field search

### 3. **Cluster Filtering**
- Reduces search space dramatically
- Improves both speed and relevance
- Adaptive to query content

### 4. **Comprehensive Evaluation**
- Tests multiple cluster configurations
- Compares different datasets
- Provides detailed performance analysis

## 🚧 Future Improvements

### Potential Enhancements
1. **Dynamic Clustering**: Update clusters as new documents are added
2. **Query Expansion**: Add related terms to improve recall
3. **Learning to Rank**: Use machine learning to improve ranking
4. **Real-time Updates**: Support for live document collections
5. **Multi-language Support**: Extend to non-English documents

### Performance Optimizations
1. **Indexing**: Use inverted indexes for faster retrieval
2. **Caching**: Cache frequent queries and results
3. **Parallel Processing**: Distribute computation across multiple cores
4. **Memory Management**: Optimize for large document collections

## 📚 References and Further Reading

### Academic Papers
- Robertson, S. E., & Jones, K. S. (1976). Relevance weighting of search terms
- Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009). Reciprocal rank fusion outperforms condorcet and individual rank learning methods

### Books
- Manning, C. D., Raghavan, P., & Schütze, H. (2008). Introduction to Information Retrieval
- Croft, W. B., Metzler, D., & Strohman, T. (2010). Search Engines: Information Retrieval in Practice

### Online Resources
- [Information Retrieval Course](https://web.stanford.edu/class/cs276/)
- [BM25 Algorithm Explanation](https://en.wikipedia.org/wiki/Okapi_BM25)
- [K-Means Clustering](https://scikit-learn.org/stable/modules/clustering.html#k-means)

## 🤝 Contributing

This project is designed for educational purposes. Feel free to:
- Experiment with different clustering algorithms
- Try other ranking methods
- Test on different datasets
- Improve the evaluation metrics
- Add new features

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Datasets**: CACM and CISI datasets from standard IR test collections
- **Libraries**: NLTK, scikit-learn, rank-bm25, and other open-source tools
- **Community**: Information retrieval research community for established methods and metrics

---

**Note**: This project is designed for educational purposes to demonstrate information retrieval concepts. For production use, additional considerations like scalability, security, and robustness would be necessary.