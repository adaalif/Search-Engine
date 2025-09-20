## 🎯 Overview Sistem

Sistem ini mengimplementasikan pipeline Information Retrieval yang komprehensif dengan komponen-komponen berikut:

1. **Document Preprocessing Pipeline** - Normalisasi dan pembersihan teks
2. **Multi-field BM25 Ranking** - Peringkat relevansi berbasis BM25 untuk multiple fields
3. **Document Clustering** - Pengelompokan dokumen menggunakan K-Means dengan TF-IDF
4. **Query Processing** - Preprocessing dan koreksi typo otomatis
5. **Reciprocal Rank Fusion (RRF)** - Penggabungan skor dari multiple ranking systems
6. **Clustering-based Relevance Boosting** - Peningkatan relevansi berdasarkan cluster similarity
7. **Web Interface & API** - UI pengguna dan RESTful API

## 🚀 Fitur Utama

### Core Search Features
- **Multi-Field BM25 Ranking**: Implementasi BM25 untuk title, abstract, dan keyphrases dengan scoring terpisah
- **Document Clustering**: Pengelompokan dokumen menggunakan K-Means clustering dengan TF-IDF vectorization
- **Reciprocal Rank Fusion (RRF)**: Penggabungan skor dari multiple ranking systems untuk hasil yang lebih akurat
- **Clustering-based Relevance Boosting**: Peningkatan skor relevansi untuk dokumen dalam cluster yang relevan
- **Automatic Typo Correction**: Koreksi kesalahan ketik menggunakan Levenshtein Distance algorithm

### Advanced Features
- **Multi-field Search**: Pencarian simultan di title, abstract, dan keyphrases
- **Similar Document Discovery**: Temukan dokumen serupa berdasarkan cluster membership
- **Real-time Query Processing**: Processing query dengan preprocessing dan normalization
- **Comprehensive Statistics**: Monitoring dan analisis performa sistem

### Web Interface & API
- **Modern Web UI**: Interface pengguna yang responsif dan user-friendly
- **RESTful API**: Endpoint API lengkap untuk integrasi dengan sistem lain
- **Document Detail View**: Halaman khusus untuk melihat detail lengkap dokumen
- **Similar Documents View**: Tampilan dokumen serupa berdasarkan clustering
- **Real-time Statistics**: Monitoring statistik sistem secara real-time

## Cara Menjalankan

1.  **Clone atau unduh repositori ini.**

2.  **Masuk ke direktori proyek:**

    ```bash
    cd Information retrieaval
    ```

3.  **Instal semua pustaka yang dibutuhkan:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Jalankan server aplikasi:**

    ```bash
    # Cara mudah
    python start.py

    # Atau langsung
    python main.py
    ```

5.  **Buka browser** dan akses `http://localhost:8000`.

## Cara Penggunaan

### UI Web

1.  **Mencari**: Masukkan query Anda pada kotak pencarian yang tersedia.
2.  **Lihat Hasil**: Hasil pencarian akan ditampilkan dalam daftar yang sudah diperingkat.
3.  **Lihat Dokumen**: Klik tombol "Lihat Dokumen Lengkap" untuk membaca konten penuh.
### Endpoint API

  - `GET /` - Halaman utama.
  - `POST /search` - Endpoint untuk pencarian via form HTML.
  - `GET /api/search?query=your_query&top_n=10` - Mendapatkan hasil pencarian dalam format JSON.
  - `GET /document/{doc_id}` - Mengambil detail dokumen berdasarkan ID.
  - `GET /stats` - Mengambil data statistik.

### Contoh Penggunaan API

```bash
# Mencari dokumen
curl "http://localhost:8000/api/search?query=computer%20vision&top_n=5"

# Mengambil detail dokumen
curl "http://localhost:8000/document/1103"

# Mengambil statistik
curl "http://localhost:8000/stats"
```

## 🔬 Pipeline Information Retrieval

Sistem ini mengimplementasikan pipeline Information Retrieval yang komprehensif dengan dua tahap utama: **Tahap Inisialisasi** dan **Tahap Pencarian**.

### 📋 Tahap Inisialisasi

#### 1. Pemuatan Dataset
- **Langkah awal**: Memuat keseluruhan Dataset INSPEC dari repositori Hugging Face ke dalam memori utama
- **Dataset**: 2000 dokumen ilmiah dengan field title, abstract, dan keyphrases
- **Format**: Dataset dikonversi ke format list untuk memudahkan processing

#### 2. Preprocessing Korpus
Setiap dokumen dalam dataset diproses dengan mengekstrak teks dari tiga field utama:
- **Title**: Judul dokumen
- **Keyphrases**: Kata kunci yang terkait
- **Abstract**: Ringkasan isi dokumen

Teks dari setiap field dinormalisasi secara independen melalui serangkaian prosedur preprocessing standar:
- **Tokenisasi**: Memecah teks menjadi token menggunakan NLTK word_tokenize
- **Case Folding**: Konversi semua karakter ke lowercase
- **Eliminasi Stop Words**: Menghapus kata-kata umum yang tidak relevan (the, and, or, dll)
- **Stemming**: Mengubah kata ke bentuk dasarnya menggunakan Porter Stemmer
- **Filtering**: Hanya mempertahankan token yang berupa alfabet

#### 3. Penyusunan Kosakata (Vocabulary)
- Seluruh token unik yang dihasilkan dari proses preprocessing pada ketiga field dikumpulkan
- Membentuk satu kosakata (vocabulary) referensi yang komprehensif
- Kosakata berfungsi sebagai kamus untuk validasi dan koreksi ejaan pada tahap pencarian

#### 4. Inisialisasi Model Ranking BM25
Berdasarkan korpus token yang telah diproses, tiga model ranking Okapi BM25 diinisialisasi secara terpisah:
- **BM25 Title**: Model khusus untuk field title
- **BM25 Keyphrases**: Model khusus untuk field keyphrases  
- **BM25 Abstract**: Model khusus untuk field abstract

Setiap model dilatih secara spesifik pada data dari field masing-masing, menghasilkan tiga model pemeringkatan yang terspesialisasi.

#### 5. Persiapan Model Clustering (Vektorisasi TF-IDF)
Secara paralel, model TF-IDF (Term Frequency-Inverse Document Frequency) dibangun:
- **Input**: Kombinasi teks dari title + abstract + keyphrases untuk setiap dokumen
- **Output**: Representasi vektor numerik untuk setiap dokumen
- **Parameter**: max_features=1000, ngram_range=(1,2), stop_words='english'
- **Tujuan**: Menghasilkan vektor untuk tahap pengelompokan (clustering)

#### 6. Document Clustering dengan K-Means
- **Algoritma**: K-Means clustering dengan TF-IDF vectors sebagai input
- **Parameter**: n_clusters=10, random_state=42, n_init=10
- **Output**: Label cluster untuk setiap dokumen
- **Mapping**: Dibuat mapping cluster-to-documents untuk efisiensi pencarian

### 🔍 Tahap Pencarian

#### 1. Penerimaan dan Koreksi Ejaan Query
Saat pengguna mengirimkan query:
- **Validasi Token**: Setiap token divalidasi terhadap kosakata yang telah disusun
- **Levenshtein Distance**: Jika token tidak ditemukan, algoritma menghitung jarak edit karakter
- **Koreksi Otomatis**: Token digantikan dengan kata dari kosakata yang memiliki jarak edit terkecil
- **Threshold**: Koreksi hanya dilakukan jika jarak edit ≤ 3 karakter

#### 2. Preprocessing Query
Query yang telah melalui koreksi ejaan diproses menggunakan alur preprocessing yang identik dengan dokumen:
- **Tokenisasi**: Memecah query menjadi token
- **Case Folding**: Konversi ke lowercase
- **Stop Word Removal**: Menghapus stop words
- **Stemming**: Mengubah ke bentuk dasar
- **Konsistensi**: Menjamin format token query sama dengan token dalam indeks

#### 3. Peringkat Multi-Field
Token query yang telah diproses digunakan untuk pencarian pada ketiga model BM25:
- **Title Search**: Pencarian pada indeks title dengan scoring BM25
- **Keyphrases Search**: Pencarian pada indeks keyphrases dengan scoring BM25
- **Abstract Search**: Pencarian pada indeks abstract dengan scoring BM25
- **Independent Ranking**: Setiap model menghasilkan daftar peringkat dokumen secara independen

#### 4. Fusi Peringkat dengan RRF (Reciprocal Rank Fusion)
Ketiga daftar peringkat digabungkan menggunakan algoritma RRF:
- **Formula RRF**: `score = 1/(k + rank)` dimana k=60 dan rank adalah posisi dokumen
- **Score Combination**: Skor final dihitung berdasarkan posisi peringkat, bukan skor mentah BM25
- **Stability**: Menghasilkan peringkat gabungan yang lebih stabil dan andal
- **Multi-field Integration**: Menggabungkan relevansi dari ketiga field secara proporsional

#### 5. Clustering-based Relevance Boosting
- **Cluster Identification**: Mengidentifikasi cluster yang relevan dengan query menggunakan cosine similarity
- **Threshold**: Cluster dengan similarity > 0.1 dianggap relevan
- **Boost Factor**: Dokumen dalam cluster relevan mendapat boost 1.2x
- **Relevance Enhancement**: Meningkatkan skor dokumen yang tematiknya sesuai dengan query

#### 6. Penyajian Hasil
Hasil pencarian yang telah diperingkat disajikan dengan informasi tambahan:
- **Ranked Results**: Daftar dokumen yang telah diperingkat berdasarkan skor RRF + clustering boost
- **Cluster Information**: Label cluster untuk setiap dokumen
- **Score Details**: Skor relevansi dan informasi cluster
- **Document Metadata**: Title, abstract, keyphrases, dan ID dokumen
- **Similar Documents**: Kemampuan untuk menemukan dokumen serupa dalam cluster yang sama

## 📊 Dataset

Mesin pencari ini menggunakan dataset **INSPEC** dari Hugging Face, yang berisi:

- **Jumlah Dokumen**: 2000 dokumen ilmiah
- **Field Structure**: 
  - `title`: Judul dokumen
  - `abstract`: Ringkasan isi dokumen  
  - `keyphrases`: Array kata kunci yang terkait
  - `id`: Unique identifier untuk setiap dokumen
- **Domain**: Topik seputar ilmu komputer dan teknik
- **Language**: Bahasa Inggris
- **Source**: Hugging Face Datasets Hub (taln-ls2n/inspec)

## 🏗️ Arsitektur Sistem

### Backend Architecture
- **Framework**: FastAPI dengan async support
- **Search Engine**: Custom BM25_RRF_Clustered_SearchEngine class
- **Clustering**: scikit-learn K-Means dengan TF-IDF vectorization
- **Text Processing**: NLTK untuk tokenization, stemming, dan stop word removal
- **Distance Algorithm**: python-Levenshtein untuk typo correction

### Frontend Architecture  
- **Template Engine**: Jinja2Templates
- **Static Files**: CSS dan JavaScript untuk UI enhancement
- **Responsive Design**: Modern web interface yang mobile-friendly
- **API Integration**: RESTful API endpoints untuk data exchange

### Data Flow
```
User Query → Typo Correction → Preprocessing → Multi-field BM25 Search → 
RRF Score Fusion → Clustering Boost → Ranked Results → Web Display
```

## 🔧 Konfigurasi Sistem

### Parameter yang Dapat Dikonfigurasi
- **`top_n`**: Jumlah hasil yang ditampilkan (default: 10)
- **`k`**: Parameter untuk RRF formula (default: 60)
- **`n_clusters`**: Jumlah cluster untuk K-Means (default: 10)
- **`typo_threshold`**: Maksimal jarak edit untuk koreksi typo (default: 3)
- **`cluster_boost`**: Faktor boost untuk dokumen dalam cluster relevan (default: 1.2)
- **`similarity_threshold`**: Threshold similarity untuk cluster relevansi (default: 0.1)

### Performance Optimization
- **Memory Management**: Dataset dimuat ke memori untuk akses cepat
- **Index Caching**: BM25 indices dan TF-IDF vectors di-cache
- **Cluster Mapping**: Pre-computed cluster-to-documents mapping
- **Vocabulary Lookup**: Hash-based vocabulary lookup untuk efisiensi

## Struktur Proyek

```
Information retrieaval/
├── main.py             # Aplikasi utama FastAPI
├── start.py            # Skrip sederhana untuk menjalankan
├── requirements.txt    # Daftar dependensi Python
├── README.md           # Dokumentasi ini
├── templates/          # Template HTML
│   ├── index.html
│   ├── search_results.html
│   └── document.html
└── static/             # File statis (CSS, JS)
    ├── css/
    │   └── style.css
    └── js/
        └── main.js
```

## 📈 Evaluasi dan Analisis Performa

### Metrik Evaluasi
- **Precision@K**: Akurasi hasil pencarian pada K dokumen teratas
- **Recall**: Kemampuan sistem menemukan dokumen relevan
- **F1-Score**: Harmonic mean dari precision dan recall
- **Silhouette Score**: Kualitas clustering (range: -1 to 1, higher is better)
- **Query Response Time**: Waktu rata-rata untuk memproses query

### Analisis Clustering
- **Cluster Distribution**: Analisis distribusi dokumen dalam setiap cluster
- **Cluster Coherence**: Konsistensi tematik dalam setiap cluster
- **Inter-cluster Similarity**: Kemiripan antar cluster
- **Intra-cluster Similarity**: Kemiripan dalam cluster

### Benchmarking
- **Baseline Comparison**: Perbandingan dengan pencarian tanpa clustering
- **RRF Effectiveness**: Analisis efektivitas Reciprocal Rank Fusion
- **Typo Correction Accuracy**: Akurasi koreksi kesalahan ketik
- **Multi-field Integration**: Kontribusi setiap field terhadap relevansi

## 🔬 Detail Teknis

### Core Algorithms
- **BM25 Ranking**: Okapi BM25 dengan parameter k1=1.2, b=0.75
- **Reciprocal Rank Fusion**: Formula `score = 1/(k + rank)` dengan k=60
- **K-Means Clustering**: Euclidean distance dengan random initialization
- **TF-IDF Vectorization**: Term frequency-inverse document frequency weighting
- **Levenshtein Distance**: Edit distance untuk typo correction

### Dependencies
- **Framework**: FastAPI dengan template Jinja2
- **Machine Learning**: scikit-learn untuk clustering dan vectorization
- **NLP Processing**: NLTK untuk text preprocessing
- **Distance Calculation**: python-Levenshtein untuk typo correction
- **BM25 Implementation**: rank-bm25 untuk ranking algorithm
- **Data Source**: Hugging Face datasets untuk corpus

## Konfigurasi

Beberapa parameter bisa diubah langsung di dalam kode `main.py`:

  - `top_n`: Jumlah hasil yang ditampilkan (default: 10).
  - `k`: Parameter untuk RRF (default: 60).
  - Batas jarak untuk koreksi typo (default: ≤ 3).

## Penyelesaian Masalah (Troubleshooting)

1.  **Gagal Memuat Dataset**: Aplikasi akan mengunduh dataset saat pertama kali dijalankan. Pastikan Anda memiliki koneksi internet yang stabil.

2.  **Masalah Memori**: Dataset dimuat ke dalam memori. Untuk dataset yang jauh lebih besar, pertimbangkan untuk implementasi *lazy loading*.

3.  **Konflik Port**: Jika port 8000 sudah digunakan, ubah port di file `main.py`:

    ```python
    uvicorn.run(app, host="0.0.0.0", port=8001)
    ```

## 🎯 Kesimpulan

Sistem Information Retrieval ini berhasil mengimplementasikan pipeline pencarian yang komprehensif dengan menggabungkan multiple ranking algorithms, document clustering, dan advanced query processing. Kombinasi BM25 dengan RRF dan clustering-based boosting menghasilkan sistem pencarian yang lebih akurat dan relevan dibandingkan dengan pendekatan tradisional.

### Keunggulan Sistem
1. **Multi-field Search**: Pencarian simultan di title, abstract, dan keyphrases
2. **Intelligent Ranking**: RRF menggabungkan skor dari multiple ranking systems
3. **Document Clustering**: Pengelompokan dokumen untuk meningkatkan relevansi
4. **Typo Tolerance**: Koreksi otomatis kesalahan ketik pengguna
5. **Scalable Architecture**: Arsitektur yang dapat dikembangkan untuk dataset yang lebih besar

### Potensi Pengembangan
- **Query Expansion**: Penambahan sinonim dan related terms
- **Learning to Rank**: Implementasi machine learning untuk ranking
- **Real-time Clustering**: Update cluster secara real-time
- **Multi-language Support**: Dukungan untuk multiple bahasa
- **Advanced Analytics**: Dashboard analisis performa sistem

## 📚 Referensi

### Academic Papers
- Robertson, S., & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends in Information Retrieval*, 3(4), 333-389.
- Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009). Reciprocal rank fusion outperforms condorcet and individual rank learning methods. *Proceedings of the 32nd international ACM SIGIR conference on Research and development in information retrieval*.
- MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. *Proceedings of the fifth Berkeley symposium on mathematical statistics and probability*.

### Technical Documentation
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- [NLTK Documentation](https://www.nltk.org/)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/)

## 📄 Lisensi

Proyek ini dibuat untuk tujuan edukasi dan penelitian. Dataset INSPEC digunakan di bawah lisensi masing-masing. Kode sumber tersedia untuk pembelajaran dan pengembangan lebih lanjut.

---

**Dibuat dengan ❤️ untuk pembelajaran Information Retrieval dan Machine Learning**
