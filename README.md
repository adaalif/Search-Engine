

Sistem mesin pencari canggih berbasis web yang mengimplementasikan algoritma Information Retrieval modern dengan FastAPI. Proyek ini menggabungkan multiple ranking algorithms, document clustering, typo correction, dan multi-field search untuk menghasilkan hasil pencarian yang akurat dan relevan.

## 🎯 Overview Sistem

Sistem ini mengimplementasikan pipeline Information Retrieval yang komprehensif dengan komponen-komponen berikut:

1. **Document Preprocessing Pipeline** - Normalisasi dan pembersihan teks
2. **Multi-field BM25 Ranking** - Peringkat relevansi berbasis BM25 untuk multiple fields
3. **Document Clustering** - Pengelompokan dokumen menggunakan K-Means dengan TF-IDF
4. **Query Processing** - Preprocessing dan koreksi typo otomatis
5. **Reciprocal Rank Fusion (RRF)** - Penggabungan skor dari multiple ranking systems
6. **Clustering-based Relevance Boosting** - Peningkatan relevansi berdasarkan cluster similarity
7. **Web Interface & API** - Antarmuka pengguna dan RESTful API

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

### Antarmuka Web

1.  **Mencari**: Masukkan query Anda pada kotak pencarian yang tersedia.
2.  **Lihat Hasil**: Hasil pencarian akan ditampilkan dalam daftar yang sudah diperingkat.
3.  **Lihat Dokumen**: Klik tombol "Lihat Dokumen Lengkap" untuk membaca konten penuh.
4.  **Statistik**: Cek statistik mesin pencari di halaman utama.

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

## Cara Kerja Algoritma

Alur kerja mesin pencari ini dibagi menjadi beberapa tahap:

1.  **Preprocessing Teks**:

      - **Tokenisasi**: Memecah teks menjadi kata-kata menggunakan NLTK.
      - **Penghapusan Stopword**: Menghapus kata-kata umum yang tidak relevan (contoh: "yang", "di", "dan").
      - **Stemming**: Mengubah kata ke bentuk dasarnya menggunakan Porter Stemmer.

2.  **Koreksi Typo**:

      - Menggunakan Levenshtein Distance untuk mencari kata terdekat dari kamus internal.
      - Mengoreksi kata jika jarak perubahannya ≤ 3.

3.  **Skoring dengan BM25**:

      - Terdapat tiga indeks BM25 terpisah untuk *title*, *abstract*, dan *keyphrases*.
      - Skor relevansi dihitung untuk setiap field secara independen.

4.  **Reciprocal Rank Fusion (RRF)**:

      - Menggabungkan skor dari ketiga field menggunakan formula RRF.
      - Dokumen diperingkat ulang berdasarkan skor gabungan untuk hasil akhir.

## Dataset

Mesin pencari ini menggunakan dataset **INSPEC** dari Hugging Face, yang berisi:

  - Dokumen-dokumen ilmiah dengan judul, abstrak, dan kata kunci.
  - 2000 jumlah dokumen.
  - Topik seputar ilmu komputer dan teknik.

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

## Detail Teknis

  - **Framework**: FastAPI dengan template Jinja2
  - **Algoritma Pencarian**: BM25 dengan RRF
  - **Koreksi Typo**: Levenshtein Distance
  - **Sumber Data**: Pustaka `datasets` dari Hugging Face

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

## Lisensi

Proyek ini dibuat untuk tujuan edukasi. Dataset INSPEC digunakan di bawah lisensi masing-masing.
