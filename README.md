# Simulasi Mesin Pencari dengan FastAPI

Ini adalah proyek simulasi mesin pencari berbasis web yang saya bangun menggunakan FastAPI. Proyek ini mengimplementasikan algoritma ranking BM25, penggabungan skor dengan Reciprocal Rank Fusion (RRF), serta fitur koreksi salah ketik (typo) secara otomatis.

## Fitur Utama

  -  **Pencarian Lanjutan Multi-Field-BM25**: Peringkat relevansi berbasis BM25 yang diterapkan pada beberapa field dokumen (judul, abstrak, kata kunci).
  -  **Koreksi Typo**: Koreksi kesalahan ketik secara otomatis saat pengguna mencari, menggunakan algoritma Levenshtein Distance.
  -  **Penggabungan Skor RRF**: Menggabungkan skor relevansi dari setiap field untuk menghasilkan peringkat akhir yang lebih akurat.
  -  **UI Web Sederhana**: Tampilan UI untuk memudahkan penggunaan.
  -  **Tampilan Dokumen**: Halaman khusus untuk melihat detail lengkap dari sebuah dokumen.
  -  **Endpoint API**: Menyediakan RESTful API untuk akses  ke fungsi pencarian.

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
