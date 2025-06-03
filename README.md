# *Prediksi Penyakit Diabetes Menggunakan Machine Learning*
---
## Domain Proyek

Perkembangan layanan streaming video seperti Netflix, Disney+, dan Amazon Prime membuat jumlah film dan serial TV yang tersedia sangat banyak. Pengguna sering bingung memilih tontonan yang sesuai preferensi mereka. Sistem rekomendasi yang memanfaatkan rating pengguna saja terkadang kurang akurat karena bisa bias dan tidak mempertimbangkan sentimen review serta tren terkini di sosial media. Oleh karena itu, proyek ini bertujuan membangun sistem rekomendasi yang menggabungkan data rating, analisis sentimen review film/series, dan tren sosial media untuk memberikan rekomendasi yang lebih relevan dan up-to-date.

referensi:


*   [Sistem Rekomendasi Film Menggunakan Content Based Filtering](https://j-ptiik.ub.ac.id/index.php/j-ptiik/article/view/9163)
*   [Sistem Rekomendasi Film Menggunakan Metode Hybrid Collaborative Filtering Dan Content-based Filtering](https://openlibrarypublications.telkomuniversity.ac.id/index.php/engineering/article/view/18066)

---
## Business Understanding

### Problem Statements

1. Pengguna kesulitan memilih film yang sesuai dengan selera mereka di platform streaming.
2. Data rating saja tidak cukup untuk memberikan rekomendasi yang relevan karena bisa bias.
3. Tren sosial media belum dimanfaatkan secara maksimal dalam sistem rekomendasi.

### Goals

1. Membangun sistem rekomendasi film yang menggabungkan rating pengguna, analisis sentimen review, dan tren sosial media.
2. Memastikan rekomendasi yang diberikan lebih relevan dengan selera pengguna dan tren terkini.
3. Mengurangi bias dalam rekomendasi dengan memanfaatkan data sosial media.

### Solution Approach

1. **Content-Based Filtering**: Menggunakan informasi yang ada di film, seperti genre dan deskripsi, untuk merekomendasikan film yang serupa dengan yang sudah dilihat pengguna.
2. **Collaborative Filtering**: Menggunakan data rating dari pengguna lain untuk memberikan rekomendasi berdasarkan kesamaan preferensi antar pengguna.
---
## Data Understanding

### Deskripsi Dataset

Dataset ini berisi data mengenai 4 data yaitu data tentang Movies, Rating, serta Tag yang telah diisikan oleh User. dengan total movies sebanyak 9742 data, ratings sebanyak 100836 data, dan tags sebanyak ... data

### Sumber Dataset
Dataset ini dapat ditemukan di [Grouplens - Movielens](https://grouplens.org/datasets/movielens/).

### Variabel pada Dataset

- **movies.csv**:
  - `movieId`: ID unik untuk setiap film.
  - `title`: Judul film.
  - `genres`: Kategori genre yang dimiliki film, dipisah dengan tanda `|`.
- **ratings.csv**:
  - `userId`: ID unik untuk setiap pengguna.
  - `movieId`: ID film yang diberi rating oleh pengguna.
  - `rating`: Rating yang diberikan pengguna terhadap film (skala 1-5).
  - `timestamp`: Waktu pengguna memberikan rating.
- **tags.csv**:
  - `userId`: ID pengguna yang memberikan tag.
  - `movieId`: ID film yang diberi tag.
  - `tag`: Kata kunci atau tag yang diberikan pengguna.
  - `timestamp`: Waktu pemberian tag.
---
## EDA (Exploratory Data Analysist)

### Exploratory Data Analysis (EDA) Variabel Movies

![Movies Variable](img/movies_info.jpg)

Variabel `movies` berisi informasi mengenai film-film dalam dataset. Informasi ini menjadi dasar dalam content-based filtering untuk mengidentifikasi kemiripan antar film berdasarkan atribut film.

#### Struktur Data `movies`

- **movieId**: ID unik film.
- **title**: Judul film lengkap termasuk tahun rilis.
- **genres**: Genre film yang dipisah dengan tanda `|`.


#### Statistik dan Pemrosesan

![Movies Describe](img/movies_info.jpg)

- Total film unik sebanyak **9742**.
- Judul film unik sekitar **9737**.
- Tidak ada data duplikat.
- Genre populer adalah Drama, Comedy, Thriller, Action, Romance, dll.

#### Top Genre

![Top Genre](img/movies_top_genre.jpg)

berdasarkan informasi diatas , top 10 genre yang paling banyak ditampilkan :

1.   Drama dengan **4361** movie
2.   Comedy dengan **3756** movie
3.   Thriller dengan **1894** movie
4.   Action dengan **1828** movie
5.   Romance dengan **1596** movie
6.   Adventure dengan **1263** movie
7.   Crime dengan **1199** movie
8.   Sci-Fi dengan **980** movie
9.   Horror dengan **978** movie
10.  Fantasy dengan **779** movie

---

### Exploratory Data Analysis (EDA) Variabel Ratings

Variabel `ratings` berisi data penilaian yang diberikan oleh pengguna terhadap film yang telah mereka tonton. Informasi ini menjadi dasar utama dalam membangun model rekomendasi berbasis collaborative filtering.

#### Struktur Data `ratings`

![Ratings Variable](img/ratings_info.jpg)

Dataset `ratings` memiliki kolom-kolom berikut:

- **userId**: Identifikasi unik untuk setiap pengguna.
- **movieId**: Identifikasi unik untuk setiap film yang diberikan rating.
- **rating**: Nilai penilaian yang diberikan, berkisar antara 1 hingga 5.
- **timestamp**: Waktu saat rating diberikan (tidak digunakan dalam pemodelan).

#### Statistik dan Karakteristik Data

![Ratings Describe](img/ratings_describe.jpg)

- Dataset berisi rating dari **610 pengguna unik**.
- Terdapat rating untuk **9724 film unik**.
- Distribusi rating menunjukkan nilai rata-rata sekitar **3.5**, dengan rentang minimum 1 dan maksimum 5.
- Tidak ditemukan data duplikat pada dataset rating.
- Setiap pengguna memberikan rating pada beberapa film berbeda dan tidak ada pengguna yang memberi rating lebih dari satu kali pada film yang sama.


#### Persebaran Ratings
![Ratings Distribution](img/ratings_persebaran.jpg)

Persebaran rating menunjukkan bahwa paling banyak movies mendapatkan rating di angka 4 yang mencapai 25.000+ movies, ini menunjukkan bahwa user sebagian besar memberikan penilaian yang baik.

---



### Exploratory Data Analysis (EDA) Variabel Tags

#### Struktur Data `tags`

![Tags Variable](img/tags_info.jpg)

- **userId**: ID pengguna.
- **movieId**: ID film.
- **tag**: Kata kunci atau frasa deskriptif.
- **timestamp**: Waktu tag diberikan (tidak dipakai dalam pemodelan).

#### Statistik
![Tag Distribution](img/tags_distribution.jpg)

- Terdapat **58 pengguna unik** yang memberikan tag.
- Tag diberikan untuk **1572 film** dengan total **1589 tag**.
- Pengguna bisa memberikan lebih dari satu tag untuk film yang sama.
- Tidak ada data duplikat pada dataset ini.

---

### Kesimpulan

Analisis EDA pada ketiga variabel utama (`ratings`, `movies`, dan `tags`) memberikan gambaran menyeluruh tentang data yang digunakan dalam sistem rekomendasi. Variabel `ratings` menyediakan interaksi pengguna-film sebagai dasar collaborative filtering. Variabel `movies` memberikan atribut film penting untuk content-based filtering. Sedangkan variabel `tags` memperkaya fitur dengan deskripsi tambahan dari pengguna. Semua variabel ini dipersiapkan secara menyeluruh melalui preprocessing agar dapat digunakan secara optimal dalam pemodelan rekomendasi.

## Data Preparation

### Movies Variable

#### 1. Memisahkan value genre kedalam bentuk list

```
# Memisahkan genres kedalam daftar list

movies['genres'] = movies['genres'].str.split('|')

```
Hasil : 

![Movie Genre Value](img/movies_prep_genre.jpg)

`genre` dipisahkan menjadi dalam bentuk **list**

#### 2. Memisahkan Title dan Year

```

movies['year'] = movies['title'].str.extract(r'\((\d{4})\)', expand=False)
movies['title'] = movies['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True).str.strip()

```
Hasil: 

![Movie Title Year](img/movies_prep_title.jpg)


Tahun rilis diekstrak dari kolom `title` menggunakan regular expression untuk menangkap angka tahun dalam tanda kurung, contohnya `(1995)`.  
Judul film kemudian dibersihkan dari tahun agar menjadi teks yang lebih bersih dan mudah diproses.

#### 3. Memberikan nilai kosong pada kolom 'year'

```
movies['year'] = movies['year'].fillna(0).astype(int)

```

![Movie Year Missing](img/movies_prep_year_missing.jpg)

Film yang tidak memiliki informasi tahun akan diisi dengan nilai `0` sebagai placeholder agar tipe data tahun konsisten dalam bentuk integer.



### Ratings Variable

#### 1. Drop kolom timestamp

```

ratings = ratings.drop(['timestamp'], axis=1)

```
Hasil:
![Ratings Drop Timestamp](img/ratings_prep_drop.jpg)

- Kolom `timestamp` yang tidak relevan untuk pemodelan dihapus.

### Tags Variable

![Tags Users Agg](img/)



```

tags_agg = tags.groupby(['userId', 'movieId'])['tag'].apply(lambda x: ','.join(x.unique())).reset_index()

```



---



## Modelling

### 1. Logistic Regression

![Logistic Regression](img/lr_matrix.jpg)
![Logistic Regression](img/lr_confusion.png)

Model pertama yang dilatih adalah **Logistic Regression** sebagai baseline. Model ini memprediksi probabilitas kelas target menggunakan **fungsi sigmoid/logistik**:

$$
P(y=1|x) = \frac{1}{1 + e^{-(wX + b)}}
$$

di mana:

* $w$: bobot fitur
* $X$: vektor fitur
* $b$: bias/intersep

**Parameter yang digunakan:**

```python
LogisticRegression(random_state=42)
```
* `random_state=42`: menjaga konsistensi hasil.

- Kelebihan:

  * Cepat dan efisien.
  * Mudah diinterpretasikan.
  * Cocok untuk relasi linier.

- Kekurangan:

  * Kurang efektif untuk pola non-linear.
  * Sensitif terhadap multikolinearitas.

pada model `Logistic Regression` hasil akurasi didapatkan mencapai **90%** dengan masing masing :

- pada **class** 0 
  
  `presisi` : 67%

  `recall`  : 86%

  `f1-score` : 75%

- pada **class** 1 
  
  `presisi` : 27%

  `recall`  : 67%

  `f1-score` : 38%

- pada **class** 2 
  
  `presisi` : 100%

  `recall`  : 67%

  `f1-score` : 38%


Hasil ini menunjukkan bahwa model ini belum bisa mengenali pola secara merata khususnya untuk **class** bernilai 1 / Prediabetes

### 2. Random Forest

![Random Forest](img/rf_matrix.jpg)
![Random Forest](img/rf_confusion.png)

Model ensambel berbasis pohon keputusan yang menggunakan teknik **bagging (Bootstrap Aggregating)**. Setiap decision tree dilatih dari subset acak data dan subset acak fitur, dan prediksi akhir diambil melalui **voting mayoritas** dari semua pohon.

$$
\hat{y} = \text{mode}(h_1(x), h_2(x), \dots, h_T(x))
$$

di mana:

* $h_t(x)$: prediksi dari pohon ke-$t$
* $T$: jumlah total pohon dalam hutan
* `mode`: memilih prediksi yang paling sering muncul

**Parameter yang digunakan:**

```python
RandomForestClassifier(random_state=42)
```

* `random_state=42`: menjaga replikasi hasil.
* `n_estimators=100` digunakan sebagai default (jumlah pohon) Tidak disebutkan secara eksplisit.

- Kelebihan:

  * Tahan terhadap overfitting.
  * Mampu menangani fitur yang berinteraksi dan non-linear.
  * Memberikan feature importance.

- Kekurangan:

  * Interpretasi kompleks.
  * Waktu inferensi dan konsumsi memori lebih tinggi.

pada model `Random Forest` hasil akurasi didapatkan mencapai **98.5** dengan masing masing :

- pada **class** 0 
  
  `presisi` : 88%

  `recall`  : 100%

  `f1-score` : 93%

- pada **class** 1 
  
  `presisi` : 100%

  `recall`  : 100%

  `f1-score` : 100%

- pada **class** 2 
  
  `presisi` : 100%

  `recall`  : 98%

  `f1-score` : 99%


Hasil ini menunjukkan bahwa model ini sudah cukup baik untuk mengenali pola dari masing-masing nilai pada target.

## Evaluasi

![Evaluasi](img/evaluasi.jpg)

Hasil Evaluasi ini menggunakan matrix `Accuracy` , `Precision` , `Recall` , dan `F1-Score`
1. **Accuracy**: Mengukur proporsi prediksi yang benar dibandingkan total prediksi.

   $Accuracy = rac{TP + TN}{TP + TN + FP + FN}$

2. **Precision**: Proporsi prediksi positif yang benar dari seluruh prediksi positif.

   $Precision = rac{TP}{TP + FP}$

3. **Recall**: Proporsi positif yang berhasil diprediksi dengan benar.

   $Recall = rac{TP}{TP + FN}$

4. **F1-Score**: Harmonik dari precision dan recall.

   $F1 = 2 * rac{Precision * Recall}{Precision + Recall}$

perbandingan antara `Logistic Regression` dan `Random Forest` menunjukkan bahwa `Random Forest` lebih baik tingkat akurasinya yaitu sebesar **98.5 %** , diikuti dengan matrix yang lain yang lebih tinggi menandakan bahwa `Random Forest` merupakan model yang lebih baik dibanding `Logistic Regression` pada kasus ini.