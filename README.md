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

![Tags Users Agg](img/tags_prep_info_user.jpg)

Fungsi diatas untuk mengetahui apakah `users` dapat memberikan `tags` pada `movies` yang sama lebih dari satu atau tidak (unique).
hasil menunjukkan bahwa `max()` mendapati nilai **173**, hal ini menunjukkan bahwa `users` dapat memberikan `tags` lebih dari 1 pada `movies` yang sama.


```

tags_agg = tags.groupby(['userId', 'movieId'])['tag'].apply(lambda x: ','.join(x.unique())).reset_index()

```

Hasil : 
![After Agg](img/tags_prep_after_agg.jpg)

Hasil diatas pada kolom `tags` memuat kumpulan `tags` yang diberikan `users` pada `movies` yang sama.

---

### Menggabungkan *ratings* dengan *tags*

```

ratings_tags = pd.merge(ratings, tags_agg, on=['userId', 'movieId'], how='left')
ratings_tags['tag'] = ratings_tags['tag'].fillna('no_tag')

```

Hasil:

![Combine Ratings and Tags](img/combine_ratings_tags.jpg)

Code diatas menggabungkan 2 df yaitu `ratings` dan `tags_agg` menjadi satu df dengan kunci yaitu **userId** dan **movieId** left join (**mempertahankan df sebelah kiri (ratings)**), kemudian mengisi missing value kolom `tags`, `movie` yang tidak memiliki nilai `tag` dengan **no_tag**



### Menggabungkan All Data

```

full_data = pd.merge(ratings_tags, movies, on='movieId', how='left')


```
Hasil:
![Combine All Data](img/combine_all_data.jpg)

Hasil diatas menggabungkan 3 df yaitu `ratings_tags` dan `movies` dengan kunci `movieId` left join (**mempertahankan df sebelah kiri (ratings_tags)**)


## Modelling

### Content - Base Filtering

#### TF-IDF

```
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
tfidf_matrix = tfidf_vectorizer.fit_transform(tags_per_movie['tag'])

print(f"Shape TF-IDF matrix: {tfidf_matrix.shape}")

```
Menerapkan `max_features` sebanyak 1000 agar fitur yang digunakan lebih spesifik

#### Cosines Similarity

```

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


```

#### Inferensi

```

def get_recommendations_by_tag(movie_id, top_n=5):
    # Cari indeks film
    idx = tags_per_movie[tags_per_movie['movieId'] == movie_id].index[0]
    
    # Hitung similarity film lain
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Urutkan berdasarkan similarity tertinggi
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Ambil top-n film paling mirip, kecuali film itu sendiri
    sim_scores = sim_scores[1:top_n+1]
    
    # Ambil indeks film rekomendasi
    movie_indices = [i[0] for i in sim_scores]
    
    # Ambil movieId dan judul film dari dataframe movies (yang sudah kamu siapkan)
    recommended_ids = tags_per_movie.iloc[movie_indices]['movieId'].values
    recommended_movies = movies[movies['movieId'].isin(recommended_ids)][['movieId', 'title']].drop_duplicates()
    
    return recommended_movies


def recommend_by_title_tag():
    movie_name = input("Masukkan judul film: ").strip().lower()
    
    # Cari movieId berdasarkan judul (case insensitive)
    matched = movies[movies['title'].str.lower() == movie_name]
    
    if matched.empty:
        print(f"Film dengan judul '{movie_name}' tidak ditemukan. Coba lagi.")
        return
    
    movie_id = matched.iloc[0]['movieId']
    print(f"Film yang dipilih: {matched.iloc[0]['title']}")
    
    recommendations = get_recommendations_by_tag(movie_id)
    
    if recommendations.empty:
        print("Tidak ada rekomendasi yang tersedia.")
    else:
        print("Rekomendasi film serupa berdasarkan tag:")
        for idx, row in recommendations.iterrows():
            print(f"- {row['title']} (movieId: {row['movieId']})")

```

![Hasil CBF](img/cbf_hasil.jpg)


Fungsi tersebut menghasilkan 5 rekomendasi terbaik berdasarkan tag yang diberikan


### Collaborative Filterring

#### Mendefinisikan Model Neural Network Embedding

```
class RecommenderNet(tf.keras.Model):
  def __init__(self, num_users, num_movies, embedding_size, **kwargs):
    super(RecommenderNet, self).__init__(**kwargs)
    self.user_embedding = layers.Embedding(
        num_users,
        embedding_size,
        embeddings_initializer='he_normal',
        embeddings_regularizer=keras.regularizers.l2(1e-6)
    )
    self.user_bias = layers.Embedding(num_users, 1)
    self.movies_embedding = layers.Embedding(
        num_movies,
        embedding_size,
        embeddings_initializer='he_normal',
        embeddings_regularizer=keras.regularizers.l2(1e-6)
    )
    self.movies_bias = layers.Embedding(num_movies, 1)

  def call(self, inputs):
    user_vector = self.user_embedding(inputs[:, 0])
    user_bias = self.user_bias(inputs[:, 0])
    movies_vector = self.movies_embedding(inputs[:, 1])
    movies_bias = self.movies_bias(inputs[:, 1])
    dot_user_movies = tf.tensordot(user_vector, movies_vector, 2)
    x = dot_user_movies + user_bias + movies_bias
    return tf.nn.sigmoid(x)
```
Model ini menggunakan embedding untuk merepresentasikan user dan film ke dalam ruang fitur berdimensi rendah. Prediksi rating dilakukan dengan mengalikan embedding user dan film, ditambah bias masing-masing, lalu diproses sigmoid agar output dalam rentang [0,1].

#### Mapping User dan Movie

```
user_ids = df_rating['userId'].unique().tolist()
user_to_index = {user_id: index for index, user_id in enumerate(user_ids)}
index_to_user = {index: user_id for index, user_id in enumerate(user_ids)}

movie_ids = df_rating['movieId'].unique().tolist()
movie_to_index = {movie_id: index for index, movie_id in enumerate(movie_ids)}
index_to_movie = {index: movie_id for index, movie_id in enumerate(movie_ids)}

x_train[:, 0] = [user_to_index[user_id] for user_id in x_train[:, 0]]
x_val[:, 0] = [user_to_index[user_id] for user_id in x_val[:, 0]]

x_train[:, 1] = [movie_to_index[movie_id] for movie_id in x_train[:, 1]]
x_val[:, 1] = [movie_to_index[movie_id] for movie_id in x_val[:, 1]]

```

Karena embedding memerlukan indeks numerik, maka userId dan movieId asli diubah menjadi indeks untuk input model.

#### Compile dan Training Model

```
model = RecommenderNet(num_users, num_movies, 50)

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)

earlystopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=8,
    epochs=100,
    callbacks=[earlystopping, checkpoint],
    validation_data=(x_val, y_val)
)
```

Model dilatih menggunakan loss binary crossentropy dengan optimizer Adam. RMSE digunakan sebagai metrik evaluasi. Early stopping mencegah overfitting dengan menghentikan training saat validasi tidak membaik.

#### Visualisasi

![Hasil Metrik](img/cb_vis.jpg)

Visualisasi ini memperlihatkan bahwa model collaborative filtering yang dikembangkan berhasil belajar dengan baik pada data training dan tetap mempertahankan performa yang konsisten pada data testing. Dengan demikian, model ini dapat diandalkan untuk memprediksi rating film pada pengguna baru secara efektif.


#### Inferensi
```
def get_collaborative_recommendations(user_id, top_n=5):
    # Cek apakah user_id ada di mapping
    if user_id not in user_to_index:
        print(f"UserId {user_id} tidak ditemukan.")
        return None
    
    user_idx = user_to_index[user_id]
    
    # Semua movie index
    all_movie_indices = list(range(num_movies))
    
    # Movie yang sudah dirating user
    rated_movie_ids = df_rating[df_rating['userId'] == user_id]['movieId'].unique()
    rated_movie_indices = [movie_to_index[movie_id] for movie_id in rated_movie_ids if movie_id in movie_to_index]
    
    # Movie yang belum dirating user
    unrated_movie_indices = [i for i in all_movie_indices if i not in rated_movie_indices]
    
    # Prediksi rating untuk film belum dirating user
    predictions = []
    batch_size = 1000  # untuk efisiensi, prediksi batch
    
    for start in range(0, len(unrated_movie_indices), batch_size):
        end = min(start + batch_size, len(unrated_movie_indices))
        batch_movie_indices = unrated_movie_indices[start:end]
        batch_user_indices = [user_idx] * len(batch_movie_indices)
        
        inputs = np.array([batch_user_indices, batch_movie_indices]).T
        preds = model.predict(inputs).flatten()
        
        for movie_idx, pred_rating in zip(batch_movie_indices, preds):
            predictions.append((movie_idx, pred_rating))
    
    # Urutkan berdasarkan prediksi rating tertinggi
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Ambil top_n rekomendasi
    top_predictions = predictions[:top_n]
    
    # Ambil movieId dari index
    recommended_movie_ids = [index_to_movie[movie_idx] for movie_idx, _ in top_predictions]
    
    # Ambil judul film dari dataframe movies
    recommended_movies = movies[movies['movieId'].isin(recommended_movie_ids)][['movieId', 'title', 'year']].drop_duplicates()
    
    return recommended_movies

def recommend_collaborative():
    user_id_input = input("Masukkan userId untuk mendapatkan rekomendasi: ").strip()
    
    # Coba konversi input ke int, jika gagal langsung return error
    try:
        user_id = int(user_id_input)
    except ValueError:
        print("Input userId harus berupa angka.")
        return
    
    # Cek apakah user_id ada di data
    if user_id not in user_to_index:
        print(f"UserId {user_id} tidak ditemukan dalam data.")
        return
    
    recommendations = get_collaborative_recommendations(user_id, top_n=5)
    
    if recommendations is not None and not recommendations.empty:
        print(f"Rekomendasi film untuk userId {user_id}:")
        for _, row in recommendations.iterrows():
            print(f"- {row['title']} {row['year']} (movieId: {row['movieId']})")
    else:
        print(f"Tidak ada rekomendasi yang ditemukan untuk userId {user_id}.")
```

![CB Inferensi](img/cb_inferensi.jpg)

