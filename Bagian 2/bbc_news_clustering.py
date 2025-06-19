import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, homogeneity_score, completeness_score
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import re
import joblib # Untuk menyimpan dan memuat model

# --- Download stop words NLTK (hanya perlu sekali dijalankan) ---
# Perbaikan: Menggunakan LookupError karena resource tidak ditemukan
# dan menambahkan pesan yang lebih jelas saat mengunduh.
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("NLTK 'stopwords' belum diunduh. Mengunduh sekarang...")
    nltk.download('stopwords')
    print("NLTK 'stopwords' berhasil diunduh.")

# --- Fungsi Pra-pemrosesan Teks ---
# Dapatkan daftar stop words dalam bahasa Inggris setelah memastikan mereka terunduh
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower() # Ubah ke huruf kecil
    text = re.sub(r'\d+', '', text) # Hapus angka
    text = re.sub(r'[^\w\s]', '', text) # Hapus tanda baca
    words = text.split() # Tokenisasi
    words = [word for word in words if word not in stop_words] # Hapus stop words
    return " ".join(words) # Gabungkan kembali menjadi string

# --- Fungsi untuk Melatih Model dan Menyimpan ---
def train_and_save_model(csv_file_path):
    print("\n--- Fase Pelatihan Model (KMeans Clustering) ---")
    print(f"Memuat dataset '{csv_file_path}'...")
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: File '{csv_file_path}' tidak ditemukan. Pastikan file berada di direktori yang sama.")
        return None, None, None # Perbaikan: Sekarang mengembalikan 3 None

    documents = df['text']
    true_labels = df['category']

    documents_preprocessed = documents.apply(preprocess_text)

    print("Melakukan vektorisasi TF-IDF...")
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, min_df=5)
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents_preprocessed)

    k = len(true_labels.unique()) # Menggunakan jumlah kategori asli sebagai k
    print(f"Jumlah cluster yang akan digunakan (k): {k}")

    print("Melatih model k-Means...")
    kmeans_model = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans_model.fit(tfidf_matrix)

    # Simpan vectorizer dan model KMeans
    joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
    joblib.dump(kmeans_model, 'kmeans_model.pkl')
    print("Model TF-IDF Vectorizer dan k-Means berhasil dilatih dan disimpan.")

    # Tampilkan evaluasi
    cluster_labels = kmeans_model.labels_
    ari = adjusted_rand_score(true_labels, cluster_labels)
    homogeneity = homogeneity_score(true_labels, cluster_labels)
    completeness = completeness_score(true_labels, cluster_labels)
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"Homogeneity Score: {homogeneity:.4f}")
    print(f"Completeness Score: {completeness:.4f}")

    # Visualisasi Confusion Matrix
    cluster_category_counts = pd.crosstab(pd.Series(cluster_labels, name='Cluster Label'), true_labels)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cluster_category_counts, annot=True, fmt='d', cmap='Blues', linewidths=.5)
    plt.title('Hubungan Cluster vs. Kategori Asli')
    plt.xlabel('Kategori Asli')
    plt.ylabel('Cluster yang Ditemukan')
    plt.show()

    # Mengidentifikasi label cluster
    cluster_names = {}
    print("\nPetakan Cluster ID ke Kategori Asli (berdasarkan observasi Heatmap):")
    for cluster_id in range(k):
        # Cari kategori asli yang paling dominan di cluster ini
        dominant_category = cluster_category_counts.loc[cluster_id].idxmax()
        cluster_names[cluster_id] = dominant_category
        print(f"Cluster {cluster_id} kemungkinan besar adalah: {dominant_category}")
    
    joblib.dump(cluster_names, 'cluster_names.pkl')
    print("Pemetaan nama cluster disimpan.")

    return tfidf_vectorizer, kmeans_model, cluster_names

# --- Fungsi untuk Memprediksi Kategori Berita Baru ---
def predict_new_article(article_text, vectorizer, model, cluster_names_map):
    print("\n--- Memprediksi Kategori Berita Baru ---")
    if not vectorizer or not model:
        print("Error: Model belum dilatih atau dimuat.")
        return "Model tidak siap"

    # Pra-pemrosesan teks input
    preprocessed_text = preprocess_text(article_text)

    # Vektorisasi teks input menggunakan vectorizer yang SUDAH DILATIH
    # transform() BUKAN fit_transform()
    text_vector = vectorizer.transform([preprocessed_text])

    # Prediksi cluster
    predicted_cluster_id = model.predict(text_vector)[0]

    # Dapatkan nama kategori yang dipetakan
    predicted_category_name = cluster_names_map.get(predicted_cluster_id, "Tidak Dikenal")

    print(f"Berita baru dikategorikan ke Cluster ID: {predicted_cluster_id}")
    print(f"Kemungkinan kategori: {predicted_category_name}")
    return predicted_category_name

# --- Bagian Utama Program ---
if __name__ == "__main__":
    print("Memulai proyek Pengelompokan Berita BBC dengan k-Means.") # Pesan awal
    csv_path = 'bbc-text.csv'
    tfidf_vectorizer_loaded = None
    kmeans_model_loaded = None
    cluster_names_map_loaded = None

    # Coba muat model yang sudah disimpan
    try:
        tfidf_vectorizer_loaded = joblib.load('tfidf_vectorizer.pkl')
        kmeans_model_loaded = joblib.load('kmeans_model.pkl')
        cluster_names_map_loaded = joblib.load('cluster_names.pkl')
        print("\nModel TF-IDF Vectorizer, k-Means, dan pemetaan nama cluster berhasil dimuat.")
    except FileNotFoundError:
        print("\nModel belum ditemukan. Melatih model baru...")
        # Panggil fungsi pelatihan dan simpan model
        tfidf_vectorizer_loaded, kmeans_model_loaded, cluster_names_map_loaded = train_and_save_model(csv_path)

    # Periksa apakah model berhasil dimuat atau dilatih
    if tfidf_vectorizer_loaded is None or kmeans_model_loaded is None:
        print("Program tidak dapat melanjutkan karena model tidak tersedia. Pastikan 'bbc-text.csv' ada.")
        exit()

    while True:
        print("\n-------------------------------------")
        print("Pilih Opsi:")
        print("1. Masukkan teks berita baru untuk dikategorikan")
        print("2. Keluar")
        choice = input("Pilihan Anda: ")

        if choice == '1':
            print("\nSilakan tempel atau ketik teks berita Anda. Ketik 'END_NEWS' di baris baru untuk mengakhiri.")
            lines = []
            while True:
                line = input()
                if line.strip().upper() == 'END_NEWS':
                    break
                lines.append(line)
            new_article_text = "\n".join(lines)

            if not new_article_text.strip():
                print("Tidak ada teks berita yang dimasukkan.")
                continue

            predicted_cat = predict_new_article(new_article_text, tfidf_vectorizer_loaded, kmeans_model_loaded, cluster_names_map_loaded)
            print(f"Hasil Prediksi: Berita ini kemungkinan masuk kategori **{predicted_cat}**.")
            print("Catatan: Kategori ini didasarkan pada cluster terdekat, bukan label asli yang dilatih secara supervised.")

        elif choice == '2':
            print("Terima kasih. Program selesai.")
            break
        else:
            print("Pilihan tidak valid. Silakan coba lagi.")