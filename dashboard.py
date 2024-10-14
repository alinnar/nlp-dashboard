import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import numpy as np

# Bagian untuk mempercantik layout
st.set_page_config(page_title="Dashboard Analisis Teks", layout="wide", page_icon="📊")

# Fungsi untuk menambahkan custom CSS
# Fungsi untuk menambahkan custom CSS
def set_custom_style():
    st.markdown(
        """
        <style>
        /* Gaya untuk halaman utama */
        .stApp {
            background-color: #87CEFA;  /* Warna biru langit */
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;  /* Font yang modern dan mudah dibaca */
        }
        /* Gaya untuk judul */
        h1 {
            color: #1E90FF;  /* Warna biru cerah untuk judul */
            text-align: center;
        }
        /* Gaya untuk tabel */
        .css-1d391kg {
            background-color: #F0F8FF;  /* Background tabel */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Fungsi untuk menampilkan data crawling
def show_data_page(data):
    st.title("Data Hasil Crawling")
    st.write("Berikut adalah data yang diperoleh dari hasil crawling:")
    st.dataframe(data)

# Fungsi untuk menampilkan Wordcloud
def show_wordcloud_page(data):
    st.title("☁️ Wordcloud")
    st.write("Visualisasi Wordcloud dari hasil crawling:")
    
    if 'stemming' in data.columns:
        data_text = ' '.join(data['stemming'].astype(str).tolist())

        wc = WordCloud(background_color='white', max_words=150, width=1000, height=300).generate(data_text)

        plt.figure(figsize=(30, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
    else:
        st.warning("Kolom 'stemming' tidak ditemukan dalam data.")

# Fungsi untuk menampilkan Top Words
def show_top_words_page(data):
    st.title("📈 Top Words")
    st.write("Berikut adalah 12 kata yang paling sering muncul:")
    
    if 'stemming' in data.columns:
        text = ' '.join(data['stemming'].astype(str))
        words = text.split()

        word_counts = Counter(words)
        top_words = word_counts.most_common(12)

        words, counts = zip(*top_words)
        colors = plt.cm.Paired(np.arange(len(words)))

        plt.figure(figsize=(20, 6))
        bars = plt.bar(words, counts, color=colors)

        plt.xlabel('Kata')
        plt.ylabel('Frekuensi')
        plt.title('Kata yang sering muncul')
        plt.xticks(rotation=45)

        for bar, num in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width() / 2 - 0.1, num + 1, str(num), fontsize=12, color='black', ha='center')

        st.pyplot(plt)
    else:
        st.warning("Kolom 'stemming' tidak ditemukan dalam data.")

# Baca file CSV (sesuaikan dengan path file CSV Anda)
file_path = "NLP_dashboard/hasil_processing.csv"  # Ganti dengan lokasi file CSV Anda
data = pd.read_csv(file_path)

# Sidebar untuk navigasi halaman
page = st.sidebar.selectbox("Pilih Halaman", ["Data Hasil Crawling", "Wordcloud", "Top Words"])

# Tampilkan halaman sesuai pilihan
if page == "Data Hasil Crawling":
    show_data_page(data)
elif page == "Wordcloud":
    show_wordcloud_page(data)
elif page == "Top Words":
    show_top_words_page(data)