import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import re
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from nltk.stem import PorterStemmer

# Bagian untuk mempercantik layout
st.set_page_config(page_title="Dashboard Analisis Teks", layout="wide", page_icon="📊")

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

    if 'cleaning' in data.columns:
        data_text = ' '.join(data['cleaning'].astype(str).tolist())

        wc = WordCloud(background_color='white', max_words=150, width=700, height=280).generate(data_text)

        plt.figure(figsize=(50, 50))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
    else:
        st.warning("Kolom 'cleaning' tidak ditemukan dalam data.")

# Fungsi untuk menampilkan Top Words
def show_top_words_page(data):
    st.title("📈 Top Words")
    st.write("Berikut adalah 12 kata yang paling sering muncul:")

    if 'cleaning' in data.columns:
        text = ' '.join(data['cleaning'].astype(str))
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
        st.warning("Kolom 'cleaning' tidak ditemukan dalam data.")

# Fungsi untuk prediksi sentimen
def predict_sentiment(text, model, vectorizer):
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)
    return prediction

# Fungsi untuk memuat halaman Sentiment Analysis
def show_sentiment_analysis_page():
    st.title("Sentiment Analysis NLP App")

    with st.form("nlpForm"):
        raw_text = st.text_area("Enter Text Here")
        submit_buttton = st.form_submit_button(label='Analyze')

    # Layout kolom
    col1, col2 = st.columns(2)
    if submit_buttton:
        # Baca data yang diunggah
        data = pd.read_csv('./dataset/predicted_sentiment_balanced.csv')  # Sesuaikan path file dengan file yang diunggah
        data['cleaned_tweet'] = data['cleaned_x']
        x = data['cleaned_tweet']
        y = data['predicted_sentiment']

        # Vectorizer dan model Naive Bayes
        vec = CountVectorizer()
        train = vec.fit_transform(x)

        model = MultinomialNB()
        model.fit(train, y)

        # Prediksi sentimen dari input pengguna
        y_pred = predict_sentiment(raw_text, model, vec)

        with col1:
            st.info("Results")
            st.write(y_pred)

            # Emoji hasil sentimen
            if y_pred == 'positive':
                st.markdown("Kata atau kalimat tersebut menunjukan Sentiment : Positive : 😍 ")
            elif y_pred == 'neutral':
                st.markdown("Kata atau kalimat tersebut menunjukan Sentiment : Netral : 🙂")
            else:
                st.markdown("Kata atau kalimat tersebut menunjukan Sentiment : Negative : 😡 ")

# Fungsi untuk menampilkan halaman lainnya
def show_data_page(data):
    st.write(data)

# Baca file CSV (sesuaikan dengan path file CSV Anda)
file_path = './dataset/hasil_processing.csv'
data = pd.read_csv(file_path)

# Sidebar untuk navigasi halaman
page = st.sidebar.selectbox("Pilih Halaman", ["Data Hasil Crawling", "Wordcloud", "Top Words", "Sentiment Analysis"])

# Tampilkan halaman sesuai pilihan
if page == "Data Hasil Crawling":
    show_data_page(data)
elif page == "Wordcloud":
    show_wordcloud_page(data)
elif page == "Top Words":
    show_top_words_page(data)
elif page == "Sentiment Analysis":
    show_sentiment_analysis_page()