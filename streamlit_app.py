import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dashboard Terapi Fisik", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("processed_physio_articles.csv")

st.title("🦵 Dashboard Analisis Artikel Terapi Fisik")
st.markdown("Menampilkan hasil analisis sentimen dan klasterisasi dari artikel PubMed.")

df = load_data()

# Filter Sidebar
st.sidebar.header("Filter Data")
sources = st.sidebar.multiselect("Sumber:", df['source'].unique(), default=df['source'].unique())
sentiments = st.sidebar.multiselect("Sentimen:", df['sentiment_label'].unique(), default=df['sentiment_label'].unique())

filtered = df[df['source'].isin(sources) & df['sentiment_label'].isin(sentiments)]

# Ringkasan
st.metric("Total Artikel", len(filtered))
st.metric("Rata-rata Polarity", round(filtered['polarity'].mean(), 3))
st.metric("Rata-rata Subjectivity", round(filtered['subjectivity'].mean(), 3))

# Pie Sentimen
st.subheader("📊 Distribusi Sentimen")
fig = px.pie(filtered, names='sentiment_label', title="Sentimen Artikel")
st.plotly_chart(fig, use_container_width=True)

# Cluster PCA
st.subheader("🧠 Visualisasi Klaster (PCA)")
fig2 = px.scatter(filtered, x='pca_1', y='pca_2', color=filtered['cluster'].astype(str), hover_data=['title'])
st.plotly_chart(fig2, use_container_width=True)

# Word Cloud
st.subheader("☁️ Word Cloud Abstrak")
text = " ".join(filtered['abstract'].dropna().astype(str))
wc = WordCloud(width=800, height=400, background_color='white').generate(text)
fig_wc, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wc, interpolation="bilinear")
ax.axis("off")
st.pyplot(fig_wc)

# Tabel
st.subheader("📄 Tabel Data Artikel")
st.dataframe(filtered[['title', 'authors', 'source', 'sentiment_label', 'polarity']])

# Download
st.download_button("⬇️ Download CSV", data=filtered.to_csv(index=False), file_name="filtered_articles.csv")
