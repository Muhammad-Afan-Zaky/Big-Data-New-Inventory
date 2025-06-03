import requests
from bs4 import BeautifulSoup
import pandas as pd
import time, re, warnings
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from datetime import datetime

warnings.filterwarnings('ignore')

class Scraper:
    def __init__(self):
        self.headers = {"User-Agent": "Mozilla/5.0"}
        self.articles = []

    def scrape_pubmed(self, query, max_articles=10):
        base_url = "https://pubmed.ncbi.nlm.nih.gov/"
        search_url = f"{base_url}?term={query.replace(' ', '+')}&size=200"
        r = requests.get(search_url, headers=self.headers)
        soup = BeautifulSoup(r.content, 'html.parser')
        links = soup.find_all('a', class_='docsum-title')[:max_articles]

        for link in links:
            try:
                article_url = base_url + link.get('href')
                article_page = requests.get(article_url, headers=self.headers)
                soup_article = BeautifulSoup(article_page.content, 'html.parser')

                title = soup_article.find('h1', class_='heading-title')
                abstract = soup_article.find('div', class_='abstract-content')
                authors = soup_article.find('div', class_='authors-list')
                journal = soup_article.find('button', class_='journal-actions-trigger')

                self.articles.append({
                    'title': title.text.strip() if title else "No title",
                    'abstract': abstract.text.strip() if abstract else "No abstract",
                    'authors': authors.text.strip() if authors else "No authors",
                    'journal': journal.text.strip() if journal else "Unknown journal",
                    'url': article_url,
                    'scraped_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'source': 'PubMed'
                })
                print(f"Scraped: {title.text.strip()[:60]}")
                time.sleep(1)
            except Exception as e:
                print(f"Error: {e}")
                continue

    def save(self, filename="physio_therapy_articles.csv"):
        df = pd.DataFrame(self.articles)
        df.to_csv(filename, index=False)
        return df


class Analyzer:
    def __init__(self, df):
        self.df = df

    def preprocess(self, text):
        if pd.isna(text): return ""
        text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
        return ' '.join(text.split())

    def sentiment(self):
        sentiments = []
        for text in self.df['abstract']:
            blob = TextBlob(str(text))
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            label = "positive" if polarity > 0.1 else "negative" if polarity < -0.1 else "neutral"
            sentiments.append({'polarity': polarity, 'subjectivity': subjectivity, 'sentiment_label': label})
        self.df = pd.concat([self.df, pd.DataFrame(sentiments)], axis=1)

    def tfidf_cluster(self, n_clusters=5):
        docs = (self.df['title'] + " " + self.df['abstract']).fillna("").apply(self.preprocess)
        tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
        X = tfidf.fit_transform(docs)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.df['cluster'] = kmeans.fit_predict(X)

        pca = PCA(n_components=2)
        result = pca.fit_transform(X.toarray())
        self.df['pca_1'], self.df['pca_2'] = result[:, 0], result[:, 1]

    def save(self, filename="processed_physio_articles.csv"):
        self.df.to_csv(filename, index=False)
        print(f"✅ File disimpan: {filename}")


if __name__ == '__main__':
    scraper = Scraper()
    topics = [
        "physical therapy genu valgum",
        "genu varum physiotherapy",
        "knock knee treatment",
        "rehabilitation knee deformity"
    ]
    for topic in topics:
        scraper.scrape_pubmed(topic, max_articles=10)
    df = scraper.save()

    analyzer = Analyzer(df)
    analyzer.sentiment()
    analyzer.tfidf_cluster()
    analyzer.save()
