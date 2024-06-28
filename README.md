<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Movie Recommendation System</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; }
        h1, h2, h3 { color: #333; }
        code { background-color: #f4f4f4; padding: 2px 4px; border-radius: 4px; }
        pre { background-color: #f4f4f4; padding: 10px; border-radius: 4px; overflow-x: auto; }
        .container { max-width: 800px; margin: auto; padding: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Movie Recommendation System using Collaborative Filtering</h1>
        <p>This project demonstrates a movie recommendation system using collaborative filtering. The dataset used includes detailed information about the top 250 English movies from IMDB. The system recommends movies based on their similarity in terms of directors, genres, actors, and plot descriptions.</p>
        
        <h2>Project Steps</h2>

        <h3>1. Importing Libraries</h3>
        <pre><code>import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
pd.set_option('display.max_columns', None)
nltk.download('punkt')
nltk.download('stopwords')</code></pre>

        <h3>2. Loading and Inspecting the Data</h3>
        <pre><code>df = pd.read_csv('/content/data/dd/IMDB_Top250Engmovies2_OMDB_Detailed.csv', index_col=False)
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)
df.head()
df.info()</code></pre>

        <h3>3. Data Preprocessing</h3>

        <h4>3.1 Cleaning the Plot Column</h4>
        <pre><code>df['clean_plot'] = df['Plot'].str.lower()
df['clean_plot'] = df['clean_plot'].apply(lambda x: re.sub('[^a-zA-Z]', ' ', x))
df['clean_plot'] = df['clean_plot'].apply(lambda x: re.sub('\s+', ' ', x))
df['clean_plot'] = df['clean_plot'].apply(lambda x: nltk.word_tokenize(x))
stopwords = nltk.corpus.stopwords.words('english')
df['clean_plot'] = df['clean_plot'].apply(lambda x: [w for w in x if w not in stopwords])</code></pre>

        <h4>3.2 Cleaning the Writer Column</h4>
        <pre><code>df['clean_Writer'] = df['Writer'].str.lower()
df['clean_Writer'] = df['clean_Writer'].astype(str).apply(lambda x: re.sub('[^a-zA-Z]', ' ', x))
df['clean_Writer'] = df['clean_Writer'].apply(lambda x: re.sub('\s+', ' ', x))
df['clean_Writer'] = df['clean_Writer'].apply(lambda x: nltk.word_tokenize(x))
df['clean_Writer'] = df['clean_Writer'].apply(lambda x: [w for w in x if w not in stopwords])</code></pre>

        <h4>3.3 Splitting and Cleaning Other Columns</h4>
        <pre><code>df['Director'] = df['Director'].apply(lambda x: x.split(','))
df['Genre'] = df['Genre'].apply(lambda x: x.split(','))
df['Actors'] = df['Actors'].apply(lambda x: x.split(','))
df['Language'] = df['Language'].apply(lambda x: x.split(','))

df['Genre'] = df['Genre'].apply(lambda x: [a.lower().replace(' ', '') for a in x])
df['Actors'] = df['Actors'].apply(lambda x: [a.lower().replace(' ', '') for a in x])
df['Director'] = df['Director'].apply(lambda x: [a.lower().replace(' ', '') for a in x])
df['Language'] = df['Language'].apply(lambda x: [a.lower().replace(' ', '') for a in x])</code></pre>

        <h4>3.4 Combining All Cleaned Columns</h4>
        <pre><code>l = []
columns = ['Director', 'Genre', 'Actors', 'clean_plot', 'clean_Writer']
for i in range(len(df)):
    words = ''
    for col in columns:
        words += ' '.join(df[col][i]) + ' '
    l.append(words)
df['clean_input'] = l
ddf = df[['Title', 'clean_input']]</code></pre>

        <h3>4. Feature Extraction</h3>
        <pre><code>from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
features = tfidf.fit_transform(ddf['clean_input'])</code></pre>

        <h3>5. Calculating Cosine Similarity</h3>
        <pre><code>from sklearn.metrics.pairwise import cosine_similarity
cos_sim = cosine_similarity(features, features)</code></pre>

        <h3>6. Building the Recommendation System</h3>
        <pre><code>index = pd.Series(ddf['Title'])

def recommend_movie(title):
    movies = []
    idx = index[index == title].index[0]
    score = pd.Series(cos_sim[idx]).sort_values(ascending=False)
    top10 = list(score.iloc[1:11].index)
    for i in top10:
        movies.append(ddf['Title'][i])
    return movies</code></pre>

        <h3>7. Example Usage</h3>
        <pre><code>recommend_movie('The Dark Knight')</code></pre>

        <p>The output is a list of 10 movies similar to "The Dark Knight".</p>

        <h2>Conclusion</h2>
        <p>This project demonstrates the process of building a movie recommendation system using collaborative filtering. The steps include data cleaning, feature extraction, similarity calculation, and building the recommendation function.</p>
    </div>
</body>
</html>
