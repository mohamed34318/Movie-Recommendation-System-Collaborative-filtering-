# Movie Recommendation System using Collaborative Filtering

```markdown

This project demonstrates a movie recommendation system using collaborative filtering. The dataset used includes detailed information about the top 250 English movies from IMDB. The system recommends movies based on their similarity in terms of directors, genres, actors, and plot descriptions.

## Project Steps

### 1. Importing Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
pd.set_option('display.max_columns', None)
nltk.download('punkt')
nltk.download('stopwords')
```

### 2. Loading and Inspecting the Data

```python
df = pd.read_csv('/content/data/dd/IMDB_Top250Engmovies2_OMDB_Detailed.csv', index_col=False)
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)
df.head()
df.info()
```

### 3. Data Preprocessing

#### 3.1 Cleaning the Plot Column

```python
df['clean_plot'] = df['Plot'].str.lower()
df['clean_plot'] = df['clean_plot'].apply(lambda x: re.sub('[^a-zA-Z]', ' ', x))
df['clean_plot'] = df['clean_plot'].apply(lambda x: re.sub('\s+', ' ', x))
df['clean_plot'] = df['clean_plot'].apply(lambda x: nltk.word_tokenize(x))
stopwords = nltk.corpus.stopwords.words('english')
df['clean_plot'] = df['clean_plot'].apply(lambda x: [w for w in x if w not in stopwords])
```

#### 3.2 Cleaning the Writer Column

```python
df['clean_Writer'] = df['Writer'].str.lower()
df['clean_Writer'] = df['clean_Writer'].astype(str).apply(lambda x: re.sub('[^a-zA-Z]', ' ', x))
df['clean_Writer'] = df['clean_Writer'].apply(lambda x: re.sub('\s+', ' ', x))
df['clean_Writer'] = df['clean_Writer'].apply(lambda x: nltk.word_tokenize(x))
df['clean_Writer'] = df['clean_Writer'].apply(lambda x: [w for w in x if w not in stopwords])
```

#### 3.3 Splitting and Cleaning Other Columns

```python
df['Director'] = df['Director'].apply(lambda x: x.split(','))
df['Genre'] = df['Genre'].apply(lambda x: x.split(','))
df['Actors'] = df['Actors'].apply(lambda x: x.split(','))
df['Language'] = df['Language'].apply(lambda x: x.split(','))

df['Genre'] = df['Genre'].apply(lambda x: [a.lower().replace(' ', '') for a in x])
df['Actors'] = df['Actors'].apply(lambda x: [a.lower().replace(' ', '') for a in x])
df['Director'] = df['Director'].apply(lambda x: [a.lower().replace(' ', '') for a in x])
df['Language'] = df['Language'].apply(lambda x: [a.lower().replace(' ', '') for a in x])
```

#### 3.4 Combining All Cleaned Columns

```python
l = []
columns = ['Director', 'Genre', 'Actors', 'clean_plot', 'clean_Writer']
for i in range(len(df)):
    words = ''
    for col in columns:
        words += ' '.join(df[col][i]) + ' '
    l.append(words)
df['clean_input'] = l
ddf = df[['Title', 'clean_input']]
```

### 4. Feature Extraction

```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
features = tfidf.fit_transform(ddf['clean_input'])
```

### 5. Calculating Cosine Similarity

```python
from sklearn.metrics.pairwise import cosine_similarity
cos_sim = cosine_similarity(features, features)
```

### 6. Building the Recommendation System

```python
index = pd.Series(ddf['Title'])

def recommend_movie(title):
    movies = []
    idx = index[index == title].index[0]
    score = pd.Series(cos_sim[idx]).sort_values(ascending=False)
    top10 = list(score.iloc[1:11].index)
    for i in top10:
        movies.append(ddf['Title'][i])
    return movies
```

### 7. Example Usage

```python
recommend_movie('The Dark Knight')
```

The output is a list of 10 movies similar to "The Dark Knight".

## Conclusion

This project demonstrates the process of building a movie recommendation system using collaborative filtering. The steps include data cleaning, feature extraction, similarity calculation, and building the recommendation function.
```

This markdown will render nicely on GitHub, providing clear sections and properly formatted code blocks.
