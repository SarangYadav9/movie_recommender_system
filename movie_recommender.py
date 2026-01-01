import pandas as pd
import numpy as np

movies = pd.read_csv("/content/tmdb_5000_movies.csv.zip")
credits = pd.read_csv("/content/tmdb_5000_credits.csv.zip")

movies = movies.merge(credits, on='title')

movies = movies[['id','genres','keywords','title','overview','cast','crew']]

movies.dropna(inplace=True)

import ast
def convert(obj):
  l = []
  for i in ast.literal_eval(obj):
    l.append(i['name'])
  return l

movies['genres']=movies['genres'].apply(convert)
movies['keywords']=movies['keywords'].apply(convert)

def convert3(obj):
  l=[]
  count = 0
  for i in ast.literal_eval(obj):
    if count !=3:
      l.append(i['name'])
      count+=1
    else:
      break
  return l

movies['cast']=movies['cast'].apply(convert3)

def fetc_director(obj):
  l=[]
  for i in ast.literal_eval(obj):
    if i['job']=="Director":
      l.append(i['name'])
  return l

movies['crew']=movies['crew'].apply(fetc_director)

movies['overview']=movies['overview'].apply(lambda x:x.split())

movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ",'') for i in x ])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ",'') for i in x ])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ",'') for i in x ])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ",'') for i in x ])

movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']
movies=movies[['id','title','tags']]
movies['tags']=movies['tags'].apply(lambda x:" ".join(x))
movies['tags']=movies['tags'].apply(lambda x:x.lower())

import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def stem(text):
  y=[]
  for i in text.split():
    y.append(ps.stem(i))
  return " ".join(y)

movies['tags']=movies['tags'].apply(stem)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

def recommend(movie):
  movie_index = movies[movies['title']==movie].index[0]
  distances = similarity[movie_index]
  movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
  for i in movies_list:
    print(movies.iloc[i[0]].title)
