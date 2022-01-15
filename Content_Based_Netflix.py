#############################
# Content Based Recommendation by using Actor/Actress Names on Netflix Movies Data
#############################

#Data Set: Netflix Movies and TV Shows from Shivam Bansal on Kaggle. Source link: https://www.kaggle.com/shivamb/netflix-shows/code
#Purposes: I aim to find closest movies or tv series based on casts similarity. I use TF-IDF and Cosine Similarity Matris for finding top 10 similar movies.


#First of all, we must import libraries.

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


#And then, importing the datas from .csv file.

df = pd.read_csv("Personal Working/Netflix_Movies_Content_Based_Filtering_By_Cast/netflix_titles.csv")
df.head(50)
df.shape

#Are there NaN values in Cast columns? If there are, filling the space " " instead NaN
df["cast"].isna().sum()
df["cast"] = df["cast"].fillna(" ")
df["cast"].head(50)

#Setting up Tf-Idf matrix

tfidf = TfidfVectorizer(stop_words="english")
tf_idf_matrix = tfidf.fit_transform(df["cast"])
type(tf_idf_matrix) #scipy.sparse.csr.csr_matrix
df["cast"].shape #(8807,)
tf_idf_matrix.shape #(8807, 31799)

#Creating Cosine Similarity Matrix

cosine_sim = cosine_similarity(tf_idf_matrix)
cosine_sim.shape #(8807, 8807) We are setting up axa matrix for similarity.
cosine_sim[1]

###Making recommendation based on similarities

#Picking film names
indices = pd.Series(df.index, index=df["title"])

#Deleting duplicate values in indices
indices = indices[~indices.index.duplicated(keep="last")]

#Choosing a film in the df for finding its similar.
df[df["title"].str.contains("Vizon",na=False)] #Checking Inception is on the list or not.
movie_index = indices["Vizontele"]

#Converting pandas dataframe similarities between Inception and other films based on casts.
similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["Score"])

#Taking most similar films' index except itself (Vizontele)
movie_indices = similarity_scores.sort_values("Score", ascending=False)[1:11].index

#Showing the 10 movies that the casts are most similar
df[["title","cast"]].iloc[movie_indices]

#                                            title                                               cast
# 8682                               Vizontele Tuuba  Yılmaz Erdoğan, Tarık Akan, Altan Erkekli, Cez...
# 2621                                Organize Isler  Yılmaz Erdoğan, Tolga Çevik, Demet Akbağ, Alta...
# 8552                             The Water Diviner  Russell Crowe, Olga Kurylenko, Yılmaz Erdoğan,...
# 906   Have You Ever Seen Fireflies? - Theatre Play  Demet Akbağ, Zerrin Sümer, Sinan Bengier, Sali...
# 2605                                  Ekşi Elmalar  Yılmaz Erdoğan, Farah Zeynep Abdullah, Songül ...
# 6725                                 Eyyvah Eyyvah  Demet Akbağ, Ata Demirer, Özge Borak, Bican Gü...
# 6070                         Aşk Tesadüfleri Sever  Mehmet Günsür, Belçim Bilgin, Ayda Aksel, Alta...
# 2620                                  Neseli Hayat  Yılmaz Erdoğan, Ersin Korkut, Büşra Pekin, Erd...
# 2309                         Ali Baba ve 7 Cüceler  Cem Yılmaz, Irina Ivkina, Çetin Altay, Zafer A...
# 981                                        G.O.R.A  Cem Yılmaz, Rasim Öztekin, Özkan Uğur, İdil Fı...



