#!/usr/bin/env python
# coding: utf-8

# #### Spotify and Youtube Analysis
# 
# ###### Bibliotecas: 
# 

# In[1]:


## paquetes necesarios

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.offline import init_notebook_mode, iplot, plot


# #### EDA, exploratory data analysis

# In[2]:


df_music = pd.read_csv('Spotify_Youtube.csv')
df_music


# In[3]:


df_music.columns


# In[4]:


delete_columns = ['Unnamed: 0',  'Url_spotify', 'Uri','Url_youtube', 'Title','Comments', 'Description', 'Licensed']
 
## delete_columns es la lista de columnas que se desean eliminar del DataFrame
## La opción axis=1 se especifica para indicar que se eliminarán columnas (si se quisieran eliminar filas, se utilizaría axis=0). 
## La opción inplace=True se utiliza para modificar el DataFrame original en lugar de crear una copia y dejar el original intacto.
                  
df_music.drop(delete_columns, axis = 1, inplace = True)                  


# In[5]:


df_music.head()


# In[6]:


df_music.info()


# In[7]:


## Eliminará todas las filas que contienen valores faltantes en el DataFrame 
df_music.dropna(axis = 0, inplace= True)


# In[8]:


## Eliminará todas las filas duplicadas según la columna especificada,es decir, eliminamos las canciones repetidas en el df
df_music.drop_duplicates(subset = 'Track', inplace = True)


# In[9]:


## obsvervamos estadisticas de las caracteristicas tecnicas de las canciones del df
df_music.describe()


# In[10]:


## vemos correlacion entre las siguientes columnas
corr = df_music[['Danceability', 'Energy', 'Views','Stream', 'Likes', 'Duration_ms']].corr()

## creamos un mapa de calor utilizando seaborn
## cmap='coolwarm' establece los colores del mapa y annot=True agrega los valores de correlación a cada celda del mapa de calor
sns.heatmap(corr, cmap='coolwarm', annot=True)

# mostramos el mapa de calor
plt.show()


# ###### La matriz muestra:
# - Danceability está ligeramente correlacionado positivamente con Energy (0.23), lo que indica que las canciones con un alto nivel de baile tienden a tener una alta energía.
# - Views y Likes están altamente correlacionados entre sí (0.89), lo que indica que las canciones con más vistas también tienden a tener más me gusta.
# - Danceability y Energy tienen una correlación baja a moderada con Views y Likes, lo que indica que no hay una relación fuerte entre estos atributos y la popularidad de la canción medida por las vistas o los me gusta.
# - La duración de la canción tiene una correlación negativa débil con Danceability (-0.12), lo que indica que las canciones con un alto nivel de baile tienden a ser más cortas. También tiene una correlación débil positiva con Views (0.04), lo que indica que las canciones más largas tienden a tener más vistas.
# ###### Es importante destacar que la correlación no implica causalidad, es decir, la relación entre dos variables puede ser causada por una tercera variable desconocida.

# ###### Distribución de las variables: Utilizar gráficos de histograma y gráficos de densidad para visualizar la distribución de las variables numéricas en el DataFrame  puede ayudar a identificar valores atípicos o patrones interesantes en los datos.

# In[11]:


## Visualizamos la distribucion de danceability para ver como se comporta
sns.histplot(data=df_music, x='Danceability', kde=True,color = 'red')
plt.title('Distribución de la bailabilidad de las canciones')
plt.xlabel('Bailabilidad')
plt.ylabel('Frecuencia')
plt.show()


# In[12]:


## Visualizamos la distribucion de energy para ver como se comporta
sns.histplot(data=df_music, x='Energy', kde=True)
plt.title('Distribución de la energía de las canciones')
plt.xlabel('Energía')
plt.ylabel('Frecuencia')
plt.show()


# ###### Podemos observar que un gran número de canciones buscan tener alta energía y bailabilidad. Teniendo en cuenta que estas variables correlacionan positivamente con likes y views, podríamos inferir que esto se realiza para que la canción tenga más alcance / sea más popular.

# In[21]:


# Creamos Data frames distintos para views en youtube y streams en spotify
df_views_songs = df_music.groupby('Track')['Views'].sum().sort_values(ascending=False)[:10]
df_streams_songs = df_music.groupby('Track')['Stream'].sum().sort_values(ascending=False)[:10]
df_views_songs
df_streams_songs

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))

# top 10 yt
ax1.set_title('Top 10 Canciones en YouTube')
df_views_songs.plot(kind='bar', ax=ax1, color = 'red')

# top 10 en spotify
ax2.set_title('Top 10 Canciones en Spotify')
df_streams_songs.plot(kind='bar', ax=ax2, color = 'lightgreen')


ax1.set_xlabel('Songs')
ax1.set_ylabel('Views')
ax2.set_xlabel('Songs')
ax2.set_ylabel('Streams')
fig.tight_layout()
plt.rcParams['font.family'] = 'Arial Unicode MS'
plt.rcParams["font.family"] = "Malgun Gothic"
plt.show()


# In[14]:


## Ahora veamos las caracteristicas tecnicas de las canciones top de Spotify

popular_songs_spt = df_music.sort_values('Views',ascending = False).head(10)
popular_songs_spt[['Track', 'Energy', 'Danceability', 'Acousticness']]


# In[15]:


## Ahora veamos las caracteristicas tecnicas de las canciones top de Youtube 

popular_songs_yt = df_music.sort_values('Stream',ascending = False).head(10)
popular_songs_yt[['Track', 'Energy', 'Danceability', 'Acousticness']]


# ###### En general, las canciones más populares en Spotify tienden a tener una energía bastante alta, lo que sugiere que son canciones que tienden a ser más animadas y emocionantes. La danzabilidad también es una característica común entre estas canciones, lo que sugiere que son canciones fáciles de bailar o moverse con ellas.
# 
# ###### La acústica de las canciones es bastante variada, pero la mayoría de ellas tienden a ser menos acústicas, lo que sugiere que la producción de estas canciones es más electrónica y/o sintética.
# 
# ######  Entre las canciones más populares, encontramos algunos éxitos como "Despacito" y "Shape of You" que fueron lanzados hace algunos años, lo que sugiere que la popularidad de estas canciones se ha mantenido durante mucho tiempo. Además, hay algunas canciones como "Gangnam Style" que fueron éxitos internacionales en su momento, pero que ahora no suenan tanto en las listas de reproducción.
# 
# ######  En general, es importante tener en cuenta que la popularidad de una canción puede deberse a una variedad de factores, incluyendo la promoción y la comercialización, así como las características musicales de la canción en sí.

# In[16]:


## Visualizamos la distribucion de la duración para ver como se comporta
## Primero es necesario traspasar los milisegundos en minutos

df_music['Duration_min'] = df_music['Duration_ms'] / 60000


plt.hist(df_music['Duration_min'], bins=20, color = 'moccasin')
plt.xlabel('Duración en minutos')
plt.xlim(0, 8)
plt.ylabel('Frecuencia')
plt.title('Distribución de la duración de las canciones')
plt.show()


# In[17]:


## Vemos el total de artistas 
total_artist = df_music['Artist'].value_counts()
total_artist


# ###### Tenemos un total de 2033 artistas, veamos los albums y artistas populares

# In[18]:


## Agrupamos los albums de los artistas, vistas en youtube y streams en spotify
album_grouped = df_music.groupby('Album')[['Views','Stream']].sum()

## Ordenamos los albums de forma descendente por cantidad de visitas y streams
album_sorted = album_grouped.sort_values(['Views', 'Stream'], ascending=False)

##  Mostramos los albums con mayor visita en Youtube y streams en Spotify
top_10_album = album_sorted.head(10)
top_10_album


# In[19]:


## Agrupamos los artistas, por visitas en youtube y streams en spotify
artist_grouped = df_music.groupby('Artist')[['Views','Stream']].sum()

## Ordenamos artistas de forma descendente por cantidad de visitas y streams
artist_sorted = artist_grouped.sort_values(['Views','Stream'], ascending = False)

## Mostramos los artistas con mayor visita en youtube y streams en spotify
top_10_artist = artist_sorted.head(10)
top_10_artist


# In[20]:


# Creamos Data frames distintos 
df_views_artist = df_music.groupby('Artist')['Views'].sum().sort_values(ascending=False)[:10]
df_streams_artist = df_music.groupby('Artist')['Stream'].sum().sort_values(ascending=False)[:10]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))

# top 10 yt
ax1.set_title('Top 10 Artistas en YouTube')
df_views_artist.plot(kind='bar', ax=ax1, color = 'red')

# top 10 en spotify
ax2.set_title('Top 10 Artistas en Spotify')
df_streams_artist.plot(kind='bar', ax=ax2, color = 'lightgreen')


ax1.set_xlabel('Artist')
ax1.set_ylabel('Views')
ax2.set_xlabel('Artist')
ax2.set_ylabel('Streams')
fig.tight_layout()
plt.show()

