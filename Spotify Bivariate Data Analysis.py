#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
print('libraries imported')


# In[2]:


# Import the data set
import types

from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share the notebook.
client_aa0682dac2ea46708a76c09ecac60e99 = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='OQ7E-F2EpeAr2rMGAtJtZWmzVBrRQ0ePT4o3uNqhMC6g',
    ibm_auth_endpoint="https://iam.cloud.ibm.com/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3-api.us-geo.objectstorage.service.networklayer.com')

body = client_aa0682dac2ea46708a76c09ecac60e99.get_object(Bucket='spotify-donotdelete-pr-riayigqj1y9rsh',Key='Spotify Data.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

df = pd.read_csv(body)


# In[3]:


# Convert release_date feature from object to date time
df['release_date'] = pd.to_datetime(df['release_date'])

# Create a new column based solely on the year of the new date time
df['year'] = df['release_date'].dt.year

# Drop old release_date column
df = df.drop(columns = 'release_date', axis = 1, inplace = False)

# Drop the id column
df = df.drop(columns = 'id', axis = 1, inplace = False)

# Convert duration_ms to duration_s
df['duration'] = round(df['duration_ms']/(1000),2)

# Drop old duration_ms column
df = df.drop(columns = 'duration_ms', axis = 1, inplace = False)

# View the first five rows of the data set after all the edits
df.head()


# ## Bivariate Analysis
# 
# In this section we will discuss the relationship between different features. The main focus is how other variables will impact popularity. This will pave a way for the next notebook in this project that will use these features and predict the popularity level of a song. 

# In[4]:


# Graphically examine the relationship between Tempo and popularity
ax = sns.scatterplot(data = df, x = 'tempo', y = 'popularity')
ax.set(xlabel = 'Tempo', ylabel = 'Popularity Level', title = 'Popularity Level vs. Tempo')
plt.show()


# In[5]:


# Graphically examine the relationship between popularity and explicit
ax = sns.boxplot(data = df, x = 'explicit', y = 'popularity')
ax.set(xlabel = 'Explicit Content', ylabel = 'Popularity Level', title = 'Popularity vs. Explicit Content')
plt.show()


# In[6]:


# Graphically examine the relationship between mode and popularity
ax = sns.boxplot(data = df, x = 'mode', y = 'popularity')
ax.set(xlabel = 'Major or Minor Key', ylabel = 'popularity', title = 'Popularity vs. Major/Minor Key Composition')
plt.show()


# In[7]:


# Graphically examine the relationship between danceability and popularity
ax = sns.scatterplot(data = df, x = 'danceability', y = 'popularity')
ax.set(xlabel = 'Danceability', ylabel = 'Popularity Level', title = 'Popularity Level vs. Danceability')
plt.show()


# In[8]:


# Graphically examine the relationship between duration(s) and popularity
ax = sns.scatterplot(data = df, x = 'duration', y = 'popularity')
ax.set(xlabel = 'Duration(s)', ylabel = 'Popularity Level', title = 'Popularity Level vs. Duration(s)')
plt.show()


# In[9]:


# Graphically examine the relationship between energy and popularity
ax = sns.scatterplot(data = df, x = 'energy', y = 'popularity')
ax.set(xlabel = 'Energy', ylabel = 'Popularity Level', title = 'Popularity Level vs. Energy')
plt.show()


# In[10]:


# Graphically examine the relationship between loudness and popularity
ax = sns.scatterplot(data = df, x = 'loudness', y = 'popularity')
ax.set(xlabel = 'Loudness Level(db)', ylabel = 'Popularity Level', title = 'Popularity Level vs. Loudness Level(db)')
plt.show()


# In[11]:


# Graphically examine the relationship between acousticness and popularity
ax = sns.scatterplot(data = df, x = 'acousticness', y = 'popularity')
ax.set(xlabel = 'Acousticness', ylabel = 'Popularity Level', title = 'Popularity Level vs. Acousticness ')
plt.show()


# In[12]:


# Graphically examine the relationship between liveness and popularity
ax = sns.scatterplot(data = df, x = 'liveness', y = 'popularity')
ax.set(xlabel = 'Liveness', ylabel = 'Popularity Level', title = 'Popularity Level vs. Liveness')
plt.show()


# In[13]:


# Graphically examine the relationship between speechiness and popularity
ax = sns.scatterplot(data = df, x = 'speechiness', y = 'popularity')
ax.set(xlabel = 'Speechiness', ylabel = 'Popularity Level', title = 'Popularity Level vs. Speechiness')
plt.show()


# In[14]:


# Graphically examine the relationship between valence and popularity
ax = sns.scatterplot(data = df, x = 'valence', y = 'popularity')
ax.set(xlabel = 'Valence', ylabel = 'Popularity Level', title = 'Popularity Level vs. Valence')
plt.show()


# In[15]:


# Graphically examine the relationship between instrumentalness and popularity
ax = sns.scatterplot(data = df, x = 'instrumentalness', y = 'popularity')
ax.set(xlabel = 'Instrumentalness', ylabel = 'Popularity Level', title = 'Popularity Level vs. Instrumentalness')
plt.show()


# In[16]:


# Graphically examine the relationship between tempo and danceability
ax = sns.scatterplot(data = df, x = 'tempo', y = 'danceability')
ax.set(xlabel = 'Tempo', ylabel = 'Danceability Level', title = 'Danceability Level vs. Tempo')
plt.show()


# In[17]:


# Graphically examine the relationship between tempo and loudness
ax = sns.scatterplot(data = df, x = 'tempo', y = 'loudness')
ax.set(xlabel = 'Tempo', ylabel = 'Loudness Level(db)', title = 'Tempo vs. Loudness Level(db)')
plt.show()


# In[18]:


# Graphically examine the relationship between explicit and loudness
ax = sns.scatterplot(data = df, x = 'explicit', y = 'loudness')
ax.set(xlabel = 'Explicit Content', ylabel = 'Loudness Level(db)', title = 'Loudness Level(db) vs. Explicit Content')
plt.show()


# In[19]:


# Graphically examine the popularity of songs in each octave key C, C#, ...
df_keys = df.copy()
keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
for index in range(len(df)):
    for key_index in range(len(keys)):
        if df['key'][index] == key_index:
            df_keys['key'][index] = keys[key_index]
            
fig = px.box(df_keys, x = df_keys.key, y = df_keys.popularity, title = 'Popularity of Songs in Each Key', color_discrete_sequence = ['darkorange'],category_orders = {
    'key': ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']})
fig.show()


# In[20]:


# Display heatmap to show correlation between variables
plt.subplots(figsize = (20,20))
sns.heatmap(df.corr(), annot = True, square = True)
plt.show()


# ## Conclusions from Bivariate Data Analysis
# When it comes down to popularity, we can see from the above heatmap that most of our features have some impact on it based on their correlation values. Some insights on the features and popularity are as follows:
# - **Tempo vs. Popularity** : Songs that are between 60 and 200 bpm are more popular
# - **Explicit vs. Popularity** : Explicit songs have a tendency to be more popular than songs with no explicit content
# - **Mode vs. Popularity** : There doesn't seem to be much difference between songs composed in a major key or a minor key when compared to popularity
# - **Danceability vs. Popularity** : More popular songs have a tendency to have a higher danceability
# - **Duration(s) vs. Popularity** : The more popular songs seem to have a duration of 120 to 300 seconds
# - **Energy vs. Popularity** : As the energy level ranges from 0.5 to 1 the songs tend to increase in popularity
# - **Loudness(db) vs. Popularity** : The more popular the song the more likely it is to fall within the decible range of -20 to 0db
# - **Acousticness vs. Popularity** : Songs that have a higher popularity level have a tendency to fall within the range of 0 to 0.4 on the acousticness scale
# - **Liveness vs. Popularity** : The more popular songs have values within the range 0 and 0.4
# - **Speechiness vs. Popularity** : Songs that rank higher regarding their popularity level have a lower speechiness value
# - **Valence vs. Popularity** : Regarding the distribution of popularity and valence it seems to be approximately uniform throughout
# - **Instrumentalness vs. Popularity** : The more popular a song is the less its instrumentalness value
# - **Key vs. Popularity** : It seems to be that the more popular keys to compose a song in are C#, G, and A#
# - **Tempo vs. Danceability** : According to the scatterplot it seems that songs with a tempo of 125bpm have a higher level of danceability
# - **Tempo vs. Loudness(db)** : There doesn't seem to be a correlation between tempo and loudness(db). 
# - **Explicit vs. Loudness(db)** : Although there are less songs with explicit content it, the lower end for explicit is towards the midrange of loudness(db) for nonexplicit songs

# In[21]:


# Find the most popular songs on Spotify
popularity = df.copy().sort_values(by = ['popularity'], ascending = False)[['popularity','name','artists']]
popularity.head(10)


# The top ten most popular songs are: *Drivers License*, *Mood*, *Positions*, *DAKITI*, *BICHOTA*, *34+35*, *Whoopty*, *Without You*, *Therefore I Am*, and *La Noche De Anoche*

# In[22]:


# Find the most popular songs on Spotify from 2018 to 2020
df_pop = df[df.popularity > 90]
df_pop.head(10)


# The most popular song for 2018 was *Snowman* by Sia. The most popular song for 2019 was *Watermelon Sugar* by Harry Styles. Finally the most popular song in 2020 was a tie between *Mood* by 24kGoldn and *Positions* by Ariana Grande

# In[23]:


# Who are the most popular artists on Spotify?
plt.figure(figsize=(9, 4))
x = df.groupby("artists")["popularity"].sum().sort_values(ascending=False).head(20)
ax = sns.barplot(x.index, x)
ax.set(xlabel = 'Artist', ylabel = 'Popularity', title = 'Most Popular Artists')
plt.xticks(rotation = 90)


# According to the graph The Beatles were the most popular artist on Spotify with the highest total popularity rating, followed by Frank Sinatra, Elivs Presley, and Fleetwood Mac

# ## Conclusion
# Overall there is always more data analysis to perform. One thing we could do is merge this data set with another dataset that contains the genre of each song. We can then perform more analysis to see which genre is the most popular, how many songs are categorized by each genre, and of course visualize these using any of the graphical visualization libraries in python. The analysis of this data set provided a lot of insights into Spotify's song collection. The next notebook in this series will focus in on building a recommender system for the aritsts on Spotify. Utilizing that we can investigate how well Spotify is able to recommend songs and artists based on a wide variety of features. 

# In[ ]:




