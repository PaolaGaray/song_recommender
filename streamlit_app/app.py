# libraries
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pickle
import pandas as pd


api_key_id = st.secrets["api"]["SPOTIFY_CLIENT_ID"]
api_key_secret = st.secrets["api"]["SPOTIFY_CLIENT_SECRET"]

# import sys
# sys.path.append('..')
# import config


# Load the KMeans model
with open('../models/Gnod_Kmeans_4.pkl', 'rb') as f:
    kmeans = pickle.load(f)



# Load your playlist dataframe (where the clusters and songs are stored)
playlist_df = pd.read_csv("../data/clean/big_playlist_df.csv")  # Adjust the path


# Set up Spotipy with your credentials
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=api_key_id,
                                                          client_secret=api_key_secret))


def fetch_song_data(song_input, artist_input):
    # Add artist in search
    result = sp.search(q=song_input, limit=1, market="DE")
    song_id = result['tracks']['items'][0]['id']
    return song_id


def get_song_features(song_id):
    # Get the song features
    feature_list = ["danceability", "energy", "loudness", "speechiness", "acousticness", 
                    "instrumentalness", "liveness", "valence", "tempo", "duration_ms"]
    song_feats = sp.audio_features([song_id])  # Fetch song features as a list
    
    # Convert to DataFrame
    feats_df = pd.DataFrame(song_feats)
    return feats_df[feature_list]


def clustify(X):
    cluster = kmeans.predict(X)
    cluster = cluster[0]  # Take the first prediction from array
    return cluster


def recommender(cluster_num):
    cluster_songs = playlist_df.loc[playlist_df['cluster'] == cluster_num]  # Find songs in the same cluster
    random_sample = cluster_songs.sample(n=1, random_state=42)  # Select one random song
    song_name = random_sample['names'].values[0]  # Get the song name
    return song_name


# Streamlit app code starts here
st.title('Song Recommendation Engine')


# Take user input for song and artist
song_input = st.text_input("Enter a song name:")
artist_input = st.text_input("Enter the artist's name:")


# Button to trigger recommendation
if st.button('Recommend a Song'):
    if song_input and artist_input:
        song_id = fetch_song_data(song_input, artist_input)
        
        if song_id:
            # Get song features and predict the cluster
            X = get_song_features(song_id)
            cluster_num = clustify(X)           
            recommended_song = recommender(cluster_num)  # Get recommendation
            
            # Display only the relevant recommendation
            st.success(f"Based on '{song_input}' by {artist_input}, we recommend: '{recommended_song}'!")
        else:
            st.error("Could not find the song. Please try again.")
    else:
        st.warning("Please enter both the song name and the artist's name.")
