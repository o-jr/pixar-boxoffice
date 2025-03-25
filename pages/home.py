import os
import pandas as pd
import streamlit as st
import requests 
from streamlit_lottie import st_lottie 

DATA_DIR = 'data/'
# Set data directory
def load_data():
    try:
        merged_df = pd.read_csv(os.path.join(DATA_DIR, 'box_office.csv'))
        genres_df = pd.read_csv(os.path.join(DATA_DIR, 'genres.csv'))
        return merged_df, genres_df
    except FileNotFoundError as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame()

merged_df, genres_df = load_data()

st.set_page_config(
    page_title="Pixar Dashboard",
    page_icon="img/logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

def display_sidebar_filters():
    st.sidebar.page_link("pages/home.py", label="Home", icon="🏠")
    st.sidebar.page_link("app.py", label="Box Office", icon="🎟️")
    st.sidebar.page_link("pages/movies.py", label="Movies", icon="🎬")

main_col1, main_col4, main_col5, main_col2 = st.columns([1, 1, 1, 2])
main_col1.image("img/logopixar.png", width=350)

# Ensure merged_df has data (replace with proper merging if needed)
films = merged_df['film'].unique().tolist()  # Use .tolist() for list conversion

# Split films into rows of 7
rows = [films[i:i + 14] for i in range(0, len(films), 14)]

for row in rows:
    cols = st.columns(14)
    for col, film in zip(cols, row):
        image_path = os.path.join('img', f'{film}.jpg')
        if os.path.exists(image_path):
            col.image(image_path, width=250)  # Only display the image, no text
        else:
            pass

# ... [rest of the code remains the same] ...
mid0, mid1, mid2, mid3 = st.columns([0.2, 1, 0.2,0.2])

mid1.markdown("<h6 style='text-align: center; color: black;'>🎬 For three decades, Disney Pixar has captivated audiences worldwide, redefining animated storytelling. More than just films, they've created cultural touchstones, sparking our imaginations and tugging at our heartstrings. Let's explore the extraordinary journey of Pixar's Golden Age. </h6>", unsafe_allow_html=True)

# Button to open app.py
if mid2.button("Box Office", icon="🎟️"):
    mid2.page_link("app.py", label="Box Office")

# Button to open data/movies.py
if mid3.button("Pixar Movies", icon="🎬"):
    mid2.page_link("pages/movies.py", label="Pixar Movies")


####### botton section ########################
bott1, bott2, bott3 = st.columns([1,0.8,1])
url = requests.get( 
    "https://lottie.host/5610cf3b-e6cb-4989-822c-02b7205be714/G0Ba20zaPI.json") 
url_json = dict() 
if url.status_code == 200: 
    url_json = url.json() 
else: 
    print("Error in URL") 
   
  
with bott1:
    st_lottie(
        url_json,
        reverse=True,  # Change the direction of the animation
        height=300,    # Height of the animation
        width=None,     # Set width to None to automatically adjust to column width
        speed=1,       # Speed of the animation
        loop=True,     # Run the animation forever
        quality='high',  # Quality of the animation
        key='Cloud'     # Unique identifier for the animation
    )

bott2.image("./img/toygif.gif")

with bott3:
    st_lottie(
        url_json,
        reverse=True,  # Change the direction of the animation
        height=300,    # Height of the animation
        width=None,     # Set width to None to automatically adjust to column width
        speed=1,       # Speed of the animation
        loop=True,     # Run the animation forever
        quality='high',  # Quality of the animation
        key='Cloud2'     # Unique identifier for the animation
    )

def main():
    display_sidebar_filters()

if __name__ == "__main__":
    main()