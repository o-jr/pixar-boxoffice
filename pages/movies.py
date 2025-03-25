import os
import pandas as pd
import streamlit as st  
import requests 
from bs4 import BeautifulSoup


st.set_page_config(
    page_title="Pixar Dashboard",
    page_icon="img/logo.png",
    layout="wide"
)

def display_sidebar_filters():
    st.sidebar.page_link("pages/home.py", label="Home", icon="🏠")
    st.sidebar.page_link("app.py", label="Box Office", icon="🎟️")
    st.sidebar.page_link("pages/movies.py", label="Movies", icon="🎬")

def format_number(number, currency=None):
                    if number is None or pd.isna(number):
                      return "N/A"
        
                    original_number = number
                    currency_prefix = '' if currency is None else currency + ' '
    
                    for unit in ['', 'K', 'M', 'B']:
                        if abs(number) < 1000.0:
                            return f"{currency_prefix}{number:.0f}{unit}"
                        number /= 1000.0    
                    return f"{currency_prefix}{original_number:.0f}"     


main_col1, main_col3, main_col4, main_col2 = st.columns([1,1,1,2])
main_col1.image("img/logopixar.png", width=350)

data_dir = 'data/'

    # Load data from all CSV files in the data directory
dfs = [pd.read_csv(os.path.join(data_dir, file_name)) for file_name in os.listdir(data_dir)
    if file_name.endswith('.csv') and file_name != 'public_response.csv']

    # Merge all DataFrames into a single DataFrame
merged_df = pd.concat(dfs, ignore_index=True)

# Load the data files
box_of_df = pd.read_csv(os.path.join(data_dir, 'box_office.csv'))
academy_df = pd.read_csv(os.path.join(data_dir, 'academy.csv'))
genres_df = pd.read_csv(os.path.join(data_dir, 'genres.csv'))


########### Filter ##############################################
# Create a segmented control widget for unique values where category is Genre
unique_genres = genres_df[genres_df['category'] == 'Genre']['value'].unique().tolist()
unique_option_map = {genre: genre for genre in unique_genres}
selected_unique_genre = main_col2.segmented_control(
    "",
    options=unique_option_map.keys(),
    format_func=lambda option: unique_option_map[option],
    selection_mode="multi"
)

# Filter the genres DataFrame based on selected unique genre
if selected_unique_genre:
    filtered_genres_df = genres_df[genres_df['value'].isin(selected_unique_genre)]
    filtered_genres = filtered_genres_df['value'].unique().tolist()
else:
    filtered_genres = unique_genres

# Initialize an empty DataFrame to store the merged data
merged_df = pd.DataFrame()

for file_name in os.listdir(data_dir):
    if file_name.endswith('.csv') and file_name != 'public_response.csv':
        file_path = os.path.join(data_dir, file_name)
        df = pd.read_csv(file_path)
        # No need to merge DataFrames, just process each file individually
        merged_df = merged_df._append(df, ignore_index=True)

# Load the public response data and merge it with the merged DataFrame
public_response_df = pd.read_csv(os.path.join(data_dir, 'public_response.csv'))
merged_df = pd.merge(merged_df, public_response_df, on='film', how='left')

# Filter the DataFrame based on filtered genres
if filtered_genres:
    merged_df = merged_df[merged_df['value'].isin(filtered_genres)]

##########  picture and KPIS ##############################
# Define the film name outside the conditional block
# Split the images into 4 rows with 7 images per row
films = merged_df['film'].unique()
rows = [films[i:i + 7] for i in range(0, len(films), 7)]

def format_runtime(minutes):
    hours = minutes // 60
    remaining_minutes = minutes % 60
    
    if hours > 0:
        return f"{hours}h {remaining_minutes}m"
    else:
        return f"{minutes}m"
    
    
  

for row in rows:
    cols = st.columns(7)
    for col, film in zip(cols, row):
        image_path = os.path.join('img', f'{film}.jpg')
        if os.path.exists(image_path):
            if col.button(film, key=film):
                # Create a container with a grey border
                with st.container(border=True):
                    # Calculate the KPIs for the selected film
                    film_box_of_df = box_of_df[box_of_df['film'] == film]
                    film_academy_df = academy_df[academy_df['film'] == film]

                    total_budget = film_box_of_df['budget'].sum()
                    total_box_office = film_box_of_df['box_office_worldwide'].sum()
                    foreign_box_office = film_box_of_df['box_office_other'].sum()
                    domestic_box_office = film_box_of_df['box_office_us_canada'].sum()
                    roi_percentage = ((total_box_office - total_budget) / total_budget) * 100 if total_budget != 0 else 0
                    total_won = film_academy_df[film_academy_df['status'] == 'Won'].shape[0]
                    total_nominated = film_academy_df[film_academy_df['status'] == 'Nominated'].shape[0]
                    col01, col02, col03, col04, col05 = st.columns(5)
                    col01.metric(label="Budget", value=f"{format_number(total_budget, currency='$')}",border=True)
                    col02.metric(label="Worldwide Box Office", value=f"{format_number(total_box_office, currency='$')}",border=True)
                    col03.metric(label="ROI", value=f"{roi_percentage:.1f}%",border=True)
                    col04.metric(label="Domestic Box Office", value=f"{format_number(domestic_box_office, currency='$')}",border=True)
                    col05.metric(label="Foreign Box Office", value=f"{format_number(foreign_box_office, currency='$')}",border=True)

                    # Load the public response data
                    public_response_df = pd.read_csv(os.path.join(data_dir, 'public_response.csv'))
                    film_df = public_response_df[public_response_df['film'] == film]
                    pixar_films_df = pd.read_csv(os.path.join(data_dir, 'pixar_films.csv'))
                    film_rating = pixar_films_df[pixar_films_df['film'] == film]['film_rating'].dropna().max()

                    film_academy_df = academy_df[academy_df['film'] == film]
                    total_won = film_academy_df[film_academy_df['status'] == 'Won'].shape[0]                    
                    total_nominated = film_academy_df[film_academy_df['status'] == 'Nominated'].shape[0]
                     # Define a function to format KPI values
                    def format_kpi(value, counts, max_value=100, rating_symbol='⭐'):
                        value = int(value) if pd.notna(value) else 0
                        counts = int(counts) if pd.notna(counts) else 0
                        return f"{value}/{max_value} ({format_number(counts)}{rating_symbol})"

                # Display KPIs with reduced font size
                          # Display additional KPIs for the selected film
                    kpi_cols = st.columns([1, 2.5, 1, 2.5])
                    kpi_cols[0].image(image_path, caption=film)

                    kpi_cols[2].markdown("**Cinema Score**")
                    max_score = film_df['cinema_score'].dropna().max()
                    if pd.isna(max_score):  # Handle cases where there's no CinemaScore data
                        kpi_cols[2].markdown("<span style='font-size: 18px;'>N/A</span>", unsafe_allow_html=True)
                    # Determine image path based on the max score
                    if max_score == 'A+':
                        image_path = 'img/apos.jpg'
                    elif max_score == 'A':
                        image_path = 'img/a.jpg'
                    elif max_score == 'A-':
                        image_path = 'img/aneg.jpg'
                    else:
                        image_path = None  # Or a default "no score" image
                        kpi_cols[2].markdown(f"<span style='font-size: 18px;'> {max_score}</span>", unsafe_allow_html=True) #Shows the cinema score if is not defined in the options.
                    if image_path:
                        kpi_cols[2].image(image_path, width=38)  # Adjust width as needed
                        
                    image_rotten = 'img/rottens.jpg'
                    image_meta = 'img/metacritic.png'
                    image_imdb = 'img/IMDB.png'
                    
                    kpi_cols[2].markdown("**Rotten Tomatoes**")
                    with kpi_cols[2]:
                        col1, col2 = st.columns([1, 3])  # Adjust ratio as needed
                        with col1:
                            st.image(image_rotten, width=38)
                        with col2:
                            st.markdown(f"<span style='font-size: 18px;'>{format_kpi(film_df['rotten_tomatoes_score'].dropna().max(), film_df['rotten_tomatoes_counts'].dropna().max())}</span>", unsafe_allow_html=True)
                    kpi_cols[2].markdown("**Meta Critic**")
                    with kpi_cols[2]:
                        col1, col2 = st.columns([1, 3])  # Adjust ratio as needed
                        with col1:
                            st.image(image_meta, width=48)
                        with col2:
                            st.markdown(f"<span style='font-size: 18px;'> {format_kpi(film_df['metacritic_score'].dropna().max(), film_df['metacritic_counts'].dropna().max(), max_value=100, rating_symbol='⭐')}</span>", unsafe_allow_html=True)
                    
                    kpi_cols[2].markdown("**IMDB Score**")
                    with kpi_cols[2]:
                        col1, col2 = st.columns([1, 3])  # Adjust ratio as needed
                        with col1:
                            st.image(image_imdb, width=48)
                        with col2:
                            st.markdown(f"<span style='font-size: 18px;'>{format_kpi(film_df['imdb_score'].dropna().max(), film_df['imdb_counts'].dropna().max(), max_value=10, rating_symbol='⭐')}</span>", unsafe_allow_html=True)
                    
                    ### ++++++++++++++++++++++++++++++++++++++######################################################################

                    film_plot = pixar_films_df[pixar_films_df['film'] == film]['plot'].dropna().tolist()
                    film_plot = " ".join(film_plot)

                    ### ++++++++++++++++++++++++++++++++++++++######################################################################

                    # Load the pixar_people data
                    pixar_people_df = pd.read_csv(os.path.join(data_dir, 'pixar_people.csv'))
                    genres_df = pd.read_csv(os.path.join(data_dir, 'genres.csv'))

                    # Filter the DataFrame based on role_type and film
                    film_directors = pixar_people_df[(pixar_people_df['role_type'] == 'Director') & (pixar_people_df['film'] == film)]['name'].unique().tolist()
                    film_writers = pixar_people_df[(pixar_people_df['role_type'] == 'Storywriter') & (pixar_people_df['film'] == film)]['name'].unique().tolist()
                    film_producers = pixar_people_df[(pixar_people_df['role_type'] == 'Producer') & (pixar_people_df['film'] == film)]['name'].unique().tolist()
                    film_musicians = pixar_people_df[(pixar_people_df['role_type'] == 'Musician') & (pixar_people_df['film'] == film)]['name'].unique().tolist()

                    # Print the release_date from pixar_films.csv

                    release_dates = pd.to_datetime(pixar_films_df[pixar_films_df['film'] == film]['release_date']).dt.year.dropna().max()
                    # Get the run_time
                    run_time = pixar_films_df[pixar_films_df['film'] == film]['run_time'].dropna().max()

                    # Filter the DataFrame based on genre and subgenre and film
                    film_genre = genres_df[(genres_df['category'] == 'Genre')& (genres_df['film'] == film)]['value'].unique().tolist()
                    film_subgenre = genres_df[(genres_df['category'] == 'Subgenre')& (genres_df['film'] == film)]['value'].unique().tolist()

                    # Format the runtime
                    formatted_runtime = format_runtime(run_time)

                    # Display the KPIs for people involved in the film
                    people_col0, col_dates, people_col1, people_col2, people_col3, people_col4 = st.columns(6)
                    kpi_cols[1].write(f"{release_dates} - ***{film_rating}*** - {formatted_runtime}")
                    kpi_cols[1].markdown(f"<p style='font-size:20x;'> {film_plot}</p>", unsafe_allow_html=True)                    
                    kpi_cols[1].markdown(f"***Genres***: " + ", ".join(film_genre), unsafe_allow_html=True)                    
                    kpi_cols[1].markdown(f"***Sub Genres***: " + ", ".join(film_subgenre), unsafe_allow_html=True)
                    kpi_cols[1].markdown("""---""")
                    kpi_cols[1].markdown(f"***Directors***:  " + ", ".join(film_directors), unsafe_allow_html=True)
                    kpi_cols[1].markdown(f"***Storywriters***: " + ", ".join(film_writers), unsafe_allow_html=True)
                    kpi_cols[1].markdown(f"***Producers***: " + ", ".join(film_producers), unsafe_allow_html=True)
                    kpi_cols[1].markdown(f"***Musicians***: " + ", ".join(film_musicians), unsafe_allow_html=True)


                    # Load the trailer data
                    trailer_df = pd.read_csv(os.path.join(data_dir, 'trailer.csv'))
                    # Get the trailer URL for the selected film
                    trailer_url = trailer_df[trailer_df['film'] == film]['trailer'].values[0] if film in trailer_df['film'].values else ""
                    kpi_cols[3].video(trailer_url)
                    pixar_df = pd.read_csv(os.path.join(data_dir, 'pixar_films.csv'))

                    # Fix for the truncated code at the end
                    merged_awards_df = pd.merge(pixar_df, academy_df[['film', 'status', 'award_type']], on='film', how='left')
                    merged_awards_df['release_date'] = pd.to_datetime(merged_awards_df['release_date']).dt.year
                    
                    # Filter awards for the current film
                    film_awards_df = merged_awards_df[merged_awards_df['film'] == film]
                    
                    # Display awards information
                    if not film_awards_df.empty:
                        # Filter for only the relevant statuses
                        filtered_awards_df = film_awards_df[film_awards_df['status'].isin(['Won', 'Won Special Achievement', 'Nominated'])]
                        # Filter and sort the DataFrame
                        sorted_awards_df = filtered_awards_df.sort_values(by='status', key=lambda x: x.isin(['Won', 'Won Special Achievement']), ascending=False)

                        # Iterate over the sorted DataFrame
                        for idx, row in sorted_awards_df.iterrows():
                            award_text = row['award_type'] if pd.notna(row['award_type']) else "Award"
                            status_text = row['status'] if pd.notna(row['status']) else ""
                            
                            # Add trophy emoji and bold formatting for won awards
                            if status_text in ['Won', 'Won Special Achievement']:
                                award_text = f"🏆 ***{award_text}***"
                                
                            kpi_cols[3].markdown(f"<span style='font-size:18px'>{award_text} - {status_text}</span>", 
                                            unsafe_allow_html=True)



##############################################
##### WEB Scrapping Pixar Films Upcoming #####
##############################################
left, mid, right = st.columns([1, 10, 1])
def main():
    display_sidebar_filters()

    # URL of the Wikipedia page
    url = "https://en.wikipedia.org/wiki/List_of_Pixar_films"
    
    # Request the page content
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    
    # Locate the <div class="mw-heading mw-heading3"> elements
    headings_divs = soup.find_all("div", class_="mw-heading mw-heading3")
    
    # We will store data "Upcoming" -> Table of films
    heading_titles = ["Upcoming"]
    
    for heading_title in heading_titles:
        # Find the div with the h3 that has the matching id
        heading_div = None
        for div in headings_divs:
            h3_tag = div.find("h3", id=heading_title)
            if h3_tag:
                heading_div = div
                break
        
        if heading_div:
            # Display the heading text (e.g., "Released", "Upcoming")
            h3_tag = heading_div.find("h3")
            st.header(h3_tag.get_text())

            # Find the next table after the heading div
            # (the first "wikitable" that appears after this heading)
            table = heading_div.find_next("table", {"class": "wikitable"})
            if table:
                # Extract rows (tr)
                rows = table.find_all("tr")
                
                # Prepare lists to store the film and release date data
                films = []
                release_dates = []
                
                # Skip the header row(s) by slicing from rows[1:]
                for row in rows[1:]:
                    cols = row.find_all(["th", "td"])
                    
                    # Check if this row contains "Story" and "Screenplay" headers
                    if len(cols) >= 2 and cols[0].get_text(strip=True) == "Story" and cols[1].get_text(strip=True) == "Screenplay":
                        continue  # Skip this row
                    
                    # We expect at least 2 columns: Film, Release date
                    if len(cols) >= 2:
                        film_col = cols[0].get_text(strip=True)
                        date_col = cols[1].get_text(strip=True)
                        films.append(film_col)
                        release_dates.append(date_col)
                
                # Display the data
                # You could also make a table or dataframe
                for film, date in zip(films, release_dates):
                    st.write(f"**{film}** – {date}")

            else:
                st.write("No table found under this heading.")


            mid.image("img/movies.png")

          

if __name__ == "__main__":
    main()