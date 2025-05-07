import os
import pandas as pd
import streamlit as st  
import numpy as np # For sample data if needed

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.ticker import FuncFormatter

st.set_page_config(
    page_title="Pixar Dashboard",
    page_icon="img/logo.png", #logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)


def display_sidebar_filters():
    home = st.sidebar.page_link("pages/home.py", label="Home", icon="🏠")
    year_filter = st.sidebar.page_link("app.py", label="Box Office", icon="🎟️")
    age_filter = st.sidebar.page_link("pages/movies.py", label="Movies", icon="🎬")


# Display the KPIs
def format_number(number, currency=None):
    """ Format a number into a string with optional currency and abbreviation units.    
    If a currency symbol is provided, it will be prepended to the formatted number.
    The function uses '', 'K', 'M' (and 'B' if needed) as units. """
    currency = '' if currency is None else currency + ' '
    for unit in ['', 'K', 'M']:
        if abs(number) < 1000.0:
            return f"{currency}{number:6.2f}{unit}"
        number /= 1000.0
    return f"{currency}{number:6.2f}B"


data_dir = 'data/'

# Load data from all CSV files in the data directory
dfs = [pd.read_csv(os.path.join(data_dir, file_name)) for file_name in os.listdir(data_dir)
    if file_name.endswith('.csv') and file_name != 'public_response.csv']

# Merge all DataFrames into a single DataFrame
merged_df = pd.concat(dfs, ignore_index=True)

# Extract genres from file names
genres_df = pd.read_csv(os.path.join(data_dir, 'genres.csv'))
genres = genres_df['value'].unique().tolist()

main_col1, main_col3, main_col4, main_col2 = st.columns([1,1,1,2])
main_col1.image("img/logopixar.png", width=350) #logo.png, width=150)

################### KPIS ##########################################
# Load the box office data and calculate the KPIs
###################################################################
box_of_df = pd.read_csv(os.path.join(data_dir, 'box_office.csv'))
academy_df = pd.read_csv(os.path.join(data_dir, 'academy.csv'))
genres_df = pd.read_csv(os.path.join(data_dir, 'genres.csv'))

########### Filter ##############################################
# Create a segmented control widget for unique values where category is Genre
unique_genres1 = genres_df[genres_df['category'] == 'Genre']['value'].unique().tolist()
unique_option_map = {genre: genre for genre in unique_genres1}
selected_unique_genre1 = main_col2.segmented_control(
    "",
    options=unique_option_map.keys(),
    format_func=lambda option: unique_option_map[option],
    selection_mode="multi",
    key="unique_genre_segmented_control_2"  # Changed key
)

# Filter the genres DataFrame based on selected unique genre
if selected_unique_genre1:
    filtered_genres_df = genres_df[genres_df['value'].isin(selected_unique_genre1)]
    filtered_films = filtered_genres_df['film'].unique().tolist()
else:
    filtered_films = genres_df['film'].unique().tolist()

# Filter box office and academy data based on filtered films
filtered_box_of_df = box_of_df[box_of_df['film'].isin(filtered_films)]
filtered_academy_df = academy_df[academy_df['film'].isin(filtered_films)]

# Calculate the sum total of budget and box_office_worldwide for filtered films
total_budget = filtered_box_of_df['budget'].sum()
total_box_office = filtered_box_of_df['box_office_worldwide'].sum()
total_box_other = filtered_box_of_df['box_office_other'].sum()
total_net = total_box_office - total_budget

# Calculate the ROI percentage
roi_percentage = ((total_box_office - total_budget) / total_budget) * 100 if total_budget > 0 else 0

# Calculate the total number of awards won and nominated for filtered films
total_won = filtered_academy_df[filtered_academy_df['status'].isin(['Won', 'Won Special Achievement'])].shape[0]
total_nominated = filtered_academy_df[filtered_academy_df['status'] == 'Nominated'].shape[0]

#Removing Delta Arrows
delta_budget = len(filtered_films)
st.write(
    """
    <style>
    [data-testid="stMetricDelta"] svg {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Create 4 columns for the KPIs
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

# Use the formatting function to format and display metric values.
kpi_col1.metric(
    label="Budget",
    value=format_number(total_budget, '$'),
    delta=f"Films {delta_budget:,.0f}",
    delta_color="off",
    border=True
)

kpi_col2.metric(
    label="Worldwide Box Office",
    value=format_number(total_box_office, '$'),
    delta=f"Foreign {format_number(total_box_other, '$')}",
    delta_color="off",
    border=True
)

kpi_col3.metric(
    label="Total Net",
    value=format_number(total_net, '$'),
    delta=f"ROI {roi_percentage:.1f}%",
    delta_color="off",
    border=True
)

kpi_col4.metric(
    label="Awards Won",
    value=total_won,
    delta=f"Nominated {total_nominated:,.0f}",
    delta_color="off",
    border=True
)


################## Line Chart ##############################
# Ensure 'release_date' and box office columns exist in the DataFrames
# Assuming data_dir is defined


# Setup layout
container = st.container(border=True)
data_dir = "data/"

# Load data
pixar_df = pd.read_csv(os.path.join(data_dir, 'pixar_films.csv'))
box_office_df = pd.read_csv(os.path.join(data_dir, 'box_office.csv'))

# Check required columns
required_columns = {
    'pixar_df': ['film', 'release_date'],
    'box_office_df': ['film', 'box_office_us_canada', 'box_office_other', 'box_office_worldwide', 'budget']
}
if all(col in pixar_df.columns for col in required_columns['pixar_df']) and \
   all(col in box_office_df.columns for col in required_columns['box_office_df']):

    # Merge and preprocess
    merged_box_office_df = pd.merge(pixar_df[['film', 'release_date']], box_office_df, on='film', how='left')
    merged_box_office_df['release_date'] = pd.to_datetime(merged_box_office_df['release_date']).dt.year

    # Melt data
    melted_df = merged_box_office_df.melt(
        id_vars=['release_date'],
        value_vars=['box_office_us_canada', 'box_office_other', 'box_office_worldwide', 'budget'],
        var_name='Box Office Type',
        value_name='Revenue'
    )

    # Rename box office types
    melted_df['Box Office Type'] = melted_df['Box Office Type'].replace({
        'box_office_us_canada': 'Domestic',
        'box_office_other': 'Foreign',
        'box_office_worldwide': 'Worldwide',
        'budget': 'Budget'
    })

    # Layout
    column1, columnnone, column2 = st.columns([3, 0.2, 3])

    column1.markdown(
        "<span style='font-size:20px; font-weight:bold;'>"
        "How have Pixar's revenues evolved?"
        "</span>",
        unsafe_allow_html=True
    )

    with container:
        selected_box_office_types = column1.segmented_control(
            label="Select Box Office Types",
            options=['Domestic', 'Foreign', 'Worldwide', 'Budget'],
            selection_mode="multi",
            default=['Domestic', 'Foreign', 'Worldwide', 'Budget']
        )

    filtered_df = melted_df[melted_df['Box Office Type'].isin(selected_box_office_types)]

    # Define annotations (adjusted release_date to a numeric year)
    ANNOTATIONS = [
        (2020, 0, "😷", "_COVID-19")
    ]
    annotations_df = pd.DataFrame(ANNOTATIONS, columns=['release_date', 'Revenue', 'emoji', 'label'])

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each box office type
    for bo_type in filtered_df['Box Office Type'].unique():
        subset = filtered_df[filtered_df['Box Office Type'] == bo_type]
        ax.plot(subset['release_date'], subset['Revenue'], label=bo_type, marker='o')

    # Format axes
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Format Y-axis as millions
    def millions_formatter(x, pos):
        return f'${int(x * 1e-6)}M'

    ax.yaxis.set_major_formatter(FuncFormatter(millions_formatter))

    # Add grid and legend
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper left')

    # Add annotations
    for idx, row in annotations_df.iterrows():
        # Emoji annotation
        ax.annotate(
            row['emoji'],
            xy=(row['release_date'], row['Revenue']),
            xytext=(10, -10),
            textcoords='offset points',
            fontsize=20,
            ha='left', va='center'
        )

        # Label annotation
        ax.annotate(
            row['label'],
            xy=(row['release_date'], row['Revenue']),
            xytext=(30, -10),
            textcoords='offset points',
            fontsize=12,
            ha='left', va='center'
        )

    # Adjust layout
    plt.tight_layout()

    # Display the plot
    column1.pyplot(fig)


########### Awards Table Side by Side ##############################
# Create a list of awards
academy_df = pd.read_csv(os.path.join(data_dir, 'academy.csv'))
pixar_df = pd.read_csv(os.path.join(data_dir, 'pixar_films.csv'))

merged_awards_df = pd.merge(pixar_df, academy_df[['film', 'status', 'award_type']], on='film', how='left')
merged_awards_df['release_date'] = pd.to_datetime(merged_awards_df['release_date']).dt.year
merged_awards_df = merged_awards_df[['release_date', 'film', 'status', 'award_type']]
filtered_awards_df = merged_awards_df[merged_awards_df['status'].isin(['Won', 'Nominated','Won Special Achievement'])]


###### Won and Nomiated Filter ##############################
# Highlight 'Won' status with light green color
def highlight_won(s):
    return ['background-color: lightgreen' if v in ['Won', 'Won Special Achievement'] else '' for v in s]

col1, colnone, col2 = st.columns([3,0.2,3])

col2.markdown(
    "<span style='font-size:20px; font-weight:bold;'>"
    "Which  movies obtained nominations or wins for awards?"
    "</span>",
    unsafe_allow_html=True
)
# Create multiselect for status filtering
status_options = ['Nominated', 'Won']  # Display options
status_selection = col2.segmented_control(
    label="",
    selection_mode="multi",
    options=status_options,
    default=status_options,  # Default to showing both
    help="Select which award statuses to display",
    
)

# Map the selections to include 'Won Special Achievement' with 'Won'
filtered_status = []
if 'Won' in status_selection:
    filtered_status.extend(['Won', 'Won Special Achievement'])
if 'Nominated' in status_selection:
    filtered_status.append('Nominated')

# Filter the dataframe based on selection
if filtered_status:
    display_df = filtered_awards_df[filtered_awards_df['status'].isin(filtered_status)]
else:
    display_df = filtered_awards_df  # Show all if no selection

# Add 🏆 emoji before award_type values
display_df = display_df.copy()  # Avoid modifying the original DataFrame
if 'Won' in status_selection:
    display_df['award_type'] = display_df.apply(
        lambda row: f"🏆 {row['award_type']}" if row['status'] in ['Won', 'Won Special Achievement'] and pd.notnull(row['award_type']) else row['award_type'],
        axis=1
    )







# Display SCATTERPLOT the filtered DataFrame
########################################
########################################   
# Ensure 'budget' and 'box_office_worldwide' columns exist in the box office DataFrame

box_office_df = pd.read_csv(os.path.join(data_dir, 'box_office.csv'))
pixar_df = pd.read_csv(os.path.join(data_dir, 'pixar_films.csv'))


# Ensure 'budget' and 'box_office_worldwide' columns exist in the box office DataFrame
if 'budget' in box_office_df.columns and 'box_office_worldwide' in box_office_df.columns:
    # Calculate ROI
    box_office_df['ROI'] = ((box_office_df['box_office_worldwide'] - box_office_df['budget']) / box_office_df['budget']) * 100
    # Format ROI to one decimal place with a % sign
    box_office_df['ROI'] = box_office_df['ROI'].map('{:.1f}%'.format)

    merged_awards_df['release_date'] = pd.to_datetime(merged_awards_df['release_date']).dt.year
    release_dates = merged_awards_df['release_date']

    # Create scatter plot using Matplotlib
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(box_office_df['budget'], box_office_df['box_office_worldwide'], s=100, alpha=0.6)

    # Add titles and labels
    ax.set_xlabel('Budget ($)', fontsize=12)
    ax.set_ylabel('Worldwide Box Office ($)', fontsize=12)
    ax.set_title('Relationship Between Film Budget and Worldwide Box Office Success', fontsize=14, fontweight='bold')
    
    # Format the axes to display dollars with commas
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'${y:,.0f}'))

    # Create a legend with film names and additional info
    for i, row in box_office_df.iterrows():
        ax.annotate(f"{row['film']}\nRelease: {release_dates[i]}\nROI: {row['ROI']}",
                    (row['budget'], row['box_office_worldwide']),
                    textcoords="offset points",
                    xytext=(5,-5),
                    ha='right',
                    fontsize=9)
        
    # Display the plot in Streamlit        
    column2.markdown(
        "<span style='font-size:20px; font-weight:bold;'>"
        "What is the relationship between a film's budget and worldwide box office success? <br>"
        "</br>"
        "</span>",
        unsafe_allow_html=True
    )
    
    with column2:        
        plt.tight_layout()
        st.pyplot(fig)
else:
    print("Required columns 'budget' or 'box_office_worldwide' are missing in the DataFrame.")

with col2:
    #st.write("### Films thats Won or Nominated for a award:")
    st.dataframe(display_df.style.apply(highlight_won, subset=['status']), height=550)



######################################################################################
########################################### directors ###########################################
# --- Helper: Load CSV with Error Handling ---
def load_csv_data(filename, required_cols):
    filepath = os.path.join(data_dir, filename)
    try:
        df = pd.read_csv(filepath)
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns in {filename}: {', '.join(missing_cols)}")
            return None, False
        return df, True
    except FileNotFoundError:
        st.error(f"'{filename}' not found in '{data_dir}'. Please add it.")
        return None, False
    except Exception as e:
        st.error(f"Error loading {filename}: {e}")
        return None, False
# Load data
people_cols = ['film', 'name', 'role_type']
box_office_cols_people = ['film', 'box_office_worldwide']
box_office_cols_genre = ['film', 'box_office_worldwide']
genre_cols = ['film', 'category', 'value']

pixar_people_df, ok1 = load_csv_data('pixar_people.csv', people_cols)
box_office_df, ok2 = load_csv_data('box_office.csv', box_office_cols_people)
genres_df, ok3 = load_csv_data('genres.csv', genre_cols)

# --- 2. Prepare Chart 2 Data (Subgenres) - Before Columns ---
genre_box_office_sum_df = None
if ok3 and ok2 and genres_df is not None and box_office_df is not None:
    try:
        genre_df_filtered = genres_df[genres_df['category'] == 'Subgenre']
        required_merge_cols = [col for col in box_office_cols_genre if col in box_office_df.columns]
        merged_genre_df = pd.merge(
            genre_df_filtered,
            box_office_df[['film'] + [col for col in required_merge_cols if col != 'film']],
            on='film',
            how='left'
        )
        merged_genre_df.dropna(subset=['box_office_worldwide'], inplace=True)

        if not merged_genre_df.empty:
            genre_box_office_sum_df_agg = merged_genre_df.groupby('value')['box_office_worldwide'].sum().reset_index()
            genre_box_office_sum_df_agg = genre_box_office_sum_df_agg.rename(columns={'value': 'genre', 'box_office_worldwide': 'total_box_office_worldwide'})
            genre_box_office_sum_df = genre_box_office_sum_df_agg.nlargest(18, 'total_box_office_worldwide') # Limit items

    except Exception as e:
        st.warning(f"Could not prepare subgenre data: {e}")
        genre_box_office_sum_df = None

# --- 3. Create Columns ---
col_people, col_spacer, col_genre = st.columns([3, 0.2, 3])

# --- 4. Column 1 (col_people): People Chart ---
with col_people:
    st.markdown(
        "<span style='font-size:18px; font-weight:bold;'>People Behind the Numbers</span>",
        unsafe_allow_html=True,
    )

    role_options_def = ['Director', 'Musician', 'Producer', 'Screenwriter', 'Storywriter', 'Co-director']
    selected_role = None
    people_chart = None

    if ok1 and pixar_people_df is not None:
        available_roles = sorted(pixar_people_df['role_type'].unique())
        role_options = [r for r in role_options_def if r in available_roles]
        default_role_ix = 0
        if 'Director' in role_options:
            default_role_ix = role_options.index('Director')
        elif role_options:
            default_role_ix = 0
        else:
            role_options = []

        if role_options:
            selected_role = st.segmented_control( # Using selectbox
                "",
                options=role_options,
                selection_mode="single",
                default="Director",
                key='role_selector'
            )
        else:
            st.warning("No roles found in people data.")

    # --- Prepare People Chart Data (Inside Column, based on selection) ---
    if selected_role and ok1 and ok2 and pixar_people_df is not None and box_office_df is not None:
        people_chart_data = None
        try:
            filtered_df = pixar_people_df[pixar_people_df['role_type'] == selected_role]
            required_merge_cols_ppl = [col for col in box_office_cols_people if col in box_office_df.columns]
            merged_people_df = pd.merge(
                filtered_df,
                box_office_df[['film'] + [col for col in required_merge_cols_ppl if col != 'film']],
                on='film',
                how='left'
            )
            merged_people_df.dropna(subset=['box_office_worldwide'], inplace=True)

            if not merged_people_df.empty:
                people_total_revenue_df = merged_people_df.groupby('name', as_index=False).agg(
                    box_office_worldwide=('box_office_worldwide', 'sum'),
                    films=('film', lambda x: ', '.join(sorted(x.unique())))
                )
                people_total_revenue_df = people_total_revenue_df.nlargest(18, 'box_office_worldwide') # Limit items
                people_chart_data = people_total_revenue_df

        except Exception as e:
            st.warning(f"Could not prepare data for role '{selected_role}': {e}")
            people_chart_data = None


        selected_role = "Revenue"
        ok1 = False  # Set this to True if you want to test the warning message

        # --- Create People Chart ---
    if people_chart_data is not None and not people_chart_data.empty:
        y_axis_title = selected_role

        # Sort data by box_office_worldwide in descending order
        people_chart_data = people_chart_data.sort_values(by='box_office_worldwide', ascending=False)

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(5, 4))

        # Create bar chart
        bars = ax.barh(people_chart_data['name'], people_chart_data['box_office_worldwide'])

        # Set y-axis title
        ax.set_ylabel(y_axis_title)

        # Set x-axis title
        ax.set_xlabel('Total Worldwide Box Office')

        # Add labels and title
        ax.set_title(f'Revenue Data for {y_axis_title}')

        # Format x-axis tick labels
        ax.tick_params(axis='x', labelrotation=45)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "${:,.0f}".format(x)))

        # Add text labels to bars
        for i, v in enumerate(people_chart_data['box_office_worldwide']):
            ax.text(v, i, "${:.2f}".format(v), color='black', ha='left', va='center')

        people_chart = fig

    elif selected_role:
        print(f"No revenue data to display for {selected_role}.")

    if people_chart:
        st.pyplot(people_chart)
    elif selected_role is None and ok1:
        st.warning("Select a role to view data.")

# --- 5. Column 2 (col_genre): Subgenre Chart ---
with col_genre:
    st.markdown(
        "<span style='font-size:18px; font-weight:bold;'>Which subgenres revealed the most lucrative for the studio?</span>",
        unsafe_allow_html=True,
    )

    # --- Spacer to align with selectbox in the other column ---
    # This aims to add vertical space roughly equivalent to the st.selectbox
    # Adjust height (e.g., 2.5rem, 42px) as needed based on visual inspection
    st.markdown("<div style='height: 2.2rem;'></div>", unsafe_allow_html=True)

    genre_chart = None

    if genre_box_office_sum_df is not None and not genre_box_office_sum_df.empty:
    # Sort the DataFrame by total_box_office_worldwide for plotting
        genre_box_office_sum_df = genre_box_office_sum_df.sort_values('total_box_office_worldwide', ascending=True)

        # Set up the figure
        fig, ax = plt.subplots(figsize=(8, 10))  # Adjust size as needed

        # Plot horizontal bars
        bars = ax.barh(
            genre_box_office_sum_df['genre'],
            genre_box_office_sum_df['total_box_office_worldwide'],
            color='skyblue'
        )

        # Add text labels to the bars
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + 0.02 * genre_box_office_sum_df['total_box_office_worldwide'].max(),  # position slightly to the right
                bar.get_y() + bar.get_height() / 2,
                f"${width/1e9:.2f}B",  # Format as billions
                va='center',
                fontsize=9,
                color='black'
            )

        # Remove spines and ticks for cleaner look
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='x', bottom=False, labelbottom=False)
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_title('')  # Add title if needed

        # Display the chart in Streamlit
        st.pyplot(fig)

    else:
        st.info("No subgenre box office data available to display.")


#########################################################
#######SCORES ##################################################
unique_genres = genres_df[genres_df['category'] == 'Genre']['value'].unique().tolist()
unique_option_map = {genre: genre for genre in unique_genres}


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

films = merged_df['film'].unique()
############# Awards Table and Scatterplot Side by Side ##########################################

if 'film' in merged_df.columns and 'imdb_score' in merged_df.columns and 'cinema_score' in merged_df.columns:
    # Define the sorting options
    sort_options = {
        'IMDB Score': 'imdb_score',
        'Metacritic Score': 'metacritic_score',
        'Rotten Tomatoes Score': 'rotten_tomatoes_score'
    }

    # Create selectbox for sorting metric (single selection)
    col1.markdown(
    "<span style='font-size:20px; font-weight:bold;'>"
    "What are the top films based on audience reception?"
    "</span>",
    unsafe_allow_html=True
)
    selected_metric = col1.segmented_control(
        label="",
        selection_mode="single",
        options=list(sort_options.keys()),
        default="IMDB Score", 
        help="Select which score to sort the films by"
    )

    # Get the actual column name for sorting
    sort_column = sort_options[selected_metric]

    # Sort the DataFrame by the selected metric in descending order
    # Make a copy to avoid modifying the original dataframe
    sorted_df = merged_df.copy()
    sorted_df = sorted_df.sort_values(by=sort_column, ascending=False)

    # Fill missing values in 'cinema_score' with a placeholder
    sorted_df['cinema_score'] = sorted_df['cinema_score'].fillna('N/A')

    # Assuming sorted_df is your DataFrame, replace with your actual DataFrame name
# Create the Matplotlib figure

    # Remove duplicates based on the 'film' column
    sorted_df = sorted_df.drop_duplicates(subset=['film'])

    # Create the Matplotlib figure
    fig, ax = plt.subplots(figsize=(10, len(sorted_df) * 0.5))  # Adjust figure size based on number of films

    # Define color mapping
    color_map = {
        'A+': '#3399FF',
        'A': 'lightblue',
        'A-': '#FFAB5B',
        'N/A': '#A5BFCC'
    }

    # Create bars
    y_positions = np.arange(len(sorted_df))  # One position per unique film
    bars = ax.barh(y_positions, 
                sorted_df[sort_column],  # Using the same sort_column
                color=[color_map.get(score, '#A5BFCC') for score in sorted_df['cinema_score']])

    # Customize the plot
    ax.set_yticks(y_positions)
    ax.set_yticklabels(sorted_df['film'])  # Use unique film names
    ax.invert_yaxis() 

    # Add text labels 
    for i, bar in enumerate(bars):
        width = bar.get_width()
        # Position text at the end of the barr
        ax.text(
            x=width + max(sorted_df[sort_column]) * 0.05,  # Explicitly use keyword 'x'
            y=bar.get_y() + bar.get_height()/2,            # Explicitly use keyword 'y'
            s=f'{width:.1f}',                              # Explicitly use keyword 's' for the text
            ha='left',                                     # Horizontal alignment
            va='center',                                   # Vertical alignment
            fontsize=13,
            color='black'
        )

    # Remove x-axis labels, ticks, and domain
    ax.set_xticks([])
    ax.set_xlabel('')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Add a legend for the colors
    legend_elements = [plt.Rectangle((0,0),1,1, color=color, label=score) 
                    for score, color in color_map.items()]
    ax.legend(handles=legend_elements, title='Cinema Score', 
            bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Streamlit's matplotlib 
    col1.pyplot(fig)


def main():
    filtered_df = display_sidebar_filters()
if __name__ == "__main__":
    main()