import os
import pandas as pd
import streamlit as st  
import altair as alt


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
# Load the data
container = st.container(border=True)
data_dir = "data/"  # Replace with the actual path to your data directory
pixar_df = pd.read_csv(os.path.join(data_dir, 'pixar_films.csv'))
box_office_df = pd.read_csv(os.path.join(data_dir, 'box_office.csv'))

# Check if required columns exist
if ('release_date' in pixar_df.columns and 
    'box_office_us_canada' in box_office_df.columns and 
    'box_office_other' in box_office_df.columns and 
    'box_office_worldwide' in box_office_df.columns and
    'budget' in box_office_df.columns
    ):

    # Merge the DataFrames on 'film'
    merged_box_office_df = pd.merge(pixar_df[['film', 'release_date']], box_office_df, on='film', how='left')
    merged_box_office_df['release_date'] = pd.to_datetime(merged_box_office_df['release_date']).dt.year

    # Melt the DataFrame to have box office columns in a single column
    melted_df = merged_box_office_df.melt(id_vars=['release_date'], 
                                          value_vars=['box_office_us_canada', 'box_office_other', 'box_office_worldwide','budget'], 
                                          var_name='Box Office Type', 
                                          value_name='Revenue')

    # Rename the Box Office Type labels
    melted_df['Box Office Type'] = melted_df['Box Office Type'].replace({
        'box_office_us_canada': 'Domestic',
        'box_office_other': 'Foreign',
        'box_office_worldwide': 'Worldwide',
        'budget': 'Budget'
    })


    # Create two columns for layout
    column1, columnnone, column2 = st.columns([3,0.2,3])

    column1.markdown(
        "<span style='font-size:20px; font-weight:bold;'>"
        "How have Pixar's revenues evolved?"
        "</span>",
        unsafe_allow_html=True
    )

    # Add a segmented control for multi-selection of box office types
    with container:
        selected_box_office_types = column1.segmented_control(
            label="",
            options=['Domestic', 'Foreign', 'Worldwide','Budget'],
            selection_mode="multi",
            default=['Domestic', 'Foreign', 'Worldwide','Budget']
    )

    # Filter the melted DataFrame based on the selected box office types
    filtered_df = melted_df[melted_df['Box Office Type'].isin(selected_box_office_types)]

    # Create a line chart using Altair
    ANNOTATIONS = [
    ("20-22", 0, "😷" , "_COVID-19"),  # (release_date, Revenue, emoji, label)
]

# Create the base line chart using Altair
line_chart = alt.Chart(filtered_df).mark_line().encode(
    x=alt.X('release_date:O', title=None),  # Remove x-axis label
    y=alt.Y('Revenue:Q', title=None),  # Remove y-axis label
    color='Box Office Type:N',
    tooltip=[
        alt.Tooltip('release_date', type='nominal', title='Release Date'),
        'Box Office Type',
        alt.Tooltip('Revenue', type='quantitative', format='$,.0f', title='Worldwide Box Office')
    ]
).properties(
    height=400  # Remove hardcoded width for responsiveness
)

# Create a DataFrame for annotations
annotations_df = pd.DataFrame(ANNOTATIONS, columns=['release_date', 'Revenue', 'emoji', 'label'])

# Create the annotation layer (emoji + text)
annotation_layer = alt.Chart(annotations_df).mark_text(
    align='left',
    baseline='middle',
    dx=10,  # Horizontal offset for the emoji
    dy=-10,  # Vertical offset for the emoji
    fontSize=20,  # Size of the emoji
).encode(
    x=alt.X('release_date:O'),
    y=alt.Y('Revenue:Q'),
    text='emoji'
) + alt.Chart(annotations_df).mark_text(
    align='left',
    baseline='middle',
    dx=30,  # Horizontal offset for the label
    dy=-10,  # Vertical offset for the label
    fontSize=12,  # Size of the label text
).encode(
    x=alt.X('release_date:O'),
    y=alt.Y('Revenue:Q'),
    text='label'
)

# Combine the line chart and annotation layers
final_chart = (line_chart + annotation_layer).properties(
    #title='Box Office Revenue Over the Years'
)

# Display the line chart in Streamlit
column1.altair_chart(final_chart, use_container_width=True)


########### Awards Table and Scatterplot Side by Side ##############################
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

# Display the filtered DataFrame
########################################
########################################   
# Load the box office data
box_office_df = pd.read_csv(os.path.join(data_dir, 'box_office.csv'))
pixar_df = pd.read_csv(os.path.join(data_dir, 'pixar_films.csv'))

# Ensure 'budget' and 'box_office_worldwide' columns exist in the box office DataFrame
scatter_plot = None
if 'budget' in box_office_df.columns and 'box_office_worldwide' in box_office_df.columns:
     # Calculate ROI
    box_office_df['ROI'] = ((box_office_df['box_office_worldwide'] - box_office_df['budget']) / box_office_df['budget']) * 100
    # Format ROI to one decimal place with a % sign
    box_office_df['ROI'] = box_office_df['ROI'].map('{:.1f}%'.format)
    
    merged_awards_df['release_date'] = pd.to_datetime(merged_awards_df['release_date']).dt.year
    
    # Create a scatter plot using Altair with increased chart size
    scatter_plot = alt.Chart(box_office_df).mark_circle(size=400).encode(
    x=alt.X('budget', title=None),  # Remove title for x-axis
    y=alt.Y('box_office_worldwide', title=None),  # Remove title for y-axis
    tooltip=[
            'film',
            alt.Tooltip('release_date', 
                        type='nominal', title='Release Date'),
            alt.Tooltip('budget', 
                       type='quantitative', 
                       format='$,.0f', 
                       title='Budget'),
            alt.Tooltip('box_office_worldwide', 
                       type='quantitative', 
                       format='$,.0f', 
                       title='Worldwide Box Office'),'ROI'
        ]
    ).properties(
        title='',
        width=500,
        height=440
    )

# Create two columns for side-by-side display
column2.markdown(
    "<span style='font-size:20px; font-weight:bold;'>"
    "What is the relationship between a films budget and worldwide box office success? <br>"
    "</br>"
    "</span>",
    unsafe_allow_html=True
)

with col2:
    #st.write("### Films thats Won or Nominated for a award:")
    st.dataframe(display_df.style.apply(highlight_won, subset=['status']), height=550)

with column2:
    #st.write("### Budget vs Total Revenue Worldwide:")
    if scatter_plot:
        st.altair_chart(scatter_plot, use_container_width=True)



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

            bars = alt.Chart(people_chart_data).mark_bar().encode(
                x=alt.X('box_office_worldwide', title=None, axis=None),
                y=alt.Y('name', sort='-x', title=y_axis_title),
                tooltip=[
                    alt.Tooltip('name', title=y_axis_title),
                    alt.Tooltip('box_office_worldwide', format='$,.0f', title='Total Worldwide Box Office'),
                    'films'
                ]
            )

            text = bars.mark_text(
                align='left',
                baseline='middle',
                dx=5,
                fontSize=10,
            ).encode(
                text=alt.Text('box_office_worldwide', format='$.2s'),
                color=alt.value('black')
            )

            people_chart = alt.layer(bars, text).configure_view(
                strokeWidth=0
            ).properties(
                width=500,
                height=400
            )
        elif selected_role:
            st.info(f"No revenue data to display for {selected_role}.")

    if people_chart:
        st.altair_chart(people_chart, use_container_width=True)
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
        # --- Create Genre Chart ---
        bars_genre = alt.Chart(genre_box_office_sum_df).mark_bar().encode(
            x=alt.X('total_box_office_worldwide', title=None, axis=None),
            y=alt.Y('genre', sort='-x', title=''),
            tooltip=[
                'genre',
                alt.Tooltip('total_box_office_worldwide', format='$,.0f', title='Total Worldwide Box Office')
            ]
        )

        text_genre = bars_genre.mark_text(
            align='left',
            baseline='middle',
            dx=5,
            fontSize=10,
        ).encode(
            text=alt.Text('total_box_office_worldwide', format='$.2s'),
            color=alt.value('black')
        )

        genre_chart = alt.layer(bars_genre, text_genre).configure_view(
            strokeWidth=0
        ).properties(
            #height=alt.Step(18) # Adjust step for bar height
            width=500,
            height=440
        )

    if genre_chart:
        st.altair_chart(genre_chart, use_container_width=True)
    else:
        st.info("No subgenre box office data available to display.")


#########################################################
#########################################################
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

    # Define color scale for cinema_score
    color_scale = alt.Scale(domain=['A+', 'A', 'A-', 'N/A'], 
                          range=['#3399FF', 'lightblue', '#FFAB5B', '#A5BFCC'])

# Create the base bar chart
    bars = alt.Chart(sorted_df).mark_bar().encode(
        x=alt.X(sort_column, 
                title=None,
                axis=alt.Axis(labels=False, ticks=False, domain=False)),
        y=alt.Y('film', sort='-x'),
        color=alt.Color('cinema_score', scale=color_scale)
    )

    # Create text labels for the values
    text = alt.Chart(sorted_df).mark_text(
        align='left',
        baseline='middle',
        dx=250,  # Nudge text to right so it doesn't overlap the bar
        fontSize=13
    ).encode(
        x=alt.X(sort_column),
        y=alt.Y('film', sort='-x', title=None),
        text=alt.Text(sort_column),  # Format to 1 decimal place
        color=alt.value('black')  # Set text color to black
    )

    # Combine bars and text
    chart = alt.layer(bars, text).encode(
        tooltip=[
            'imdb_score',
            'cinema_score',
            'metacritic_score',
            'rotten_tomatoes_score'
        ]
    ).properties(
    ).configure_mark(
        tooltip=alt.TooltipContent('encoding')
    )

    # Display the chart in Streamlit
    col1.altair_chart(chart, use_container_width=True)


def main():
    filtered_df = display_sidebar_filters()
if __name__ == "__main__":
    main()