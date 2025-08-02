import streamlit as st
import pandas as pd
import numpy as np
import requests
from pathlib import Path
import joblib
import plotly.express as px
import plotly.graph_objects as go
from geopy.geocoders import Nominatim
from geopy.distance import great_circle
from geopy.distance import geodesic
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression        # Only if you refit or simulate model locally
from sklearn.preprocessing import StandardScaler         # If you're scaling any new inputs
from collections import Counter
import time
import json
import ast

# --- Page Configuration ---
# This must be the first Streamlit command in your script
st.set_page_config(
    page_title="Gurgaon Flat Finder",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Theme Toggle ---
# We use session state to keep track of the theme
if 'theme' not in st.session_state:
    st.session_state.theme = "dark"



def load_css():
    # Base styles
    base_css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        body { font-family: 'Inter', sans-serif; }
        [data-testid="stSidebar"] { padding-top: 2rem; }
        .stButton>button {
            border-radius: 0.5rem; padding: 0.75rem 1.5rem; font-weight: bold;
            transition: all 0.2s ease-in-out;
            background: linear-gradient(to right, #6366f1, #8b5cf6);
            color: white !important; border: none;
        }
        .stButton>button:hover {
            transform: scale(1.02); box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        div[role="radiogroup"] { gap: 0.5rem; }
    </style>
    """

    # Dark theme styles
    dark_theme = """
    <style>
        .stApp { background-color: #0f172a; }
        [data-testid="stSidebar"] { background-color: #1e293b; border-right: 1px solid #334155; }
        h1, h2, h3 { color: #f1f5f9; }
        .stMarkdown, p, label { color: #94a3b8; }
        [data-testid="stMetricValue"] { color: #f1f5f9; }
        .stSelectbox div[data-baseweb="select"] > div { background-color: #334155; }
        div[role="radiogroup"] label p { color: #cbd5e1 !important; font-weight: 500; }

        /* --- THIS IS THE FIX --- */
        /* Targets the text inside the selectbox in dark mode */
        [data-testid="stSelectbox"] div[data-baseweb="select"] > div > div {
            color: #f1f5f9 !important;
        }
    </style>
    """

    # Light theme styles
    light_theme = """
    <style>
        .stApp { background-color: #f1f5f9; color: #334155; }
        [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e2e8f0; }
        h1, h2, h3 { color: #1e293b; }
        [data-testid="stSidebar"] h1 { color: #1e293b; }
        .stMarkdown, p, label { color: #475569; }
        [data-testid="stMetricValue"] { color: #1e293b !important; }
        .stSelectbox div[data-baseweb="select"] > div { background-color: #e2e8f0; }
        div[role="radiogroup"] label p { color: #334155 !important; font-weight: 500; }

        /* --- THIS IS THE FIX --- */
        /* Targets the text inside the selectbox in light mode */
        [data-testid="stSelectbox"] div[data-baseweb="select"] > div > div {
            color: #1e293b !important;
        }
    </style>
    """

    # Apply styles
    st.markdown(base_css, unsafe_allow_html=True)
    if st.session_state.get('theme', 'dark') == "dark":
        st.markdown(dark_theme, unsafe_allow_html=True)
    else:
        st.markdown(light_theme, unsafe_allow_html=True)




load_css()

# --- Load Data and Model ---
# @st.cache_data
# def load_data():
#     try:
#         df = pd.read_parquet(r'C:\Users\aryan\Desktop\Capstone Project\Data Preprocessing New\gurgaon_properties_final_df.parquet')
#         # df = df[df['Furnishing'] != 'Under Construction']
#     except FileNotFoundError:
#         st.error(r"Error: Data file ('C:\Users\aryan\Desktop\Capstone Project\Data Preprocessing New\gurgaon_properties_final_df.parquet') not found.")
#         st.stop()
#     try:
#         with open('map.geojson', 'r') as f:
#             geojson_data = json.load(f)
#     except FileNotFoundError:
#         st.error("Error: GeoJSON file ('map.geojson') not found.")
#         st.stop()
#     return df, geojson_data


# --- Configuration for the remote Parquet file ---
PARQUET_URL = "https://github.com/iamaryan07/Capstone-Project-Real-Estate/releases/download/v1.0/gurgaon_properties_final_df.parquet"
PARQUET_PATH = Path("gurgaon_properties_final_df.parquet")

@st.cache_data
def load_data():
    # --- Part 1: Download and Load the Parquet File ---
    if not PARQUET_PATH.exists():
        with st.spinner("Downloading dataset... this may take a moment."):
            try:
                r = requests.get(PARQUET_URL)
                r.raise_for_status() # This will raise an error for bad status codes
                with open(PARQUET_PATH, 'wb') as f:
                    f.write(r.content)
            except requests.exceptions.RequestException as e:
                st.error(f"Error downloading data file: {e}")
                st.stop()

    try:
        df = pd.read_parquet(PARQUET_PATH)
    except Exception as e:
        st.error(f"Error loading Parquet file: {e}")
        st.stop()

    # --- Part 2: Load the Local GeoJSON File ---
    try:
        # Build a full, reliable path to the geojson file
        script_dir = Path(__file__).parent
        geojson_path = script_dir / "map.geojson"

        with open(geojson_path, 'r') as f:
            geojson_data = json.load(f)
    except FileNotFoundError:
        st.error("Error: GeoJSON file ('map.geojson') not found. Make sure it's in your GitHub repository.")
        st.stop()

    return df, geojson_data


MODEL_PATH = Path("property_price_model.pkl")
MODEL_URL = "https://github.com/iamaryan07/Capstone-Project-Real-Estate/releases/download/v1.0/property_price_model.pkl"

@st.cache_resource
def load_model():
    """
    Downloads the model from a URL if it doesn't exist,
    then loads and returns the model.
    """
    # Download the model if it's not already here
    if not MODEL_PATH.exists():
        with st.spinner("Downloading model... This may take a moment."):
            try:
                r = requests.get(MODEL_URL)
                r.raise_for_status()  # Raise an exception for bad status codes
                with open(MODEL_PATH, 'wb') as f:
                    f.write(r.content)
                st.success("Model downloaded!")
            except requests.exceptions.RequestException as e:
                st.error(f"Error downloading model: {e}")
                return None
    
    try:
        model = joblib.load(r'C:\Users\aryan\Desktop\Capstone Project\Joblib\property_price_model.pkl')
    except FileNotFoundError:
        st.error("Error: Saved model file ('property_price_model.pkl') not found.")
        model = None
    return model

df, geojson_data = load_data()
model = load_model()

if model:
    st.success("Model loaded successfully!")
    # ... rest of your app code ...
else:
    st.error("Model could not be loaded. App cannot proceed.")

# --- Sidebar ---
st.sidebar.title("Gurgaon Flat Finder")


if st.sidebar.button("Toggle Theme", use_container_width=True):
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
    st.rerun()

st.sidebar.markdown("---")

# Custom nav with improved styles
with st.sidebar:
    st.markdown('<div class="nav-container">', unsafe_allow_html=True)
    page = st.sidebar.radio(
    "Navigation", 
    ("Home", "Price Prediction", "Analytics Dashboard", "Recommend Society", "Insights"),
    key="sidebar-nav"
    )

    st.markdown("</div>", unsafe_allow_html=True)



# --- NEW: Home Page ---
def home_page():
    st.title("Gurgaon Flat Finder")
    st.subheader("Price Prediction & Analytics")
    st.markdown("---")

    # Hero Section with Image and Metrics
    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        st.image('https://images.unsplash.com/photo-1580587771525-78b9dba3b914?q=80&w=1974&auto=format&fit=crop', use_container_width=True)
    with col2:
        st.markdown("### Welcome to the Future of Real Estate Analysis.")
        st.markdown("Gurgaon Flat Finder is a comprehensive platform that leverages machine learning to demystify the Gurgaon property market. Whether you're buying, selling, or researching, our tools provide the clarity you need.")
        st.markdown("---")
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Properties Analyzed", f"{df.shape[0]:,}")
        mcol2.metric("Sectors Covered", df['Sector'].nunique())
        mcol3.metric("Avg. Price / sq.ft.", f"₹ {int((df['Price']*10000000 / df['Built Up Area']).mean()):,}")
    
    st.markdown("---")

    # How It Works Section
    st.header("How It Works")
    step1, step2, step3 = st.columns(3)
    with step1:
        st.markdown("### 1. Advanced Statistical Modeling")
        st.markdown("Enter property details into our intuitive form to get a valuation from our fine-tuned regression model.")
    with step2:
        st.markdown("### 2. Interactive Analytics")
        st.markdown("Explore the Gurgaon market with interactive maps and charts. Analyze trends, compare sectors, and understand what drives property values.")
    with step3:
        st.markdown("### 3. Smart Recommendations")
        st.markdown("Find your next property with our intelligent recommenders. Search by location and radius or find similar properties to one you already love.")

    st.markdown("---")

    # Key Features Section
    st.header("Key Features")
    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        st.markdown("""
        <div class="feature-card">
            <h3>Price Prediction</h3>
            <p>Our core feature. Get a highly accurate price prediction for any property in Gurgaon based on our fine-tuned Extra Trees model.</p>
        </div>
        """, unsafe_allow_html=True)
    with fcol2:
        st.markdown("""
        <div class="feature-card">
            <h3>Analytics Dashboard</h3>
            <p>Go beyond predictions. Our interactive dashboard with maps and charts allows you to become an expert on the Gurgaon property market.</p>
        </div>
        """, unsafe_allow_html=True)
    with fcol3:
        st.markdown("""
        <div class="feature-card">
            <h3>What-If Insights</h3>
            <p>Simulate how changes to a property's features—like adding a bedroom or improving the furnishing—will impact its estimated market value.</p>
        </div>
        """, unsafe_allow_html=True)



# --- Page 2: Price Prediction ---
def prediction_page():
    st.title("Predict Property Price")
    st.markdown("Fill in the property details below to get an estimated price from our Extra Trees model.")
    
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            unique_sectors = sorted(df['Sector'].unique().tolist())
            numerically_sorted_sectors = sorted(
                unique_sectors,
                key=lambda x: int(x.replace('Sector ', '')) if 'Sector' in x else 999
            )
            sector = st.selectbox('Sector', numerically_sorted_sectors)
            
            bedroom = st.selectbox('Bedrooms', sorted(df['Bedroom'].unique()))
            bathroom = st.selectbox('Bathrooms', sorted(df['Bathroom'].unique()))
            balcony = st.selectbox('Balconies', sorted(df['Balcony'].unique()))
            property_age = st.selectbox('Property Age', ['0 to 1 Year Old', '1 to 5 Year Old', '5 to 10 Year Old', '10+ Year Old'])
            furnishing = st.selectbox('Furnishing', ['Unfurnished', 'Semi Furnished', 'Furnished'])
            power_backup = st.selectbox('Power Backup', ['None', 'Partial', 'Full'])
            
        with col2:
            built_up_area = st.slider('Built Up Area (sq. ft.)', int(df['Built Up Area'].min()), max_value= 3000, value= 1500, step= 50)
            floor_num = st.slider('Floor Number', min_value= 0, max_value= int(df['Total Floor'].max()), value=5)
            total_floor = st.slider('Total Floors in Building', int(df['Total Floor'].min()), int(df['Total Floor'].max()), 10)
            covered_parking = st.slider('Covered Parking Spots', int(df['Covered_Parking'].min()), int(df['Covered_Parking'].max()), 1)
            open_parking = st.slider('Open Parking Spots', int(df['Open_Parking'].min()), int(df['Open_Parking'].max()), 1)
            rating = st.slider('Property Rating', min_value=3.0, max_value=5.0, value=3.5, step=0.1)

            
        st.markdown("---")
        col3, col4, col5 = st.columns(3)
        with col3:
             nearby = st.selectbox('Nearest Amenity Type', sorted(df['Nearby'].unique()))
        with col4:
             overlooking = st.selectbox('View (Overlooking)', sorted(df['Overlooking'].unique()))
        with col5:
            st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True) # Spacer
            servant_room = st.checkbox('Servant Room')
            store_room = st.checkbox('Store Room')
            study_room = st.checkbox('Study Room')

    if st.button('Predict Price', use_container_width=True, type="primary"):
        if model is None:
            st.error("Model not loaded. Cannot make a prediction.")
        else:
            with st.spinner('Calculating...'):
                total_parking = covered_parking + open_parking
                input_data = {
                    'Sector': [sector], 
                    'Built Up Area': [int(built_up_area)], 
                    'Bedroom': [int(bedroom)],
                    'Bathroom': [int(bathroom)], 
                    'Balcony': [int(balcony)], 
                    'Servant Room': [1 if servant_room else 0],
                    'Store Room': [1 if store_room else 0], 
                    'Study Room': [1 if study_room else 0],
                    'Floor Num': [int(floor_num)], 
                    'Total Floor': [int(total_floor)], 
                    'Property Age': [property_age],
                    'Furnishing': [furnishing], 
                    'Power Backup': [power_backup],
                    'Covered_Parking': [int(covered_parking)], 
                    'Open_Parking': [int(open_parking)],
                    'Total Parking': [total_parking], 
                    'Rating': [float(rating)],
                    'Nearby': [nearby], 
                    'Overlooking': [overlooking]
                }
                input_df = pd.DataFrame(input_data)
                prediction = model.predict(input_df)
                predicted_price = prediction[0]
                
                st.success(f"The estimated price of the property is **₹ {predicted_price:.2f} Crores**")

# --- Page 3: Analytics Dashboard ---
def analytics_page():
    df['Rating'] = df['Rating'].astype('float64')
    df['Built Up Area'] = df['Built Up Area'].astype('float64')
    df['Price'] = df['Price'].astype('float64')
    
    st.title("Analytics Dashboard")
    st.markdown("Visual insights into the Gurgaon property market.")

    sector_avg_price = df.groupby('Sector')['Price'].mean().reset_index()

    st.subheader("Gurgaon Property Price Map")
    
    # --- Step 1: Create the map using the NEW function name ---
    fig = px.choropleth_map(
        sector_avg_price,
        geojson=geojson_data,
        locations='Sector',
        featureidkey='properties.name',
        color='Price',
        color_continuous_scale="YlOrRd",
        map_style="carto-darkmatter",
        zoom=10,
        center={"lat": 28.4595, "lon": 77.0266},
        opacity=0.7,
        labels={'Price': 'Average Price (Cr)'}
    )

    # --- Step 2: Update the hover template ---
    fig.update_traces(
        hovertemplate='<b>%{location}</b><br><br>Average Price: %{z:.1f} Cr<extra></extra>'
    )

    # --- Step 3: Final layout updates ---
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)


# Viz 5
    st.markdown(" ")
    st.markdown("---")  
    st.markdown(" ")  
    
    # --- Sector Comparison Radar Chart ---
    st.subheader("Sector Comparison Radar Chart")
    st.markdown("Select up to 5 sectors to compare their characteristics against the rest of Gurgaon.")

    # 1. Calculate overall stats for all sectors (for percentile calculation)
    @st.cache_data
    def calculate_sector_stats(df):
        # --- NEW: Pre-process data for analysis ---
        df_processed = df.copy()
        
        # Create 'Extra Rooms' feature
        df_processed['Extra Rooms'] = df_processed['Study Room'] + df_processed['Servant Room'] + df_processed['Store Room']
        
        # Create numerical mappings for ordinal features
        age_map = {'10+ Year Old': 0, '5 to 10 Year Old': 1, '1 to 5 Year Old': 2, '0 to 1 Year Old': 3}
        furnish_map = {'Unfurnished': 0, 'Semi Furnished': 1, 'Furnished': 2}
        power_map = {'None': 0, 'Partial': 1, 'Full': 2}
        
        df_processed['Property Age Num'] = df_processed['Property Age'].map(age_map)
        df_processed['Furnishing Num'] = df_processed['Furnishing'].map(furnish_map)
        df_processed['Power Backup Num'] = df_processed['Power Backup'].map(power_map)

        # Calculate mean for all relevant metrics
        stats = df_processed.groupby('Sector').agg({
            'Price': 'mean',
            'Built Up Area': 'mean',
            'Rating': 'mean',
            'Total Parking': 'mean',
            'Extra Rooms': 'mean',
            'Property Age Num': 'mean',
            'Furnishing Num': 'mean',
            'Power Backup Num': 'mean'
        }).reset_index()
        return stats

    sector_stats = calculate_sector_stats(df)

    # 2. User input: Multi-select box for sectors
    numerically_sorted_sectors = sorted(
        df['Sector'].unique().tolist(),
        key=lambda x: int(x.replace('Sector ', '')) if 'Sector' in x else 999
    )
    selected_sectors = st.multiselect(
        'Select sectors to compare (up to 5)',
        numerically_sorted_sectors,
        default=['Sector 65', 'Sector 102']
    )

    # 3. Logic to handle user selection
    if len(selected_sectors) > 0 and len(selected_sectors) <= 5:
        comparison_df = sector_stats[sector_stats['Sector'].isin(selected_sectors)].copy()
        
        # --- UPDATED: New list of metrics ---
        metrics = ['Price', 'Built Up Area', 'Rating', 'Total Parking', 'Extra Rooms', 'Property Age Num', 'Furnishing Num', 'Power Backup Num']
        
        percentile_data = {}
        for metric in metrics:
            comparison_df[f'{metric} Pctl'] = comparison_df[metric].apply(
                lambda x: (sector_stats[metric] < x).mean() * 100
            )
            percentile_data[metric] = comparison_df[f'{metric} Pctl'].tolist()

        # 4. Display the data table
        st.write("### Percentile Rankings (0-100)")
        st.dataframe(comparison_df[['Sector'] + [f'{m} Pctl' for m in metrics]].rename(columns={
            'Price Pctl': 'Price',
            'Built Up Area Pctl': 'Area',
            'Rating Pctl': 'Rating',
            'Total Parking Pctl': 'Parking',
            'Extra Rooms Pctl': 'Extra Rooms',
            'Property Age Num Pctl': 'Modernity',
            'Furnishing Num Pctl': 'Furnishing Level',
            'Power Backup Num Pctl': 'Power Backup'
        }).set_index('Sector').round(1))

        # 5. Create the Radar Chart
        fig_radar = go.Figure()
        colors = px.colors.qualitative.Vivid
        
        # --- UPDATED: New labels for the chart axes ---
        radar_metrics = ['Price', 'Area', 'Rating', 'Parking', 'Extra Rooms', 'Modernity', 'Furnishing Level', 'Power Backup']
        
        for i, sector in enumerate(selected_sectors):
            sector_data = comparison_df[comparison_df['Sector'] == sector]
            percentiles = [sector_data[f'{m} Pctl'].iloc[0] for m in metrics]
            
            fig_radar.add_trace(go.Scatterpolar(
                r=percentiles,
                theta=radar_metrics, # Use the new labels
                fill='toself',
                name=sector,
                line_color=colors[i % len(colors)],
                hovertemplate=f'<b>{sector}</b><br>%{{theta}}: %{{r:.1f}}th Percentile<extra></extra>'
            ))

        fig_radar.update_layout(
            height= 600,
            width= 800,
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, ticks=''),
                angularaxis=dict(tickfont=dict(size=12))
            ),
            showlegend=True,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#94a3b8'
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    elif len(selected_sectors) > 5:
        st.warning("Please select a maximum of 5 sectors.")
    else:
        st.info("Select one or more sectors to see the comparison.")


# Viz 2
    st.markdown(" ")
    st.markdown("---")
    st.markdown(" ")
    st.subheader("Price vs. Built Up Area")
    
    unique_sectors = sorted(df['Sector'].unique().tolist())
    numerically_sorted_sectors = sorted(
        unique_sectors,
        key=lambda x: int(x.replace('Sector ', '')) if 'Sector' in x else 999
    )
    
    sector_list = ['Overall Gurgaon'] + numerically_sorted_sectors
    selected_sector = st.selectbox('Select a Sector to Analyze', sector_list)

    # 2. Prepare the data based on the selection
    if selected_sector == 'Overall Gurgaon':
        data_to_plot = df
        title_text = 'Price vs. Area Relationship (Overall Gurgaon)'
    else:
        data_to_plot = df[df['Sector'] == selected_sector]
        title_text = f'Price vs. Area Relationship ({selected_sector})'
    
    # 3. Create and display the plot
    # This separation ensures the plot is always created with the correct, fully prepared data
    theme_colors = ["#818cf8", "#f59e0b", "#10b981", "#ec4899"]
    
    scatter_fig = px.scatter(
        data_to_plot, 
        x='Built Up Area', 
        y='Price', 
        color= 'Furnishing',
        title=title_text,
        labels={'Built Up Area': 'Area (sq. ft.)', 'Price': 'Price (Cr)'},
        opacity=0.6, 
        trendline="ols",
        # color_discrete_sequence= theme_colors
        # trendline_color_override="#6366f1"
    )
    st.plotly_chart(scatter_fig, use_container_width=True)
    
    
    # Viz 3
    st.markdown(" ")
    st.markdown("---")
    st.markdown(" ")
    st.subheader("Interactive Bivariate Analysis")
    st.markdown("Select a sector and two features to see how they relate to each other.")
    
    numerical_cols = ['Price', 'Built Up Area', 'Rating']
    categorical_cols = ['Property Age', 'Furnishing', 'Power Backup', 'Nearby', 'Overlooking', 'Servant Room', 'Store Room', 'Study Room', 'Bedroom', 'Bathroom', 'Balcony', 'Floor Num', 'Total Floor', 'Covered_Parking', 'Open_Parking', 'Total Parking']


    # --- Create the user inputs in columns (define them only once) ---
    col1, col2, col3 = st.columns(3)
    with col1:
        # Use your numerically sorted list for the sector dropdown
        sector_list = ['Overall Gurgaon'] + numerically_sorted_sectors
        selected_sector = st.selectbox("Select a Sector", sector_list)
    with col2:
        # Give the selectboxes unique labels
        feature1 = st.selectbox("Select the X-axis Feature", df.drop(columns= 'Sector').columns, index= 18)
    with col3:
        feature2 = st.selectbox("Select the Y-axis Feature", df.drop(columns= 'Sector').columns, index= 9)

    # --- Filter data based on sector selection ---
    if selected_sector == 'Overall Gurgaon':
        data_to_plot = df
    else:
        data_to_plot = df[df['Sector'] == selected_sector]

    # --- Logic to determine the correct plot type ---
    if not data_to_plot.empty:
        if feature1 == feature2:
            st.warning("Please select two different features for the X and Y axes.")
        
        elif feature1 in numerical_cols and feature2 in numerical_cols:
            # Numeric vs. Numeric -> Scatter Plot
            st.write(f"### {feature2} vs. {feature1} in {selected_sector}")
            fig = px.scatter(
                data_to_plot, 
                x=feature1, 
                y=feature2,
                opacity=0.6, 
                trendline="ols",
                trendline_color_override="#6366f1"
            )
            st.plotly_chart(fig, use_container_width=True)

        elif (feature1 in categorical_cols and feature2 in numerical_cols):
            # Categorical (X) vs. Numeric (Y) -> Violin Plot
            st.write(f"### {feature2} Distribution by {feature1} in {selected_sector}")
            fig = px.box(
                data_to_plot, 
                x=feature1, 
                y=feature2,
                color=feature1
            )
            st.plotly_chart(fig, use_container_width=True)

        elif (feature1 in numerical_cols and feature2 in categorical_cols):
            # Numeric (X) vs. Categorical (Y) -> Violin Plot (axes flipped)
            st.write(f"### {feature1} Distribution by {feature2} in {selected_sector}")
            fig = px.violin(
                data_to_plot, 
                x=feature2, 
                y=feature1,
                color=feature2
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else: # Both are categorical
            # Categorical vs. Categorical -> Heatmap of Counts
            st.write(f"### Relationship between {feature1} and {feature2} in {selected_sector}")
            crosstab = pd.crosstab(data_to_plot[feature1], data_to_plot[feature2])
            fig = px.imshow(
                crosstab, 
                text_auto=True, 
                aspect="auto",
                color_continuous_scale="Blues"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"No data available for '{selected_sector}'. Please choose a different sector.")
        
        
# Viz 4
    st.markdown(" ")
    st.markdown("---")
    st.markdown(" ")
    
    # --- NEW: Key Price Drivers & Price Distribution ---
    st.subheader("Key Insights")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Key Price Drivers")
        # In a real app, you would calculate these from your final trained model.
        # For this example, we'll use pre-calculated values based on your earlier analysis.
        feature_importance = pd.DataFrame({
            'Feature': ['Built Up Area', 'Bathroom', 'Total Floor', 'Rating', 'Servant Room', 'Bedroom', 'Property Age', 'Furnishing', 'Floor Num', 'Power Backup'],
            'Importance': [0.35, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02]
        }).sort_values(by='Importance', ascending=True)

        fig_imp = px.bar(feature_importance, x='Importance', y='Feature', orientation='h',
                         title='Top 10 Most Important Features',
                         labels={'Importance': 'Relative Importance', 'Feature': 'Property Feature'})
        fig_imp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_imp, use_container_width=True)

    with col2:
        st.markdown("#### Price Distribution by Feature")
        # Define the list of features the user can analyze
        features_to_compare = ['Bedroom', 'Bathroom', 'Furnishing', 'Power Backup', 'Property Age']
        
        # Create a dropdown for the user to select a feature
        selected_feature = st.selectbox('Compare Price Distribution by Feature', features_to_compare)
        
        # Create the violin plot based on the user's selection
        fig_dist = px.violin(
            df, 
            x=selected_feature, 
            y='Price', 
            title=f'Price Distribution by {selected_feature}',
            labels={'Price': 'Price (in Crores)', selected_feature: selected_feature},
            color=selected_feature,
            color_discrete_sequence=px.colors.qualitative.Pastel,
            box=True # Adds a small box plot inside the violin
        )
        fig_dist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_dist, use_container_width=True)
       
                
  
# --- Page 4: Hybrid Recommendation System ---
def recommendation_page():
    st.title("Property Recommender")
    st.markdown("Find properties near a landmark within a radius.")
    
    st.markdown("""
<style>
    .section-header {
        font-size: 22px;
        font-weight: 600;
        margin-top: 30px;
        color: #f8fafc;
    }
    .section-subtext {
        font-size: 15px;
        color: #cbd5e1;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


    # --- Configuration for the remote CSV file ---
    REC_DATA_URL = "https://github.com/iamaryan07/Capstone-Project-Real-Estate/releases/download/v1.0/data_recommendation_v2.csv"
    REC_DATA_PATH = Path("data_recommendation_v2.csv")

    # --- Reusable download function ---
    def download_file(url, path, message):
        if not path.exists():
            with st.spinner(message):
                try:
                    r = requests.get(url)
                    r.raise_for_status()
                    with open(path, 'wb') as f:
                        f.write(r.content)
                except Exception as e:
                    st.error(f"Error downloading {path.name}: {e}")
                    return False
        return True

    @st.cache_resource
    def setup_recommenders():
        # Load property data
        # Part 1: Download the CSV file from the Release
        if not download_file(REC_DATA_URL, REC_DATA_PATH, "Downloading recommendation data..."):
            st.error("Could not download recommendation data. Recommender system unavailable.")
            st.stop()
    
        df_rec = pd.read_csv(REC_DATA_PATH)
        
        df_rec.drop(columns='Unnamed: 0', inplace=True)
        df_rec.dropna(subset=['NearbyPlaces'], inplace=True)
        df_rec['NearbyPlaces'] = df_rec['NearbyPlaces'].apply(ast.literal_eval)

        # Load both caches
        with open(r"C:\Users\aryan\Desktop\Capstone Project\Joblib\geo_cache.pkl", "rb") as f:
            geo_cache = joblib.load(f)

        with open(r"C:\Users\aryan\Desktop\Capstone Project\Joblib\geo_cache_old.pkl", "rb") as f:
            geo_cache_old = joblib.load(f)

        # Add coordinates to df_rec from geo_cache_old
        df_rec['Geo_Key'] = df_rec['Property Name'].str.strip().str.lower() + ', ' + df_rec['Sector'].str.strip().str.lower() + ', gurgaon'
        df_rec['Coordinates'] = df_rec['Geo_Key'].map(geo_cache_old)

        # Keep only properties with coordinates
        df_rec = df_rec[df_rec['Coordinates'].notnull()].copy()
        
        unique_landmarks = sorted([k.replace(', gurgaon', '').strip('"').strip("'").capitalize() for k in geo_cache.keys() if geo_cache[k] is not None])
        landmark_coords = {k.replace(', gurgaon', '').strip('"').strip("'").capitalize(): v for k, v in geo_cache.items() if v is not None}
        
        # -- 2. Setup for Price/BHK-based Recommender --
        features = ['Bedroom', 'Price', 'Built Up Area']
        scaler = MinMaxScaler()
        df_scaled = df_rec.copy()
        df_scaled[features] = scaler.fit_transform(df_scaled[features])
        
        nn_model = NearestNeighbors(n_neighbors=10, algorithm='ball_tree')
        nn_model.fit(df_scaled[features])

        return df_rec, landmark_coords, nn_model, scaler
    
    df_rec, landmark_coords, nn_model, scaler = setup_recommenders()
    
    rec_type = st.selectbox(
    "Select Recommendation Type",
    ["Location-Based (Landmark)", "Similar Price & Configuration", "Hybrid (Location + Price)"])
    
    def recommend_properties_by_price(property_name):
        try:
            features = ['Bedroom', 'Price', 'Built Up Area']
            society_df = df_rec[df_rec['Property Name'] == property_name]
            analysis_df = society_df.groupby('Bedroom').agg({'Price': 'mean', 'Built Up Area': 'mean'}).reset_index()
            all_recs = pd.DataFrame()
            
            for i, row in analysis_df.iterrows():
                query_df_raw = pd.DataFrame([[row['Bedroom'], row['Price'], row['Built Up Area']]], columns=features)
            
                # 2. Now, pass this DataFrame to the scaler
                query_point_scaled = scaler.transform(query_df_raw)
                
                # 3. Convert the scaled array back to a DataFrame for the NN model
                query_df_scaled = pd.DataFrame(query_point_scaled, columns=features)
                
                distances, indices = nn_model.kneighbors(query_df_scaled)
                all_recs = pd.concat([all_recs, df_rec.iloc[indices[0]]])
            
            unique_recs = all_recs.drop_duplicates(subset=['Property Name'])
            # return unique_recs[unique_recs['Property Name'] != property_name].head(10)
            return unique_recs[unique_recs['Property Name'] != property_name].drop_duplicates(subset='Property Name')
        except:
            return pd.DataFrame()
        
    
    def recommend_properties_by_location(landmark):
        landmark_coord = landmark_coords.get(landmark)
        if not landmark_coord:
            st.warning("Coordinates for the selected landmark are missing.")
            return

        # Calculate distance from landmark
        def compute_distance(row):
            try:
                return geodesic(landmark_coord, row['Coordinates']).km
            except:
                return None

        df_rec['DistanceFromLandmark'] = df_rec.apply(compute_distance, axis=1)
        
        # Filter within radius and sort
        filtered = df_rec[df_rec['DistanceFromLandmark'] <= selected_radius]
        filtered = filtered.sort_values(by='DistanceFromLandmark')
        
        return filtered
    
    
    def recommend_properties_by_location_society(property_name):
        property_coord_df = df_rec[df_rec['Property Name'] == property_name][['Coordinates']]
        if property_coord_df.empty or property_coord_df['Coordinates'].isnull().all():
            st.warning("Coordinates for the selected Society are missing.")
            return pd.DataFrame()

        property_coord = property_coord_df['Coordinates'].iloc[0]
        
        # Calculate distance from landmark
        def compute_distance(row):
            try:
                return geodesic(property_coord, row['Coordinates']).km
            except:
                return None

        df_rec['DistanceFromSociety'] = df_rec.apply(compute_distance, axis=1)
        
        # Filter within radius and sort
        filtered = df_rec
        filtered = filtered[filtered['Property Name'] != property_name]
        filtered = filtered.drop_duplicates(subset='Property Name')
        filtered = filtered.sort_values(by='DistanceFromSociety')
        return filtered
    
    if rec_type == "Location-Based (Landmark)":
        st.markdown('<div class="section-header">Location-Based Recommender</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subtext">Find properties near a landmark within your chosen radius.</div>', unsafe_allow_html=True)
    
        # --- UI ---
        col1, col2 = st.columns([0.7, 0.3])

        with col1:
            selected_landmark = st.selectbox("Select a Landmark", sorted(list(set(landmark_coords.keys()))))
        with col2:
            selected_radius = st.slider("Select Radius (km)", 1, 30, 5)
            

        if st.button("Find Nearby Societies", use_container_width=True):
            with st.spinner("Searching..."):
                filtered = recommend_properties_by_location(selected_landmark)            

                if filtered.empty:
                    st.info("No properties found within the selected radius.")
                else:
                    st.subheader("Top Nearby Societies")
                    st.dataframe(filtered[['Property Name', 'Sector', 'DistanceFromLandmark']].drop_duplicates().head(20))

    elif rec_type == "Similar Price & Configuration": 
        st.markdown('<div class="section-header">Price & Configuration Recommender</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subtext">Get similar societies based on Bedroom, Price, and Built-up Area.</div>', unsafe_allow_html=True)
    
        
        property_by_price = st.selectbox("Select a Society", sorted(df_rec['Property Name'].unique().tolist()))          
        if st.button("Find Similar Societies", use_container_width= True):
            with st.spinner('Searching...'):
                df_price_rec = recommend_properties_by_price(property_by_price) 
                if df_price_rec.empty:
                    st.info("Cannot Find Similar Properties")
                else:
                    top_props = df_price_rec[['Property Name', 'Sector', 'Built Up Area', 'Bedroom', 'Price', 'URL']] \
                    .drop_duplicates(subset='Property Name') \
                    .head(5)

                    st.subheader("Top Similar Societies")

                    for _, row in top_props.iterrows():
                        st.markdown(f"""
                        <div style="padding: 10px; margin-bottom: 10px; border-bottom: 1px solid #444;">
                            <div style="font-size:16px; font-weight:bold; color:#f0f0f0;">{row['Property Name']}, Gurgaon</div>
                            <div style="font-size:14px; color:#ccc;">
                                <b>Sector:</b> {row['Sector']}<br>
                                <b>Bedrooms:</b> {row['Bedroom']} &nbsp;&nbsp;
                                <b>Area:</b> {int(row['Built Up Area']):,} sq.ft<br>
                                <b>Price:</b> ₹{row['Price']} Cr<br>
                                <a href="{row['URL']}" target="_blank" style="color:#1E90FF; text-decoration:none;">Visit Listing</a>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
    elif rec_type == "Hybrid (Location + Price)":
        st.markdown('<div class="section-header">Hybrid Recommender</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subtext">Combine both location and configuration to get smarter recommendations.</div>', unsafe_allow_html=True)
        
        
        col3, col4 = st.columns([0.7, 0.3])
        
        with col3:
            selected_property = st.selectbox("Select the Society", sorted(df_rec['Property Name'].unique().tolist()))
        with col4:
            preference = st.selectbox('Prioritize', ['Location Similarity', 'Price & Configuration'])
            
        if st.button("Get Hybrid Recommendations", use_container_width=True, type="primary"):
    #         return filtered[['Property Name', 'Sector', 'Avg_Nearby_Distance']].drop_duplicates().head(10)
            with st.spinner("Calculating Hybrid Recommendations..."):
                location_recs = recommend_properties_by_location_society(selected_property)
                price_recs = recommend_properties_by_price(selected_property)
                
                if preference == 'Location Similarity':
                    w1, w2 = 0.7, 0.3
                else:
                    w1, w2 = 0.3, 0.7
                    
                location_recs['score'] = (10 - location_recs.reset_index().index) * w1
                price_recs['score'] = (10 - price_recs.reset_index().index) * w2
                
                merged_recs = location_recs.merge(price_recs, on='Property Name', how='outer').fillna(0)
                merged_recs['final_score'] = merged_recs['score_x'] + merged_recs['score_y']
                
                # pre_final_recommendation = merged_recs[merged_recs[]]
                final_recommendations = merged_recs.sort_values(by='final_score', ascending=False).head(5)
                
                # --- Display Results ---
                if not final_recommendations.empty:
                    st.subheader("Top 5 Recommendations:")
                    for i, row in final_recommendations.iterrows():
                        # --- THE FIX IS HERE ---
                        # Change row['PropertyName'] to row['Property Name']
                        # st.success(f"**{row['Property Name']}** (Score: {row['final_score']:.2f})")
                        st.markdown(f"""
        <div style="background-color:#1e293b; padding: 16px; margin-bottom: 12px; border-radius: 10px;
                    border-left: 6px solid #38bdf8;">
            <div style="font-size:18px; font-weight:600; color:#f8fafc;">{row['Property Name']}</div>
            <div style="font-size:14px; color:#cbd5e1;">Score: <span style="font-weight:500;">{row['final_score']:.2f}</span></div>
        </div>
    """, unsafe_allow_html=True)

                else:
                    st.warning("Could not find enough unique recommendations to combine.")
                    


def insights_page(model):
    st.title("What-If Analysis")
    st.markdown("Simulate how a change in a single property feature affects its estimated price.")

    if model is None:
        st.error("The prediction model is not loaded, so insights cannot be generated.")
        st.stop()

    # Create a baseline "typical" property
    base_property_data = {}
    for col in df.drop(columns='Price').columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            base_property_data[col] = df[col].median()
        else:
            base_property_data[col] = df[col].mode()[0]
    
    base_df = pd.DataFrame([base_property_data])

    st.subheader("Simulate a Property Change")

    feature_to_change = st.selectbox(
        "Select a feature to change", 
        df.drop(columns='Price').columns,
        key='feature_select_insights'
    )

    col1, col2 = st.columns(2)
    
    upper_limits = {
        'Rating': 5.0, 'Bedroom': 5, 'Bathroom': 5, 'Balcony': 5,
        'Servant Room': 1, 'Study Room': 1, 'Store Room': 1,
        'Total Parking': 10, 'Covered_Parking': 5, 'Open_Parking': 5, 'Built Up Area': 3000
    }
    
    with col1:
        st.markdown(f"**Original Value of '{feature_to_change}'**")
        if df[feature_to_change].dtype == 'object':
            
            if feature_to_change == 'Sector':
                unique_values = sorted(
                    df[feature_to_change].unique().tolist(),
                    key=lambda x: int(x.replace('Sector ', '')) if 'Sector' in x else 999
                )
            else:
                unique_values = sorted(df[feature_to_change].unique().tolist())
            
            original_value = st.selectbox("Original Value", unique_values, key='original_value_cat')
        else:
            max_val = upper_limits.get(feature_to_change, df[feature_to_change].max())
            
            # --- THE FIX: Conditional logic for data types ---
            if feature_to_change == 'Rating':
                original_value = st.number_input(
                    "Original Value", 
                    value=float(base_df[feature_to_change].iloc[0]),
                    max_value=float(max_val), 
                    step=0.1,
                    key='original_value_num'
                )
            else: # All other numericals are integers
                original_value = st.number_input(
                    "Original Value", 
                    value=int(base_df[feature_to_change].iloc[0]),
                    max_value=int(max_val),
                    step=1,
                    key='original_value_num'
                )

    with col2:
        st.markdown(f"**New Value of '{feature_to_change}'**")
        if df[feature_to_change].dtype == 'object':
            
            if feature_to_change == 'Sector':
                unique_values = sorted(
                    df[feature_to_change].unique().tolist(),
                    key=lambda x: int(x.replace('Sector ', '')) if 'Sector' in x else 999
                )
            else:
                unique_values = sorted(df[feature_to_change].unique().tolist())
            
            new_value = st.selectbox("New Value", unique_values, key='new_value_cat')
        else:
            max_val = upper_limits.get(feature_to_change, df[feature_to_change].max())
            current_original_value = st.session_state.original_value_num # Get value from state

            # --- THE FIX: Conditional logic for data types ---
            if feature_to_change == 'Rating':
                new_value = st.number_input(
                    "New Value", 
                    value=float(current_original_value) + 0.1,
                    max_value=float(max_val),
                    step=0.1,
                    key='new_value_num'
                )
            else: # All other numericals are integers
                new_value = st.number_input(
                    "New Value", 
                    value=int(current_original_value) + 1,
                    max_value=int(max_val),
                    step=1,
                    key='new_value_num'
                )
            
    if st.button("Calculate Price Impact", use_container_width=True, type="primary"):
        with st.spinner("Simulating..."):
            before_df = base_df.copy()
            before_df[feature_to_change] = original_value
            
            after_df = base_df.copy()
            after_df[feature_to_change] = new_value

            price_before = model.predict(before_df)[0]
            price_after = model.predict(after_df)[0]
            price_diff = price_after - price_before

            st.success(f"The estimated price changes from **₹{price_before:.2f} Cr** to **₹{price_after:.2f} Cr**")
            
            st.metric(
                label="Price Impact", 
                value=f"₹ {abs(price_diff * 100):.2f} Lakhs", 
                delta=price_diff,
                delta_color="inverse" 
            )



# --- Main App Logic ---
if page == "Home":
    home_page()
elif page == "Price Prediction":
    prediction_page()
elif page == "Analytics Dashboard":
    analytics_page()
elif page == 'Recommend Society':
    recommendation_page()
elif page == 'Insights':
    insights_page(model)
