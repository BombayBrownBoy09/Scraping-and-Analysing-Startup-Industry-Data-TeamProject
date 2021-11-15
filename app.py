#Load libraries
import tldextract
import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np
import wget
import base64
import shap; shap.initjs()
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
import datetime
from pytrends.request import TrendReq # Trends API
import urllib.request
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
import streamlit.components.v1 as components
from sklearn.metrics import f1_score
import lightgbm as lgb
from scipy.misc import derivative

warnings.filterwarnings("ignore")

#Set title and favicon
st.set_page_config(page_title='Startup Discovery Tool', page_icon='https://duke.edu/_themes/duke/img/favicon.ico')

###################### CSS Styling ############################################################################################################
#Hide rainbow bar
hide_decoration_bar_style = '''
    <style>
        header {visibility: hidden;}
    </style>
'''
st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)

#Hide hamburger menu & footer
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

#Fonts
# H1
st.markdown(
  """
  <style>
    h1 {
    font-family: Avenir,Helvetica Neue,sans-serif;
    }
    h2 {
    font-family: Avenir,Helvetica Neue,sans-serif;
    }
    h3 {
    font-family: Avenir,Helvetica Neue,sans-serif;
    }
    h4 {
    font-family: Avenir,Helvetica Neue,sans-serif;
    }
    body {
    font-family: Avenir,Helvetica Neue,sans-serif;
    }
  </style>""",
  unsafe_allow_html=True
)
###################### CSS Styling ############################################################################################################


#Create initial titles/subtitles
st.markdown('<h1 style="font-family:Avenir,Helvetica Neue,sans-serif;"> Startup Discovery Tool </h1>', unsafe_allow_html=True)
st.text("")
st.markdown('<p style="font-family:Avenir,Helvetica Neue,sans-serif;"> The following application provides insights into the specific impact different attributes of a startup contribute to either be acquired, IPO, close, or continue to operate. </p>', unsafe_allow_html=True)
st.markdown('<p style="font-family:Avenir,Helvetica Neue,sans-serif;"> To start, please select a category/industry of interest: </p>', unsafe_allow_html=True)


@st.cache(persist=True) #We use this to cache the info and not load the data every time we scroll up/down
def load_data(nrows):
    """
    Function to load the primary data:
    :param nrows: Number of rows to load
    """
    URL_cb = 'https://raw.githubusercontent.com/realonbebeto/Startup-App/main/recom_data/main_data.csv'
    clean_cb = pd.read_csv(URL_cb, parse_dates=['funding_rounds', 'founded_at', 'first_funding_at','last_funding_at'])
    return clean_cb

#Load 10,000 rows of data & make copies to use with different sections
clean_cb = load_data(100000)

#[0, 1, 2, 3] = ['acquired', 'closed', 'ipo', 'operating']
# We'll only look at companies that had their first funding round in the year 2000 and beyond
date_filter = datetime.datetime(2004,1,1)
clean_cb = clean_cb[clean_cb.first_funding_at >= date_filter]

# Select only companies in the US
clean_cb = clean_cb[clean_cb.country_code == 'USA']

# Extract the year of first funding
clean_cb['first_funding_year'] = pd.to_datetime(clean_cb.first_funding_at.dt.year, format='%Y')

# Drop all columns with no founding year
clean_cb.dropna(subset=['founded_at'], inplace=True)


# Scrape Econ Indicators
@st.cache(persist=True) 
def scrape_econ_data():
    """
    Function to scrape economic indicators (GDP) for a given time period
    """
    URL = 'https://www.macrotrends.net/countries/USA/united-states/gdp-growth-rate'
    header = { 'User-Agent' : 'Mozilla/5.0 (Windows NT 6.1; Win64; x64)' }

    # Define request
    req = urllib.request.Request(URL, headers=header)
    response = urllib.request.urlopen(req)
    soup = BeautifulSoup(response, 'lxml')
    prettyHTML = soup.prettify() 
    td = list(soup.findAll('td'))

    # Extract relevant elements
    gdp_year = [string if ('<td style="text-align:center">' in str(string)) | ('<td style="text-align:center;">' in str(string)) else "" for string in td]
    gdp_year = [i for i in gdp_year if i]

    # Clean the list
    gdp_year_clean = []
    for s in gdp_year:
        s = str(s)
        start = s.find('>')+1
        end = s.find('<', start)
        s = s[start:end]
        gdp_year_clean.append(s)

    # Create pairs of data points and extract year
    years = gdp_year_clean[::3].copy()
    gdp_year_clean = [string if '%' in str(string) else "" for string in gdp_year_clean]
    gdp_year_clean = [i for i in gdp_year_clean if i]       
    gdp_growth = gdp_year_clean[::2].copy()
    annual_change = gdp_year_clean[1::2].copy()

    # Create a dataframe with the data
    gdp_df = pd.DataFrame({"Year":years, "GDP_Growth":gdp_growth, "GDP_Annual_Change":annual_change})
    gdp_df['Year'] = pd.to_datetime(gdp_df.Year,format='%Y')
    gdp_df['GDP_Growth'] = gdp_df.GDP_Growth.str.replace("%","").astype('float')
    gdp_df['GDP_Annual_Change'] = gdp_df.GDP_Annual_Change.str.replace("%","").astype('float')
    return gdp_df

# Get econ data
gdp_df = scrape_econ_data()


# Join data
# Merge clean CB data with GDP data
@st.cache(persist=True) 
def join_primary_econ(primary_df, gdp_df):
    """
    Returns joined datasets (primary data + Econ (GDP) data)
    :primary_df: Dataframe with the primary data
    :gdp_df: Dataframe with the Econ data (GDP)
    """
    df_clean_cb_gdp = primary_df.merge(gdp_df, how='left', left_on='first_funding_year', right_on='Year').copy()
    df_clean_cb_gdp.drop(['Year'], axis=1, inplace=True)
    return df_clean_cb_gdp

df_clean_cb_gdp = join_primary_econ(clean_cb, gdp_df).copy()


# Get Google Trends Data
@st.cache(persist=True) 
def get_trends_data ():
    """
    Get Relative Search Interest Data for Startups
    """
    # Connect to API
    pytrends = TrendReq(hl='en-US', tz=360) 

    # Build Payload
    kw_list = ["startups"] # List of keywords to get data 
    pytrends.build_payload(kw_list, geo='US', timeframe='2004-01-01 2014-12-31') 

    # Get interest over Time
    rsi_data = pytrends.interest_over_time() 
    rsi_data = rsi_data.reset_index() 
    rsi_data = rsi_data.rename(columns={"startups": "Search Interest for Startups"})
    return rsi_data

# Get Google Trends Data
rsi_data = get_trends_data()

# Round down date to the first day of the month
df_clean_cb_gdp['first_funding_at_round'] = df_clean_cb_gdp.first_funding_at.to_numpy().astype('datetime64[M]')


# Joing Primary and RSI Data
@st.cache(persist=True) 
def join_primary_rsi(primary_df, rsi_data):
    """
    Returns joined datasets (primary data + Google Trends Data (RSI))
    :primary_df: Dataframe with the primary data
    :rsi_data: Dataframe Google Trends RSI Data
    """
    df_clean_cb_gdp_rsi = primary_df.merge(rsi_data, how='left', left_on='first_funding_at_round', right_on='date')
    df_clean_cb_gdp_rsi['Search Interest for Startups'].fillna(0, inplace=True) # If there's no data it means RSI was 0 at that time, therefore, we use 0 to impute NaNs
    df_clean_cb_gdp_rsi.drop(['date','isPartial'], axis=1, inplace=True)
    return df_clean_cb_gdp_rsi

df_clean_cb_gdp_rsi = join_primary_rsi(df_clean_cb_gdp, rsi_data).copy()


# Processing & Feature Engineering
@st.cache(persist=True) 
def feature_eng_proc(df):
    """
    Creates specific features and removes irrelevant variables
    :param df: Dataframe to process
    """
    # Calculate time to first funding
    df['time_to_funding'] = (df.first_funding_at - df.founded_at).dt.days.astype('int')

    # Drop duplicate companies
    df.normalized_name.nunique()
    df.drop_duplicates(subset=['normalized_name'], inplace=True)

    # Convert founding rounds to int
    df.funding_rounds = df.funding_rounds.astype('int')

    # Drop irrelevant columns (dates have been used for other representations, so they are no longer necessary)
    # Also, because our focus is only for US based companies, we don't need the continents or country code
    df.drop(['Unnamed: 0','index', 'id', 'normalized_name','founded_at','closed_at','tag_list',
                            'first_milestone_at','last_milestone_at','first_funding_year','first_funding_at_round',
                            'first_funding_at','last_funding_at','avg_vote','closed','AS','EU','NA','SA','UT', 
                            'BZ','HE','LE','OT','PC','TR','country_code'], axis=1, inplace=True)
    return df

df_clean_cb_gdp_rsi = feature_eng_proc(df_clean_cb_gdp_rsi).copy()

# Filter by Category
category = st.selectbox("", ['health', 'tech'], key='cluster_box_shap', index=0) #Add a dropdown element to select the category

# Categories
if category == 'health':
    category = ['health', 'medical', 'biotech']
else:
    category = ['web','mobile','software','ecommerce']

# Filter dataframe based on the selected category
df_train = df_clean_cb_gdp_rsi[df_clean_cb_gdp_rsi.category_code.isin(category)].copy()

# Create features & target and split dataset
X = df_train.drop(['status','category_code'], axis=1).copy()
y = df_train['status'].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Conver to lgb datasets
lgb_train = lgb.Dataset(X_train, y_train)
lgb_test = lgb.Dataset(X_test, y_test)

# Custom Loss Function (Multiclass F1-Score)
def evalerror(preds, df_train):
    """
    Custom multiclass f1-score
    :param preds: Predictions for each observation
    :param df_train: Training dataset
    """
    labels = df_train.get_label()
    preds = preds.reshape(4, -1).T
    preds = preds.argmax(axis = 1)
    f_score = f1_score(labels , preds,  average = 'weighted')
    return 'f1_score', f_score, True


# MODELING
# Define Params
params = {'boosting_type': 'gbdt',
          'objective': 'multiclass',
          'metric': 'multi_logloss',
          'num_class':4,
          'min_data_in_leaf':300,
          'feature_fraction':0.8,
          'bagging_fraction':0.8,
          'bagging_freq':5,
          'max_depth':8,
          'num_leaves':70,
          'learning_rate':0.04}

# Train model
gbm = lgb.train(params, 
                lgb_train,
                feval=evalerror,
                num_boost_round=500,
                valid_sets=[lgb_train, lgb_test],
                early_stopping_rounds=10)


# Explain the model's predictions with SHAP
#[0, 1, 2, 3] = ['acquired', 'closed', 'ipo', 'operating']
# Relationships = "Representation of the people involved in the team for that startup"
st.markdown('<h2 style="font-family:Avenir,Helvetica Neue,sans-serif;"> Startup Attribute’s Impact </h2>', unsafe_allow_html=True)
st.markdown('<p style="font-family:Avenir,Helvetica Neue,sans-serif;"> On this particular industry, according to our data these are some of the attributes of startup that seem to have a greater impact on a given company status. </p>', unsafe_allow_html=True)

explainer = shap.TreeExplainer(gbm)
shap_values = explainer.shap_values(X)


shap.summary_plot(shap_values, X, class_names=["Acquired", "Closed", "IPO'd", "Operating"], max_display=5)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot(bbox_inches='tight')
plt.clf()