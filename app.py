#Load libraries
import tldextract
import plotly.express as px
import plotly.graph_objects as go
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
from scipy import stats
import urllib.request
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
import streamlit.components.v1 as components
from sklearn.metrics import f1_score
import lightgbm as lgb
from scipy.misc import derivative
import pickle


warnings.filterwarnings("ignore")

#Set title and favicon
st.set_page_config(page_title='Startup Industry Insights', page_icon='https://duke.edu/_themes/duke/img/favicon.ico')
st.write('<html lang="es"><html translate="no">', unsafe_allow_html=True) #Avoid translations

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
    title {
    font-family: Avenir,Helvetica Neue,sans-serif;
    }
    header {
    font-family: Avenir,Helvetica Neue,sans-serif;
    }
    subheader {
    font-family: Avenir,Helvetica Neue,sans-serif;
    }
  </style>""",
  unsafe_allow_html=True
)
###################### CSS Styling ############################################################################################################


#Create initial titles/subtitles
st.markdown('<h1 style="font-family:Avenir,Helvetica Neue,sans-serif;"> Startup Industry Insights </h1>', unsafe_allow_html=True)
st.text("")
st.markdown('<p style="font-family:Avenir,Helvetica Neue,sans-serif;"> The following application makes it easy to access aggregate financial information about startups, it is meant to help users understand a given industry under the lens of historical data related to startup attributes (i.e. funding, age, etc), economic indicators (GDP), and the relative interest of startups in general measured by the volume of search engine queries. </p>', unsafe_allow_html=True)

st.write("")
st.markdown('<p style="font-family:Avenir,Helvetica Neue,sans-serif;"> To start, please select a category/industry of interest: </p>', unsafe_allow_html=True)


@st.cache(persist=True) #We use this to cache the info and not load the data every time we scroll up/down
def load_data():
    """
    Function to load the primary data:
    """
    URL_Primary = "https://raw.githubusercontent.com/omartinez182/510-Startup-Project/main/Data/primary_data.csv"
    clean_cb = pd.read_csv(URL_Primary, parse_dates=['funding_rounds', 'founded_at', 'first_funding_at','last_funding_at'])
    return clean_cb

#Load 10,000 rows of data & make copies to use with different sections
clean_cb = load_data().copy()

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
categories = st.selectbox("", ['Health', 'Tech'], key='category_selector', index=0) #Add a dropdown element to select the category

# Categories
if categories == 'Health':
    category = ['health', 'medical', 'biotech']
    # Load SHAP values for the selected category
    URL_SH = "https://github.com/omartinez182/510-Startup-Project/blob/main/Data/SHAP/health_shap.txt?raw=true"
    shap_values = pd.read_pickle(URL_SH)
else:
    category = ['web','mobile','software','ecommerce']
    # Load SHAP values for the selected category
    URL_ST = "https://github.com/omartinez182/510-Startup-Project/blob/main/Data/SHAP/tech_shap.txt?raw=true"
    shap_values = pd.read_pickle(URL_ST)

# Filter dataframe based on the selected category
df_train = df_clean_cb_gdp_rsi[df_clean_cb_gdp_rsi.category_code.isin(category)].copy()

# Descriptive Statistics by Status
st.write("")
st.markdown('<p style="font-family:Avenir,Helvetica Neue,sans-serif;"> Now please select a status of interest, namely the status of the companies. </p>', unsafe_allow_html=True)

status_cats = st.selectbox("", ['Acquired', 'Closed', 'IPO', 'Operating'], key='status_selector', index=2) 
st.write("")

if status_cats == 'Acquired':
  status = 0
elif status_cats == 'Closed':
  status = 1
elif status_cats == 'IPO':
  status = 2
else:
  status = 3

# Top 5 Companies that IPO'd in the selected Industry
@st.cache(persist=True) 
def load_data_permalinks():
    """
    Function to load the permalinks data
    """
    URL_permalinks = "https://raw.githubusercontent.com/omartinez182/510-Startup-Project/main/Data/permalinks.csv"
    ids = pd.read_csv(URL_permalinks)
    return ids

ids = load_data_permalinks().copy()


st.header('Descriptive Statistics')

st.markdown('<p style="font-family:Avenir,Helvetica Neue,sans-serif;"> Key metrics for the selected category: </p>', unsafe_allow_html=True)
st.write("")

clean_cb_ids = clean_cb[clean_cb.category_code.isin(category)].copy()
clean_cb_ids = clean_cb_ids[clean_cb_ids.status == status].sort_values(by="funding_total_usd", ascending=False)
clean_cb_ids = clean_cb_ids.merge(ids, how='left')
clean_cb_ids = clean_cb_ids[['normalized_name','permalink','founded_at','first_funding_at','funding_rounds','funding_total_usd']]
clean_cb_ids.loc[:,'founded_at'] = clean_cb_ids['founded_at'].dt.year
clean_cb_ids.loc[:,'first_funding_at'] = clean_cb_ids['first_funding_at'].dt.year
year_most_funding = stats.mode(clean_cb_ids.first_funding_at)[0][0] # Year with more funding rounds
clean_cb_ids.loc[:,'funding_rounds'] = clean_cb_ids.loc[:,'funding_rounds'].astype('int')
avg_funding_round = int(np.mean(clean_cb_ids['funding_rounds'])) # Average # of Funding Rounds
clean_cb_ids.loc[:,'funding_total_usd'] = clean_cb_ids.loc[:,'funding_total_usd'].astype('int')
avg_funding = np.round(np.mean(clean_cb_ids['funding_total_usd']),2) # Average Funding
clean_cb_ids['funding_total_usd'] = clean_cb_ids['funding_total_usd'].apply(lambda x: "${:,.1f}k".format((x/1000)))
clean_cb_ids = clean_cb_ids.rename(columns={"normalized_name":"Company",
                                    "permalink": "More info At",
                                    "founded_at": "Founded In",
                                    "first_funding_at":"Year of First Funding",
                                    "funding_rounds":"Funding Rounds",
                                    "funding_total_usd": "Funding Total in $US"})


col1, col2, col3 = st.columns(3)
col1.metric("Year with The Most Funding Rounds", "{}".format(year_most_funding), "")
col2.metric("Avg. Funding", "${:,.1f}k".format((avg_funding/1000)), "")
col3.metric("Avg. # of Rounds", "{}".format(avg_funding_round), "")

top_company = clean_cb_ids.iloc[:1,:1].values[0][0].title()
cb_url_company = clean_cb_ids.iloc[:1,1:2].values[0][0]
st.markdown('<h4 style="font-family:Avenir,Helvetica Neue,sans-serif;"> Popular Company in this Industry & Status: <a href="{}" target="_blank"> {}</a> </h4>'.format(cb_url_company, top_company), unsafe_allow_html=True)
st.write("")
st.write("")


st.markdown('<p style="font-family:Avenir,Helvetica Neue,sans-serif;"> You can access and download additional data of this and other companies in this industry by ticking the box below. </p>', unsafe_allow_html=True)

def get_table_download_link(df):
            """Generates a link allowing the data in a given panda dataframe to be downloaded
            in:  dataframe
            out: href string
            """
            df = df.reset_index()
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(
                csv.encode()
            ).decode()  # some strings <-> bytes conversions necessary here
            return f'<a style="font-family:Avenir,Helvetica Neue,sans-serif;" href="data:file/csv;base64,{b64}" download="companies_export.csv">Download CSV</a>'

if st.checkbox('More Data', False): #Creates a checkbox to show/hide the data
    st.write(clean_cb_ids)
    st.markdown(get_table_download_link(clean_cb_ids), unsafe_allow_html=True)


st.write("")
st.markdown("""<hr style="height:1px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)


# Comparison by status
st.header('Exploratory Analysis')
st.write("")

# Grouped data
# Create dataframe to compare by status
status_comp = clean_cb[clean_cb.category_code.isin(category)].copy().sort_values(by="funding_total_usd", ascending=False)
status_comp = status_comp[['status','normalized_name','founded_at','first_funding_at','funding_rounds','funding_total_usd']]
status_comp.loc[:,'status'] = status_comp['status'].astype("object")
status_comp.loc[:,'founded_at'] = status_comp['founded_at'].dt.year
status_comp.loc[:,'first_funding_at'] = status_comp['first_funding_at'].dt.year
status_comp.loc[:,'funding_rounds'] = status_comp.loc[:,'funding_rounds'].astype('int')
status_comp.loc[:,'funding_total_usd'] = status_comp.loc[:,'funding_total_usd'].astype('int')
status_comp = status_comp.rename(columns={"status":"Status",
                                          "normalized_name":"Company",
                                          "founded_at": "Founded In",
                                          "first_funding_at":"Year of First Funding",
                                          "funding_rounds":"Funding Rounds",
                                          "funding_total_usd": "Funding Total in $US"})
# Map categories from int to labels
cats = ['Acquired', 'Closed', 'IPO', 'Operating']
cat_map = dict(zip(np.sort(status_comp['Status'].unique()), cats))
status_comp['Status'] = [cat_map[x] for x in status_comp['Status']]

# Group data by status
groupby_status = pd.DataFrame(status_comp.groupby(by='Status').median())

st.markdown('<p style="font-family:Avenir,Helvetica Neue,sans-serif;"> Select an attribute to compare across company status: </p>', unsafe_allow_html=True)
#Comparison between Zones
selected_col = st.selectbox("", groupby_status.columns[2:], key='variable_box',index=0) #Add a dropdown element

# Plotly figure
fig_group_stat = go.Figure(data=[go.Bar(x=groupby_status.index, y=groupby_status[selected_col])])
fig_group_stat.update_traces(marker_color='#1c2d54', opacity=.99)
fig_group_stat.update_layout(title_text='Median {} by Status'.format(selected_col))
st.plotly_chart(fig_group_stat, use_container_width=True) 

st.markdown("""<hr style="height:1px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)


# SHAP
#[0, 1, 2, 3] = ['acquired', 'closed', 'ipo', 'operating']
# Relationships = "Representation of the people involved in the team for that startup"
st.header('Attribute Impact')
st.write("")

st.markdown('<p style="font-family:Avenir,Helvetica Neue,sans-serif;"> On this particular industry, according to our data these are some of the attributes of startup that seem to have a greater impact on a given company status. </p>', unsafe_allow_html=True)

X = df_train.drop(['status','category_code'], axis=1).copy()

shap.summary_plot(shap_values, X, class_names=["Acquired", "Closed", "IPO'd", "Operating"], max_display=5)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot(bbox_inches='tight')
plt.clf()

st.write("")
st.write("")
st.markdown('<p style="font-family:Avenir,Helvetica Neue,sans-serif;"> As we can see, different attributes contribute differently to classifying whether a company will be on each of the possible categories (status). For more details on how this is calculated please see the SHAP paper <a href="https://arxiv.org/pdf/1705.07874.pdf" target="_blank"> here</a>.</p>', unsafe_allow_html=True)

st.write("")
st.write("")
st.markdown('<p style="font-family:Avenir,Helvetica Neue,sans-serif;"> <b>* Disclaimer: </b> <i> This tool is not meant to replace a financial advisor, its purpose is rather to provide some direction into which key metrics should be considered when analyzing a potential industry. All insights are meant to be descriptive of the data and not to be used for inference. Nonetheless, we hope the tool can be leveraged by all types of users to efficiently get a better starting point for more informed conversations and research.</i> </p>', unsafe_allow_html=True)

st.text("")
st.text("")
st.write('<h4 style="font-family:Avenir,Helvetica Neue,sans-serif;"> For any suggestions on improvements please see the project <a href= https://github.com/omartinez182/510-Startup-Project target="_blank"> Github repo</a>.</h4>', unsafe_allow_html=True)
