"""
Title:      DCLA Distribution Fairness Across Zip Codes
URL:        https://aeridona.github.io/ 
"""

import json
from urllib.request import urlopen

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import sklearn as sk
import sklearn.model_selection
import sklearn.linear_model
import sklearn.naive_bayes
import sklearn.svm
from sklearn.preprocessing import PolynomialFeatures
import sklearn.ensemble
import pandas as pd
import numpy as np


def clean_data(csv_grants="DCLA_Programs_Funding.csv",csv_org_info="DCLA_Cultural_Organizations.csv",csv_zip_demographics="Demographic_Statistics_By_Zip_Code.csv"):
    
    """
    Prior to this, I exported the "DCLA Funding" dataset from NYC open data where the application
    number contained FY20, the most recent fiscal year with data.
    I also downloaded "Demographic Statistics by Zip Code, changing its "JURISDICTION NAME" column
    name to "Postcode" to match the column in the other file I downloaded, "DCLA_Cultural_Organizations.
    The DCLA_Programs_Funding csv file had its "organization" column changed to "organization name" for the same
    reason
    """

    zip_demo = pd.read_csv(csv_zip_demographics)
    grants = pd.read_csv(csv_grants)
    org_info = pd.read_csv(csv_org_info)

    """As I'm focusing our analysis on ethnicity only, I drop irrelevant regional information columns below.
       Also, keeping only the percentage of each ethnicity per zip rather than the count for easier analysis."""
    
    bad_zip_columns = ["COUNT PARTICIPANTS", "COUNT FEMALE", "PERCENT FEMALE", "COUNT MALE", 
                       "PERCENT MALE",	"COUNT GENDER UNKNOWN",	"PERCENT GENDER UNKNOWN",	
                       "COUNT GENDER TOTAL", "PERCENT GENDER TOTAL", "COUNT ETHNICITY TOTAL",	
                       "COUNT PERMANENT RESIDENT ALIEN",	
                       "PERCENT PERMANENT RESIDENT ALIEN", "COUNT US CITIZEN", "PERCENT US CITIZEN", 
                       "COUNT OTHER CITIZEN STATUS", "PERCENT OTHER CITIZEN STATUS", "COUNT CITIZEN STATUS UNKNOWN",
                       "PERCENT CITIZEN STATUS UNKNOWN", "COUNT CITIZEN STATUS TOTAL", "PERCENT CITIZEN STATUS TOTAL",	
                       "COUNT RECEIVES PUBLIC ASSISTANCE",	"PERCENT RECEIVES PUBLIC ASSISTANCE", "COUNT NRECEIVES PUBLIC ASSISTANCE",
                       "PERCENT NRECEIVES PUBLIC ASSISTANCE", "COUNT PUBLIC ASSISTANCE UNKNOWN", "PERCENT PUBLIC ASSISTANCE UNKNOWN",
                       "COUNT PUBLIC ASSISTANCE TOTAL",	"PERCENT PUBLIC ASSISTANCE TOTAL", "COUNT PACIFIC ISLANDER",
                       "COUNT HISPANIC LATINO", "COUNT AMERICAN INDIAN", "COUNT ASIAN NON HISPANIC",
                       "COUNT WHITE NON HISPANIC", "COUNT BLACK NON HISPANIC", "COUNT OTHER ETHNICITY",
                       "COUNT ETHNICITY UNKNOWN", "COUNT ETHNICITY TOTAL"]

    bad_grant_columns = "Application #"

    """Eliminating irrelevant columns in the organization info, keeping zip + name + lat and longitude coordinates."""

    bad_org_info_columns = ["Address", "City", "State", "Main Phone #", "Discipline",
                            "Council District", "Community Board",
                            "Census Tract", "BIN", "BBL", "NTA"]

    """Dropping all irrelevant columns."""

    zip_demo = zip_demo.drop(bad_zip_columns, axis=1)
    grants = grants.drop(bad_grant_columns, axis=1)
    org_info = org_info.drop(bad_org_info_columns, axis=1)

    """Merging the columns into one dataset, with zip+org_info merged on postcode and grants merged
       with the first merge on organization name."""

    zip_demo['Postcode'] = zip_demo['Postcode'].apply(str)

    """Postcode is originally a int64 column, so casting to string for merge."""

    csv_one = pd.merge(zip_demo, org_info, on='Postcode')
    csv_two = pd.merge(csv_one, grants, on="Organization Name")
    
    csv_two.to_csv('Project_CSV.csv')

    """I didn't code this, but I deleted all columsn where PERCENT ETHNICITY was equal to 0 (indicating no demographic information."""

    
def white_nonwhite_piechart(csv="Project_CSV.csv"):
    original_csv = pd.read_csv(csv)

    """In this project, I am dividing the organizations by zip code, and defining a white-majority
       area as an area where over 50% of the population is white, and vice versa."""
       
    white_majority_df = original_csv[original_csv['PERCENT WHITE NON HISPANIC'] > 0.50]
    white_majority_total_allocated = white_majority_df['Total Final Award'].sum()

    nonwhite_majority_df = original_csv[original_csv['PERCENT WHITE NON HISPANIC'] < 0.50]
    nonwhite_majority_total_allocated = nonwhite_majority_df['Total Final Award'].sum()
    
    amounts = [white_majority_total_allocated, nonwhite_majority_total_allocated]
    labels = ['White Majority Areas', 'Nonwhite Majority Areas']
    plt.pie(amounts, labels=labels)
    plt.show()

def borough_piechart(csv="PROJECT_CSV.csv"):
    original_csv = pd.read_csv(csv)
    new_csv = original_csv.groupby(['Borough'])['Total Final Award'].sum()
    new_csv.plot(kind='pie', y='Total Final Award')
    plt.show()
    
def percentage_amount_line_chart(csv="PROJECT_CSV.csv"):
    """For this plot, I set the x axis to the percentage of white residents (0.0-1.0 for 0%-100%,
    and the y axis to amount granted."""
    
    original_csv = pd.read_csv(csv)
    
    new_csv = original_csv.groupby(['PERCENT WHITE NON HISPANIC'])['Total Final Award'].sum().plot(legend=True)
    
    plt.title("White Population Percentage vs Amount Awarded")
    plt.xlabel("Percentage of Residents that are White")
    plt.ylabel("Total Amount Awarded")
    plt.grid(True)
    plt.show()

def scatterplot(csv="PROJECT_CSV.csv"):
    original_csv = pd.read_csv(csv)
    original_csv["Information"] =  original_csv["Organization Name"] + " | Total Awarded: "
    + original_csv["Total Final Award"].astype(str)
    """The above renders what appears on hover for the scatterplot"""

    org_plot = go.Figure(data=go.Scattergeo
    (
            locationmode = 'USA-states',
            lon= original_csv['Longitude'],
            lat = original_csv['Latitude'],
            text = original_csv['Information'],
            mode = 'markers',
            marker = dict(
                size = 6,
                opacity = 1.0,
                reversescale = True,
                autocolorscale = False,
                symbol = 'square',
                line = dict(
                    width=1,
                    color='rgba(102, 102, 102)'
                ),
                colorscale = 'Reds',
                cmin = original_csv['Total Final Award'].min(),
                color = original_csv['Total Final Award'],
                cmax = original_csv['Total Final Award'].max(), 
                colorbar_title="FY20 Total Amount Awarded"
            )))

    org_plot.update_geos(
        fitbounds="locations"
    )
    """Update geos just centers the map on the location of the information."""

    org_plot.update_layout(
            title = 'Cultural Organizations with FY20 Amount Awarded',
            geo = dict(
                scope='usa',
                projection_type='albers usa',
                showland = True,
                landcolor = "rgb(250, 250, 250)",
                subunitcolor = "rgb(217, 217, 217)",
                countrycolor = "rgb(217, 217, 217)",
                countrywidth = 0.5,
                subunitwidth = 0.5
            ),
        )

    org_plot.show()

def choropleth(csv="PROJECT_CSV.csv"):
    original_csv = pd.read_csv(csv)
    original_csv = original_csv.groupby(['Postcode'])['Total Final Award'].sum().reset_index()    
    """Because the information I have is organized by zip code, not FIPS or county, I load in a GeoJSON
       containing the zip code boundaries of NY.
       
       Furthermore, because the geoJSON is MASSIVE, I am loading it from a raw file somehow helpfully posted online."""

    with urlopen("https://raw.githubusercontent.com/OpenDataDE/State-zip-code-GeoJSON/master/ny_new_york_zip_codes_geo.min.json") as response:
        zip_codes = json.load(response)

    fig = px.choropleth(
        original_csv, 
        geojson=zip_codes, 
        locations = 'Postcode', 
        featureidkey = "properties.ZCTA5CE10",
        color = 'Total Final Award', 
        color_continuous_scale="twilight", 
        range_color=(original_csv['Total Final Award'].min(), original_csv['Total Final Award'].max()),
        scope="usa", 
        labels={'Total Final Award':'Total Amount Awarded'}
        )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    fig.update_geos(fitbounds="locations")
    fig.show()
    """For the website, I rendered this again in plotly's chart studio so that it could be hosted simply in html.
        For that plot, the zip code geoJSON has much more simplified geometry because otherwise
        the graph was too big for the free hosting plan. Thanks mapshaper!"""
    
def split_data(data, target, random_state = 21):
    x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(data, target, random_state=random_state)
    x_train= x_train.values.reshape(-1, 1)
    x_test = x_test.values.reshape(-1, 1)
    return x_train, x_test, y_train, y_test

def fit_linear_regression():
    csv = pd.read_csv('PROJECT_CSV.csv')
    
    
    polynomial = sk.preprocessing.PolynomialFeatures(degree=12, include_bias=False)
    csv['Total Final Award'] = polynomial.fit_transform(csv['Total Final Award'].values.reshape(-1,1))
    
    x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(csv['PERCENT WHITE NON HISPANIC'], csv['Total Final Award'])

    """Reshaped the training sets to be 2D arrays."""
    x_train= x_train.values.reshape(-1, 1)
    x_test = x_test.values.reshape(-1, 1)
    
    linear_model = sk.linear_model.LinearRegression(fit_intercept=True).fit(x_train,y_train)

    score = linear_model.score(x_test,y_test)
    """The score was 0.0006779082522238022. I think that's bad."""
    
    y_predicted = linear_model.predict(x_test)

    plt.scatter(x_test, y_test, color='blue')
    plt.plot(x_test, y_predicted, color='red', linewidth=3)
    plt.xticks(())
    plt.yticks(())
    plt.show()

