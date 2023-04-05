#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import python libararies 
import pandas as pd1
import numpy as np1
import seaborn as sns1
import matplotlib.pyplot as plt4


# In[2]:


def read_worldbank_data(filename):
    """
    A function that reads in a World Bank dataset from a CSV file and returns two Pandas dataframes:
    dtf_year: a df with one row for each year of data, with columns for country name, indicator name, and values for each year.
    dtf_country: a df with one row for each country, with columns for year, indicator name, and values for each year.
    """
    
    # Read in the data and skip the first 4 rows
    dtf = pd1.read_csv(filename, skiprows=4)
    
    # Drop unnecessary columns
    dtf_country = dtf.drop(columns=['Country Code', 'Indicator Code' ,'1960','1961','1962','1963'
                                    ,'1964','1965','1966','1967','1968','1969','1970','1971','1972'
                                    ,'1973','1974','1975','1976','1977','1978','1979','1980','1981'
                                    ,'1982','1983','1984','1985','1986','1987','1988','1989','1990'
                                    ,'1991','1992','1993','1994','1995','1996','1997','1998','1999'
                                    ,'2001','2002','2003','2004','2006','2007','2008','2009','2011'
                                    ,'2012','2013','2014','2016','2017','2018','2019','Unnamed: 66'], inplace=True)
    
    # Set the index to be the country name and transpose the dataframe
    dtf_country = dtf.set_index('Country Name').T
    
    # Reset the index to include both country name and year
    dtf_year = dtf.set_index('Country Name').reset_index()
    
    # Return both dataframes
    return dtf_year, dtf_country


# In[3]:


#read top 5 data using head function 
dtf_year, dtf_country = read_worldbank_data('new climate.csv')
dtf_year.head()


# In[4]:


# call country name 
dtf_country


# In[5]:


# drop country name and indicator on axis-1 
data2=dtf_year.drop(["Country Name","Indicator Name"],axis=1)


# In[6]:


# replace with null to 0
dtf_year.fillna(0, inplace=True)
dtf_year


# In[7]:


# Display summary statistics for all numerical columns

dtf2= dtf_year.describe()
dtf2


# In[8]:


def skew(dist):
    """ Calculates the centralised and normalised skewness of dist. """
    
    # calculates average and std, dev for centralising and normalising
    aver = np1.mean(dist)
    std = np1.std(dist)
    
    # now calculate the skewness
    value = np1.sum(((dist-aver) / std)**3) / len(dist-1)
    
    return value


def kurtosis(dist):
    """ Calculates the centralised and normalised excess kurtosis of dist. """
    
    # calculates average and std, dev for centralising and normalising
    aver = np1.mean(dist)
    std = np1.std(dist)
    
    # now calculate the kurtosis
    value = np1.sum(((dist-aver) / std)**4) / len(dist-1) - 3.0
    
    return value

print("skewness =", np1.round(skew(data2), 2))
print("\n\nkurtosis =", np1.round(kurtosis(data2), 2))


# In[9]:


#replace with second fun null to 0 
dtf_year.fillna(0, inplace=True)
dtf_year


# In[10]:


def attribute_function(indicators, df):
    """
    create a attribute funcation with take a two aggrument such as indicators and df named.
    indicators-take a indicators name from dataset,
    df- the pandas df containing climate change dataset
    
    """

    # Split the given indicators on commas and use them to filter the dataframe
    indicators_list = indicators.split(",")
    df = df[df['Indicator Name'].isin(indicators_list)]

    # Return the resulting dataframe
    return df



# In[11]:


def country(cty, df):
    """
    This function takes two arguments: 'cty' and 'df'. 

    cty - A string representing the countries name  for which we want to extract data.
    df - The pandas dataframe containing the climate change dataset.

    Returns:
    A new pandas dataframe that includes the data only for the name of specific counries name .
    """
    df = df[df['Country Name'].isin([cty])]
    # using set index funcation for indicator name 
    df=df.set_index("Indicator Name")
    # using drop fun for drop the country name 
    df=df.drop("Country Name",axis=1)
    # using transpose 
    df=df.T
    # reutrn fun 
    return df

US_pop_Compare=country("United States",dtf_year)


# In[12]:


# define the four indicator name 
total_pop=US_pop_Compare[["CO2 emissions from liquid fuel consumption (kt)",
                          "CO2 emissions (kt)",
                          "Other greenhouse gas emissions, HFC, PFC and SF6 (thousand metric tons of CO2 equivalent)",
                          "Population, total"]]


# In[13]:


# call the correlation 
ed=total_pop.corr()
ed


# In[14]:


# give the fig zie 
ax, fig = plt4.subplots(figsize=(15, 9))

# Create the heatmap and set the x-tick labels to be rotated by 45 degrees
ax = sns1.heatmap(ed, annot=True)
# give the direction of the diagram 
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
# call the plot 
plt4.show()


# In[15]:


# Use the attribute_function() to get the dataframe for the desired indicator
urban_agglo = attribute_function("Population in urban agglomerations of more than 1 million (% of total population)",dtf_year)

# Filter the dataframe to include only the countries in the list
countries = ['Pakistan', 'United States', 'China']
urban_agglo = urban_agglo[urban_agglo['Country Name'].isin(countries)]

#  showing the data for each country in the list in contex of step diagram 
plt4.step(urban_agglo['Country Name'], urban_agglo['2020'])

# Rotae the 90 degree digram 
plt4.xticks(rotation=90)

# give the lable name in in context of y-lable 
plt4.ylabel('Population < 1 million')

# Add a title to the plot
plt4.title('Population in urban agglomerations of more than 1 million (% of total population) by Country')

# call plot 
plt4.show()


# In[16]:


# Filter dt_year to only include rows containing the 'Urban population living in areas where elevation is below 5 meters (% of total population)' indicator
urban_pop_elevation = attribute_function("Urban population living in areas where elevation is below 5 meters (% of total population)", dtf_year)

# Print the resulting dataframe
urban_pop_elevation


# In[17]:


# Define a list of countries to plot
countries = ['Pakistan', 'United States', 'China']

# Filter the dataframe to include only the countries in the list
urban_pop_elevation = urban_pop_elevation[urban_pop_elevation['Country Name'].isin(countries)]

# a bar plot showing the data for each country in the list
plt4.bar(urban_pop_elevation['Country Name'], urban_pop_elevation['2000'])

# Rotate the x-axis labels by 90 degrees for better readability
plt4.xticks(rotation=90)

# Add a y-axis label
plt4.ylabel('Urban population (% of total population)')

# Add a title to the plot
plt4.title('Urban population')

# Display the plot
plt4.show()


# In[18]:


# define country name and indicators for % of total final land area
countries = [ 'United States', 'China', 'Zimbabwe']
# give the ind name from dataset
ind = ['Renewable energy consumption (% of total final energy consumption)','GDP (current US$)']
# give the country name and ind name 

dtf = dtf_year[dtf_year['Country Name'].isin(countries) & dtf_year['Indicator Name'].isin(ind)]

# Group the data by country and compute the mean
df_mean = dtf.groupby('Indicator Name').mean().reset_index()

#   show the result
df_mean


# In[25]:


# Define a list of countries and indicators to plot
countries = ['Pakistan', 'United States', 'China', 'Zimbabwe']
# give the two ind name 
ind = ['Renewable energy consumption (% of total final energy consumption)','CO2 emissions (kt)' ]

# Filter the dataframe to include only the selected countries and indicators
dt = dtf_year[dtf_year['Country Name'].isin(countries) & dtf_year['Indicator Name'].isin(ind)]

# Group the filtered data by indicator and calculate the mean for each group
df_mean = dtf.groupby('Indicator Name').mean().reset_index()

# Set the size of the figure
plt4.figure(figsize=(11, 8))

# Extract the GDP data from the grouped dataframe and plot it, using circles (markers) to denote each data point
gdp_dat = df_mean[df_mean['Indicator Name'] == 'CO2 emissions (kt)']
plt4.plot(dt['Country Name'], marker='o', label='GDP')

# Add a label to the x-axis
plt4.xlabel('Value')

# Add a label to the y-axis
plt4.ylabel('Country names')

# Add a title to the plot
plt4.title('GDP for Select Countries')

# Add a legend to the plot, showing what the plotted line represents
plt4.legend()

# Display the plot
plt4.show()


# In[20]:


# define 10 data from data set

dtf = dtf.head(10)
# use the ggplot for ploting 

plt4.style.use('ggplot')
# select year columns and give the fig size 
dtf.boxplot(column=[ '2000', '2005', '2010', '2015'], figsize=(15, 5), vert=False, flierprops=dict(marker='o', markersize=8, markerfacecolor='gray'))

# give the tital name 
plt4.title("Box Plot of Urban Population Growth (2000 - 2015)")
# lable name 
plt4.xlabel('Urban Population Growth (%)')

plt4.show()


# In[21]:


# replace with null to 0
dtf.fillna(0, inplace=True)
dtf


# In[22]:


# copy the funcation 
df1 = dtf
# set index
df1.set_index("Indicator Name",inplace = True)
#df1 = df1.loc["'Urban population (% of total population')"]
df1 = df1.reset_index(level = "Indicator Name")
df1 = df1.head(15)
# call fun
df1


# In[23]:


# drop 2021 column year 
df1.set_index("Country Name",inplace = True)
# drop 2021
df1 = df1.drop("2021",axis = 1)
df1


# In[24]:


# Drop the first column of the dataframe
df2 = df1.drop(df1.columns[0], axis=1)

# Set a title for the plot
plt4.title("Forest area (% of land area)")

# Set the figure size for the heatmap using seaborn
sns1.set(rc = {'figure.figsize':(12,7)})

# Create a heatmap of the data using seaborn and annotate each cell with one decimal point
p1 = sns1.heatmap(df2, annot=True, fmt=".1f", linewidth=1)


# In[ ]:





# In[ ]:




