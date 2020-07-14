#!/usr/bin/env python
# coding: utf-8

# # Part 1
# 

# In[2]:


import pandas as pd
import numpy as np
import requests


# ## pulling data out of XML

# In[3]:


source=requests.get('https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M').text


# In[6]:


from bs4 import BeautifulSoup
sp=BeautifulSoup(source,'xml')
table=sp.find('table')
column_names=['Postalcode','Borough','Neighborhood']
df = pd.DataFrame(columns=column_names)
df


# ## import data into Data Frame

# In[9]:




for tr_cell in table.find_all('tr'):
    row_data=[]
    for td_cell in tr_cell.find_all('td'):
        row_data.append(td_cell.text.strip())
    if len(row_data)==3:
        df.loc[len(df)] = row_data
df.head()


# ## Ignore cells with a borough that is Not assigned

# In[14]:


df=df[df['Neighborhood']!= 'Not assigned']
df=df[df['Borough'] != 'Not assigned']
df.head()


# ## combine neighborhoods with the same Postalcode

# In[18]:


new_df=df.groupby('Postalcode')['Neighborhood'].apply(lambda x: "%s" % ', '.join(x))
new_df=new_df.reset_index(drop=False)
merged_df = pd.merge(df[['Postalcode','Borough']], new_df, on='Postalcode')
merged_df.drop_duplicates(inplace=True)
merged_df


# In[19]:


# number of rows of dataframe
merged_df.shape


# # Part 2

# In[21]:


get_ipython().system('pip install geocoder')


# In[26]:


import geocoder # import geocoder
def get_geocode(postal_code):
    # initialize your variable to None
    lat_lng_coords = None
    while(lat_lng_coords is None):
        g = geocoder.google('{}, Toronto, Ontario'.format(postal_code))
        lat_lng_coords = g.latlng
    latitude = lat_lng_coords[0]
    longitude = lat_lng_coords[1]
    return latitude,longitude


# In[30]:


geo_df=pd.read_csv('http://cocl.us/Geospatial_data')
geo_df.rename(columns={'Postal Code':'Postalcode'},inplace=True)
geo_merged = pd.merge(geo_df,merged_df, on='Postalcode')
geo_merged.head()


# In[31]:


new_geo_merged=geo_merged[['Postalcode','Borough','Neighborhood','Latitude','Longitude']]
new_geo_merged.head()


# In[32]:


toronto_data=new_geo_merged[new_geo_merged['Borough'].str.contains("Toronto")]
toronto_data.head()


# # Part 3

# In[33]:


CLIENT_ID = 'CMMOWFY3NUDIMWQJZ5RCDQ5XPIHZQOGMBVI20RJAGIX3UAJN'
CLIENT_SECRET = 'XSX2J1KYKZ51DLUASURX0CVLTFJJA20E4203XSUDIILYJKRI'
VERSION = '20180604'

def getNearbyVenues(names, latitudes, longitudes):
    radius=500
    LIMIT=100
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[34]:


toronto_venues = getNearbyVenues(names=toronto_data['Neighborhood'],
                                   latitudes=toronto_data['Latitude'],
                                   longitudes=toronto_data['Longitude']
                                  )


# In[35]:


toronto_venues.head()
toronto_venues.groupby('Neighborhood').count()


# In[53]:


toronto_onehot = pd.get_dummies(toronto_venues[['Venue Category']], prefix="", prefix_sep="")
toronto_onehot.drop(['Neighborhood'],axis=1,inplace=True) 
toronto_onehot.head()


# In[54]:


toronto_onehot.insert(loc=0,column='Neighborhood', value=toronto_venues['Neighborhood'] )
toronto_onehot


# In[55]:


toronto_grouped = toronto_onehot.groupby('Neighborhood').mean().reset_index()
toronto_grouped.head()


# In[56]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[57]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = toronto_grouped['Neighborhood']

for ind in np.arange(toronto_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(toronto_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# ## Making Clusters for Neighborhood

# In[59]:


from sklearn.cluster import KMeans
# set number of clusters
kclusters = 5

toronto_grouped_clustering = toronto_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(toronto_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10]


# In[60]:


# add clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

toronto_merged = toronto_data

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
toronto_merged = toronto_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

toronto_merged.head()


# In[61]:


neighborhoods_venues_sorted.head()


# In[64]:


get_ipython().system('pip install geopy')


# In[65]:


from geopy.geocoders import Nominatim 
address = 'Toronto, CA'

geolocator = Nominatim(user_agent="ny_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Manhattan are {}, {}.'.format(latitude, longitude))


# In[ ]:





# In[68]:


import folium 
import matplotlib.cm as cm
import matplotlib.colors as colors
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(toronto_merged['Latitude'], toronto_merged['Longitude'], toronto_merged['Neighborhood'], toronto_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# In[ ]:




