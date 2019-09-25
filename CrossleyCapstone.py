### ********************************************************************************* ###
### --------------------------------------------------------------------------------- ###
### --------------------------------------------------------------------------------- ###
### --------------- 2017 Citi Bike Operating Report by Kevin Crossley --------------- ###
### --------------------------------------------------------------------------------- ###
### --------------------------------------------------------------------------------- ###
### ********************************************************************************* ###


### ------------------------------------------------ ###
### --------------- Import Libraries --------------- ###
### ------------------------------------------------ ###
print('Started library import')

from sklearn import svm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
import numpy as np
from geopy import distance
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import requests
import io
import osmnx as ox
import networkx as nx
from matplotlib.collections import LineCollection
import time
import requests
from bs4 import BeautifulSoup
import seaborn as sns

t4 = time.time()

### ------------------------------------------------ ###
### --------------- Source Bike Data --------------- ###
### ------------------------------------------------ ###

## Need Citi Bike data from all of 2017


# Create save_data function to grab data from citi bike website and save 2017 data to files for each month
def save_data():

    ## Data hosted in the following url format:
    url_start = 'https://s3.amazonaws.com/tripdata/2017'  # initial location string
    url_end = '-citibike-tripdata.csv.zip'  # end of string
    # list of months as 2 digit strings
    url_month = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

    # Use for loop to run through each month's url to save data
    for index in range(0, 12):
        # Combine url pieces to grab zip file of each month in 2017
        url = url_start + url_month[index] + url_end

        # print(url)

        ## Get Zip file from url and extract csv to file
        r = requests.get(url)  # http request to url
        z = zipfile.ZipFile(io.BytesIO(r.content))  # create zip file object
        z.extractall('/Users/kevincrossley/Documents/DevUp/Capstone Data')  # extract csv from object and save


### --------------------------------------------------------------- ###
### --------------- Create Graph of Manhattan Roads --------------- ###
### --------------------------------------------------------------- ###

def manhattan_data():

    ## Use OSMNX library to download and save a street map of Manhattan from Open Street Map
    # Library config
    ox.config(use_cache=True, log_console=True)

    # Get the drive-able roads on Manhattan using OpenStreetMap api's to request data
    G = ox.graph_from_place('Manhattan Island, New York City, New York, USA', network_type='drive')

    # Save file to disk as graphML format
    ox.save_load.save_graphml(G, filename='ManhattanDrive.graphml',
                              folder='/Users/kevincrossley/Documents/DevUp/Capstone Data', gephi=False)


### --------------------------------------------------- ###
### --------------- Source Weather Data --------------- ###
### --------------------------------------------------- ###

def scrape_weather():

    ## Use Beautiful Soup to grab 2017 NYC weather data
    # URL Format: use date part of url for each month in 2017
    # https://www.wunderground.com/history/airport/KJRB/2017/1/1/MonthlyHistory.html

    # Create months vector for getting data
    months = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
    urls = []

    # Create urls list with one url for each month of data
    for month in months:
        url = 'https://www.wunderground.com/history/airport/KJRB/2017/' + month + '/1/MonthlyHistory.html'
        urls.append(url)

    # Put all data in all_data
    all_data = []

    # Nested for loops to go to each url and find each table of data
    # Then need to go through each 'body' element and each 'row' element to grab data
    for url in urls:
        # request html from each url
        page = requests.get(url)
        # parse page with beautiful soup
        soup = BeautifulSoup(page.content, 'html.parser')

        # store data for each month in 'data'
        data = []
        # find the table with the weather data
        table = soup.find('table', attrs={'id': 'obsTable'})
        # each day of data stored in its own body element
        table_body = table.find_all('tbody')

        # for each body (day of data)
        for body in table_body:
            # assign each row to rows
            rows = body.find_all('tr')

            # for each row of data
            for row in rows:
                # assign each data element to the 'data' list
                cols = row.find_all('td')
                cols = [ele.text.strip() for ele in cols]
                data.append([ele for ele in cols])
        # add each month of data to 'all_data'
        all_data.append(data)

    # turn to data frame and save to csv
    df_data = pd.DataFrame(all_data)
    df_data.to_csv('/Users/kevincrossley/Documents/DevUp/Capstone Data/weather.csv')


### ------------------------------------------------- ###
### --------------- Combine Bike Data --------------- ###
### --------------------------------------------------###

# Create function to read csv's and combine into one 2017 file
def combine_data():
    ## read csv's into pandas data frames and create file for entire year

    # create filepath string using same method as url
    filepath_start = '/Users/kevincrossley/Documents/DevUp/Capstone Data/2017'
    filepath_end = '-citibike-tripdata.csv'
    filepath_month = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

    # create filepath strings for each month
    filepath_jan = filepath_start + filepath_month[0] + filepath_end
    filepath_feb = filepath_start + filepath_month[1] + filepath_end
    filepath_mar = filepath_start + filepath_month[2] + filepath_end
    filepath_apr = filepath_start + filepath_month[3] + filepath_end
    filepath_may = filepath_start + filepath_month[4] + filepath_end
    filepath_jun = filepath_start + filepath_month[5] + filepath_end
    filepath_jul = filepath_start + filepath_month[6] + filepath_end
    filepath_aug = filepath_start + filepath_month[7] + filepath_end
    filepath_sep = filepath_start + filepath_month[8] + filepath_end
    filepath_oct = filepath_start + filepath_month[9] + filepath_end
    filepath_nov = filepath_start + filepath_month[10] + filepath_end
    filepath_dec = filepath_start + filepath_month[11] + filepath_end

    # create data frames for each month, including adding an index column as the first column
    # don't need january because all other months are appended to it
    df_feb = pd.read_csv(filepath_feb, index_col=0)
    df_mar = pd.read_csv(filepath_mar, index_col=0)
    df_apr = pd.read_csv(filepath_apr, index_col=0)
    df_may = pd.read_csv(filepath_may, index_col=0)
    df_jun = pd.read_csv(filepath_jun, index_col=0)
    df_jul = pd.read_csv(filepath_jul, index_col=0)
    df_aug = pd.read_csv(filepath_aug, index_col=0)
    df_sep = pd.read_csv(filepath_sep, index_col=0)
    df_oct = pd.read_csv(filepath_oct, index_col=0)
    df_nov = pd.read_csv(filepath_nov, index_col=0)
    df_dec = pd.read_csv(filepath_dec, index_col=0)

    ## append data frames together to have one frame of all of 2017
    with open(filepath_jan, 'a') as f:
        df_feb.to_csv(f, header=False)
        df_mar.to_csv(f, header=False)
        df_apr.to_csv(f, header=False)
        df_may.to_csv(f, header=False)
        df_jun.to_csv(f, header=False)
        df_jul.to_csv(f, header=False)
        df_aug.to_csv(f, header=False)
        df_sep.to_csv(f, header=False)
        df_oct.to_csv(f, header=False)
        df_nov.to_csv(f, header=False)
        df_dec.to_csv(f, header=False)


### ------------------------------------------------------------------ ###
### --------------- Initial Analysis and Visualization --------------- ###
### ------------------------------------------------------------------ ###

### -------------------------- Basic Stats -------------------------- ###


def basic_stats(df):

    tb1 = time.time()
    print('Started Basic Stats')

    # Total Rides
    rows, cols = df.shape
    total_rides = rows

    # Total Ride Time
    total_ride_time = df['Trip Duration'].sum()
    total_ride_time = round(total_ride_time / 31536000, 2)  # convert to years

    # Total Bikes
    total_bikes = df['Bike ID'].nunique()

    # Total Uses
    users = df['User Type'].value_counts()
    users = users.tolist()
    subscribers = users[0]
    customers = users[1]
    total_uses = subscribers + customers

    # Subscriber Percentage
    sub_percent = subscribers / total_uses
    sub_percent = round(sub_percent * 100, 2)

    # Customers Percentage
    cust_percent = customers / total_uses
    cust_percent = round(cust_percent * 100, 2)


    print('Total Rides: ', total_rides, '\n',
          'Total Ride Time: ', total_ride_time, '\n',
          'Total Bikes: ', total_bikes, '\n',
          'Subscribers Percentage: ', sub_percent, '\n',
          'Customers Percentage: ', cust_percent, '\n')

    # Total Rides:        163,64,657
    # Total Ride Time:           516.11
    # Total Bikes:            14,204
    # Subscribers Percentage:     89.18 %
    # Customers Percentage:       10.82 %

    tb2 = time.time()
    print('Time to run basic stats: ', round(tb2 - tb1, 4))


### ---------- 1. Top 5 stations with the most starts (including number of starts) ---------- ###

def question1(df, G):

    tq1_1 = time.time()
    print('Started Question 1')

    # Can use describe method to quickly get most common start station as 'top' category
    # ds = df['Start Station ID'].describe()
    # print('top Start Station: ', ds['top'])
    # print('count of starts: ', ds['freq'])

    # Otherwise, can get list of every station in order of most starts
    num_stations = 5  # just want top 5
    counts = df['Start Station ID'].value_counts()
    # print('top 5 stations--')
    # print(counts[0:num_stations])

    # top 5 stations:
    #     ID      Count
    # 1. 519    162,716
    # 2. 497    112,218
    # 3. 402    108,590
    # 4. 435    107,133
    # 5. 426    105,610
    # total: 16,364,660

    # initialize objects
    dtops = {'name': [], 'lat': [], 'lon': []}
    dtops = pd.DataFrame(dtops)

    # use for loop to create data frame with station names, latitudes, and longitudes for top stations
    for ii in range(0, num_stations):
        # find the first row containing the most popular start station
        row = df[df['Start Station ID'] == counts.index[ii]].iloc[0]

        # assign the name, latitude, and longitude of the station to dict
        drow = {'name': [row['Start Station Name']], 'lat': [row['Start Station Latitude']], 'lon': [row['Start Station Longitude']]}

        # turn dict into data frame and append to dtops
        drow = pd.DataFrame(drow)
        dtops = dtops.append(drow)

    # reset index after for loop
    dtops = dtops.reset_index(drop=True)
    # print(dtops)

    # dtops on entire data set:
    #                     name        lat        lon
    # 0  Pershing Square North  40.751873 -73.977706
    # 1     E 17 St & Broadway  40.737050 -73.990093
    # 2     Broadway & E 22 St  40.740343 -73.989551
    # 3        W 21 St & 6 Ave  40.741740 -73.994156
    # 4  West St & Chambers St  40.717548 -74.013221

    ## Create graph, but do not display yet
    fig, ax = ox.plot_graph(G, show=False, close=False)

    # assign coords of top stations to x and y
    x, y = dtops['lon'].tolist(), dtops['lat'].tolist()

    # create scatter plot of red dots for plotting on top of street map
    ax.scatter(x, y, c='red', s=50, zorder=5)  # zorder parameter ensures scatter plots in front of streets

    # create rank list
    n = [1, 2, 3, 4, 5]

    # add labels to the dots with their number, and give them an offset and semi-transparent text box
    for i, txt in enumerate(n):
        ax.annotate(txt, (x[i], y[i]), bbox=dict(facecolor='white', edgecolor='None', alpha=0.65),
                    xytext=(10, 10), textcoords='offset points')

    tq1_2 = time.time()

    print('Time to run question 1: ', round(tq1_2 - tq1_1, 4))

    # plt.show()


### ---------- 2. Trip duration by user type ---------- ###

def question2(df):
    ## Calculate stats on trip durations based on two user types - Subscriber and Customer

    tq2_1 = time.time()
    print('Started Question 2')

    ## Prep Data
    # Create set that drops unknown user type for analysis 2
    df2 = df[pd.notnull(df['User Type'])]

    # reset index for dropped rows
    df2 = df2.reset_index(drop=True)

    # Convert from seconds to minutes
    df2['Trip Duration'] = df2['Trip Duration'] / 60

    # Calculate means of each type using groupby() method to sort
    info = df2.groupby('User Type')['Trip Duration'].describe()
    # print(info)

    fig2 = df2.boxplot(column='Trip Duration', by='User Type')

    tq2_2 = time.time()
    print('Time to run question 2: ', round(tq2_2 - tq2_1, 4))

    # plt.show()

    #                  count         mean           std   min    25%     50%     75%        max
    # User Type
    # Customer       1769423  		   42  		    667  	1  	  14  	  21  	  29  	 159712 (min)
    # Subscriber    14579325   		   14   		166	    1  	   6   	  10   	  16  	 162266

    # Customer     1769423.0  2499.931518  40041.185539  61.0  843.0  1289.0  1733.0  9582723.0 (seconds)
    # Subscriber  14579325.0   811.377824   9945.224346  61.0  353.0   573.0   956.0  9735948.0


### ---------- 3. Most popular trips based on start station and stop station ---------- ###

def question3(df, G):

    tq3_1 = time.time()
    print('Starting Question 3')

    ###################
    ## Copy in and adapt ox.plot_graph_route to only output route between stations as Line Collection
    ## Comment out only one line (the last one) to prevent function from outputting a plot and instead
    ## change the return parameter to simply the Line Collection that can be added to the existing plot

    def plot_graph_route(G, route, bbox=None, fig_height=6, fig_width=None,
                         margin=0.02, bgcolor='w', axis_off=True, show=True,
                         save=False, close=True, file_format='png', filename='temp',
                         dpi=300, annotate=False, node_color='#999999',
                         node_size=15, node_alpha=1, node_edgecolor='none',
                         node_zorder=1, edge_color='#999999', edge_linewidth=1,
                         edge_alpha=1, use_geom=True, origin_point=None,
                         destination_point=None, route_color='r', route_linewidth=4,
                         route_alpha=0.5, orig_dest_node_alpha=0.5,
                         orig_dest_node_size=100, orig_dest_node_color='r',
                         orig_dest_point_color='b'):
        """
        Plot a route along a networkx spatial graph.
        Parameters
        ----------
        G : networkx multidigraph
        route : list
            the route as a list of nodes
        bbox : tuple
            bounding box as north,south,east,west - if None will calculate from
            spatial extents of data. if passing a bbox, you probably also want to
            pass margin=0 to constrain it.
        fig_height : int
            matplotlib figure height in inches
        fig_width : int
            matplotlib figure width in inches
        margin : float
            relative margin around the figure
        axis_off : bool
            if True turn off the matplotlib axis
        bgcolor : string
            the background color of the figure and axis
        show : bool
            if True, show the figure
        save : bool
            if True, save the figure as an image file to disk
        close : bool
            close the figure (only if show equals False) to prevent display
        file_format : string
            the format of the file to save (e.g., 'jpg', 'png', 'svg')
        filename : string
            the name of the file if saving
        dpi : int
            the resolution of the image file if saving
        annotate : bool
            if True, annotate the nodes in the figure
        node_color : string
            the color of the nodes
        node_size : int
            the size of the nodes
        node_alpha : float
            the opacity of the nodes
        node_edgecolor : string
            the color of the node's marker's border
        node_zorder : int
            zorder to plot nodes, edges are always 2, so make node_zorder 1 to plot
            nodes beneath them or 3 to plot nodes atop them
        edge_color : string
            the color of the edges' lines
        edge_linewidth : float
            the width of the edges' lines
        edge_alpha : float
            the opacity of the edges' lines
        use_geom : bool
            if True, use the spatial geometry attribute of the edges to draw
            geographically accurate edges, rather than just lines straight from node
            to node
        origin_point : tuple
            optional, an origin (lat, lon) point to plot instead of the origin node
        destination_point : tuple
            optional, a destination (lat, lon) point to plot instead of the
            destination node
        route_color : string
            the color of the route
        route_linewidth : int
            the width of the route line
        route_alpha : float
            the opacity of the route line
        orig_dest_node_alpha : float
            the opacity of the origin and destination nodes
        orig_dest_node_size : int
            the size of the origin and destination nodes
        orig_dest_node_color : string
            the color of the origin and destination nodes
        orig_dest_point_color : string
            the color of the origin and destination points if being plotted instead
            of nodes
        Returns
        -------
        fig, ax : tuple
        """

        # # plot the graph but not the route
        # fig, ax = plot_graph(G, bbox=bbox, fig_height=fig_height, fig_width=fig_width,
        #                      margin=margin, axis_off=axis_off, bgcolor=bgcolor,
        #                      show=False, save=False, close=False, filename=filename,
        #                      dpi=dpi, annotate=annotate, node_color=node_color,
        #                      node_size=node_size, node_alpha=node_alpha,
        #                      node_edgecolor=node_edgecolor, node_zorder=node_zorder,
        #                      edge_color=edge_color, edge_linewidth=edge_linewidth,
        #                      edge_alpha=edge_alpha, use_geom=use_geom)

        # the origin and destination nodes are the first and last nodes in the route
        origin_node = route[0]
        destination_node = route[-1]

        if origin_point is None or destination_point is None:
            # if caller didn't pass points, use the first and last node in route as
            # origin/destination
            origin_destination_lats = (G.nodes[origin_node]['y'], G.nodes[destination_node]['y'])
            origin_destination_lons = (G.nodes[origin_node]['x'], G.nodes[destination_node]['x'])
        else:
            # otherwise, use the passed points as origin/destination
            origin_destination_lats = (origin_point[0], destination_point[0])
            origin_destination_lons = (origin_point[1], destination_point[1])
            orig_dest_node_color = orig_dest_point_color

        # scatter the origin and destination points
        ax.scatter(origin_destination_lons, origin_destination_lats, s=orig_dest_node_size,
                   c=orig_dest_node_color, alpha=orig_dest_node_alpha, edgecolor=node_edgecolor, zorder=4)

        # plot the route lines
        edge_nodes = list(zip(route[:-1], route[1:]))
        lines = []
        for u, v in edge_nodes:
            # if there are parallel edges, select the shortest in length
            data = min(G.get_edge_data(u, v).values(), key=lambda x: x['length'])

            # if it has a geometry attribute (ie, a list of line segments)
            if 'geometry' in data and use_geom:
                # add them to the list of lines to plot
                xs, ys = data['geometry'].xy
                lines.append(list(zip(xs, ys)))
            else:
                # if it doesn't have a geometry attribute, the edge is a straight
                # line from node to node
                x1 = G.nodes[u]['x']
                y1 = G.nodes[u]['y']
                x2 = G.nodes[v]['x']
                y2 = G.nodes[v]['y']
                line = [(x1, y1), (x2, y2)]
                lines.append(line)

        # add the lines to the axis as a linecollection
        lc = LineCollection(lines, colors=route_color, linewidths=route_linewidth, alpha=route_alpha, zorder=3)
        ax.add_collection(lc)

        # save and show the figure as specified
        # fig, ax = save_and_show(fig, ax, save, show, close, filename, file_format, dpi, axis_off)
        return lc


    ###################

    num_stations = 5

    # group data by dual column start/end station ID pairs
    df3 = df.groupby(['Start Station ID', 'End Station ID']).size().reset_index(name='count')

    # sort grouped data by count
    df3 = df3.sort_values(by=['count'], ascending=False).reset_index(drop=True)

    ## get list of unique stations
    # first list every station
    statList = df3['Start Station ID'].iloc[0:num_stations].tolist() + \
               df3['End Station ID'].iloc[0:num_stations].tolist()

    # convert to set to get only uniques, convert back to list for use later
    statSet = set(statList)
    statList = list(statSet)

    # initialize objects
    droutes = {'Start Station ID': [], 'name': [], 'lat': [], 'lon': []}
    droutes = pd.DataFrame(droutes)

    # use for loop to create data frame with station names, latitudes, and longitudes for top stations
    for jj in statList:
        # find the first row containing the most popular start station
        row2 = df[df['Start Station ID'] == jj].iloc[0]

        # assign the name, latitude, and longitude of the station to dict
        drow2 = {'Start Station ID': row2['Start Station ID'], 'name': [row2['Start Station Name']],
                 'lat': [row2['Start Station Latitude']], 'lon': [row2['Start Station Longitude']]}

        # turn dict into data frame and append to droutes
        drow2 = pd.DataFrame(drow2)
        droutes = droutes.append(drow2)

    # reset index after for loop
    droutes = droutes.reset_index(drop=True)
    droutes['Start Station ID'] = droutes['Start Station ID'].astype('int64')
    droutes['Start Station ID'] = droutes['Start Station ID'].astype('category')

    ##  create list of lat lon pairs for start and end station for each trip
    # temp data frame just of top routes
    df3_temp = df3.iloc[0:num_stations]

    # merge name, lat, lon data into this table for start station
    df3_temp = df3_temp.merge(droutes, on='Start Station ID', how='left')

    # rename columns to avoid ambiguity
    df3_temp.rename(columns={'name': 'start name', 'lat': 'start lat', 'lon': 'start lon'}, inplace=True)

    # rename column in droutes to aid table merge
    droutes.rename(columns={'Start Station ID': 'End Station ID'}, inplace=True)

    # merge end station name, lat, lon
    df3_temp = df3_temp.merge(droutes, on='End Station ID', how='left')

    # rename to avoid ambiguity
    df3_temp.rename(columns={'name': 'end name', 'lat': 'end lat', 'lon': 'end lon'}, inplace=True)


    # print(df3_temp)

    #    Start Station ID  End Station ID  count                     start/end name  start lat  start lon
    # 0               432            3263   7994                  E 7 St & Avenue A  40.726218 -73.983799
    #                                                        Cooper Square & E 7 St  40.729236 -73.990868
    # 1              2006            2006   7169             Central Park S & 6 Ave  40.765909 -73.976342
    #
    # 2              2006            3282   6318             Central Park S & 6 Ave  40.765909 -73.976342
    #                                                               5 Ave & E 88 St  40.783070 -73.959390
    # 3               281             281   5670  Grand Army Plaza & Central Park S  40.764397 -73.973715
    #
    # 4               514             426   5403                   12 Ave & W 40 St  40.760875 -74.002777
    #                                                         West St & Chambers St  40.717548 -74.013221

    ## Use for loop to create route objects between each top station route
    route = []
    for kk in range(0, num_stations):

        # get the nearest network node to each point
        orig_node = ox.get_nearest_node(G, (df3_temp['start lat'].iloc[kk], df3_temp['start lon'].iloc[kk]))
        dest_node = ox.get_nearest_node(G, (df3_temp['end lat'].iloc[kk], df3_temp['end lon'].iloc[kk]))

        # find the route between these nodes
        route.append(nx.shortest_path(G, orig_node, dest_node, weight='length'))

    # plot first route
    fig, ax = ox.plot_graph_route(G, route[0], node_size=0, show=False, close=False)

    # Call modified local plot_graph_route function (not ox.plot_graph_route)
    # This modified function will add the remaining stations and routes to the existing axis
    for hh in range(1, num_stations):
        lc = plot_graph_route(G, route[hh], node_size=0)  # add station scatter points and return route
        ax.add_collection(lc)  # add route to axes

    x, y = df3_temp['end lon'].tolist(), df3_temp['end lat'].tolist()

    # create rank list
    n = [1, 2, 3, 4, 5]

    # add labels to the dots with their number, and give them an offset and semi-transparent text box
    for i, txt in enumerate(n):
        ax.annotate(txt, (x[i], y[i]), bbox=dict(facecolor='white', edgecolor='None', alpha=0.65),
                    xytext=(10, 10), textcoords='offset points')

    tq3_2 = time.time()
    print('Time to run question 3: ', round(tq3_2 - tq3_1, 4))
    # plt.show()


### ---------- 4. Rider performance by gender and age based on avg trip distance (station to station),
###               median speed (distance travelled / trip duration) ---------- ###

def question4(df):

    tq4_1 = time.time()
    print('Started Question 4')

    df4 = df

    # Create a column of bucketed ages
    # Young = 0-30
    # Middle = 31-50
    # Old = 51-118 (max based on adding 100 to every year < 1900. min birth year = 1900, max age is 117)
    df4['age bucket'] = pd.cut(df4['Age'], [0, 30, 50, 120], labels=['Young', 'Middle Aged', 'Old'])

    # Calculate shortest path distance
    # Set up temp df for convenience, contains pairs of lat/lons
    coords = pd.DataFrame()
    coords['Start Coordinates'] = list(zip(df4['Start Station Latitude'], df4['Start Station Longitude']))
    coords['End Coordinates'] = list(zip(df4['End Station Latitude'], df4['End Station Longitude']))

    # initialize list, calculate size of df
    distances = []
    rows, cols = df4.shape

    tloop = time.time()
    # Use for loop to calculate distance between each set of start/end points, append to distances list
    for gg in range(0, rows):
        # Calculate shortest path in distance in miles
        length = distance.distance(coords['Start Coordinates'].iloc[gg], coords['End Coordinates'].iloc[gg]).miles
        distances.append(length)

        if gg % 50000 == 0:
            tloop2 = time.time()
            print('Finished ', gg, 'out of ', rows, 'in ', round(tloop2-tloop, 4))
            tloop = time.time()

    # Add distances to df4 data frame
    df4['distance'] = distances

    # convert trip duration from seconds to hours
    df4['Trip Duration'] = df4['Trip Duration'] / 3600

    # Create speed column by dividing distance by time (miles/hrs) = mph
    df4['speed'] = df4['distance'] / df4['Trip Duration']

    # get mean of each column by age bucket and Gender
    age_means = df4.groupby(df4['age bucket']).mean()
    gender_means = df4.groupby(df4['Gender']).mean()

    # drop 0 (0 = 'unknown') category from gender column
    gender_means = gender_means.iloc[1:]
    print('\n')
    print(age_means[['distance', 'speed']])
    print(gender_means[['distance', 'speed']])
    print('\n')

    # age bucket  distance       speed
    # Young       1.150412    6.158020
    # Middle Aged 1.144603    6.000817
    # Old         1.101410    5.461068
    #
    # Gender distance      speed
    # 1     1.126227    6.121426
    # 2     1.177831    5.486445

    ## Plots
    # Two Bar Charts with paired bars

    # New Figure
    fig4 = plt.figure()

    # Two axes - for distance and speed
    ax4 = fig4.add_subplot(111)
    ax4_2 = ax4.twinx()  # uses same x axis

    # Bar width
    width = 0.2

    # Create pandas plots of distance and speed by age group
    age_means['distance'].plot.bar(ax=ax4, color='red', width=width, position=1)
    age_means['speed'].plot.bar(ax=ax4_2, color='blue', width=width, position=0)

    # Set axis labels
    ax4.set_ylabel('Distance (mi)')
    ax4_2.set_ylabel('Speed (mph)')

    # Set category labels
    ax4.set_xticklabels(['Young', 'Middle Aged', 'Old'], rotation=0)

    # New Figure
    fig5 = plt.figure()

    # Two axes - for distance and speed
    ax5 = fig5.add_subplot(111)
    ax5_2 = ax5.twinx()  # uses same x axis

    # Create pandas plots of distance and speed by gender
    gender_means['distance'].plot.bar(ax=ax5, color='red', width=width, position=1)
    gender_means['speed'].plot.bar(ax=ax5_2, color='blue', width=width, position=0)

    # Set axis labels
    ax5.set_ylabel('Distance (mi)')
    ax5_2.set_ylabel('Speed (mph)')

    # Set category labels
    ax5.set_xticklabels(['Men', 'Women'], rotation=0)

    tq4_2 = time.time()
    print('Time to run question 4: ', round(tq4_2 - tq4_1, 4))

    # plt.show()


### ---------- 5. What is the busiest bike in NYC in 2017? How many times was it used?
###               How many minutes was it in use? ---------- ###

def question5(df):

    tq5_1 = time.time()
    print('Started Question 5')

    # Calculate most common bike
    bikes = (df['Bike ID']).value_counts()

    ## Calculate most used bike
    # Sum up trip duration by bike ID
    biketime = df['Trip Duration'].groupby(df['Bike ID']).sum().reset_index(name='sum')

    biketime = biketime.sort_values(by=['sum'], ascending=False).reset_index(drop=True)

    # Divide by 3600 sec/hr
    biketime['sum'] = biketime['sum'] / 3600


    print('Bike Uses \n', bikes[0:5])
    print('Bike Time (hrs) \n', biketime[0:5])

    # Bike Uses
    # Bike ID   Uses
    # 25738   2514
    # 25275   2409
    # 27161   2376
    # 26565   2370
    # 27111   2349
    #
    # Bike Time(hrs)
    # Bike ID     sum
    # 27076   3309.928611
    # 21523   2957.924722
    # 18251   2893.053333
    # 27556   2879.696667
    # 20999   2782.363611

    ## Plot Results

    # Create Figure and Axes
    fig5 = plt.figure()
    ax5 = fig5.add_subplot(121)
    ax5_2 = fig5.add_subplot(122)

    # Create plots of the two items
    bikes[0:5].plot.bar(ax=ax5, color='blue')
    biketime[0:5].plot.bar(x='Bike ID', y='sum', ax=ax5_2, color='blue')

    # Customize labels
    ax5.set_ylabel('Bike Uses')
    ax5.set_xlabel('Bike ID')
    ax5_2.set_ylabel('Bike Use Time (hr)')
    ax5_2.legend_.remove()

    ax5.set_xticklabels(ax5.get_xticklabels(), rotation=0)
    ax5_2.set_xticklabels(ax5_2.get_xticklabels(), rotation=0)

    tq5_2 = time.time()
    print('Time to run question 5: ', round(tq5_2 - tq5_1, 4))

    # plt.show()


### -------------------------------------------------------------------- ###
### --------------- Create and Evaulate Prediction Model --------------- ###
### -------------------------------------------------------------------- ###

def model1(df_model, f):

    tm = time.time()
    print('Started Model 1')

    # Create Dummy Variables for:
    # Weather Events (0 = no event, 1 = fog, 2 = rain, 3 = thunderstorm, 4 = snow)
    # print(df_model['Event'].unique())
    # df_model['Event'] = df_model['Event'].cat.reorder_categories(['NoEvent', 'Rain', 'Snow'], ordered=True)
    # df_model['Event'] = df_model['Event'].cat.codes

    # Represent start time as number of minutes after midnight
    temp = pd.DatetimeIndex(df_model['Start Time'])
    df_model['Start Time'] = (pd.to_numeric(temp.hour) * 60) + pd.to_numeric(temp.minute)

    # Scale time data
    df_model['Start Time'] = df_model['Start Time'] / 1440  # 1440 minutes in a day = max value of Start Time

    # One-Hot User Type, Gender, and Weather Events (0/1 columns for each category)
    df_model = pd.concat([df_model, pd.get_dummies(df_model['Event'])], axis=1)
    # df_model = pd.concat([df_model, pd.get_dummies(df_model['Gender'])], axis=1)
    df_model = pd.concat([df_model, pd.get_dummies(df_model['User Type'])], axis=1)

    # Drop rows starting and ending at same place
    # First convert to string type for easy comparison
    df_model['Start Station ID'] = df_model['Start Station ID'].astype('str')
    df_model['End Station ID'] = df_model['End Station ID'].astype('str')
    # Then drop rows with matching start and end stations
    df_model = df_model.drop(df_model[df_model['Start Station ID'] == df_model['End Station ID']].index)
    # Then convert back to category
    df_model['Start Station ID'] = df_model['Start Station ID'].astype('category')
    df_model['End Station ID'] = df_model['End Station ID'].astype('category')

    # Drop outliers from trip duration
    df_model = df_model.drop(df_model[df_model['Trip Duration'] > 5000].index)
    # Can drop all values greater than 1000 with no significant impact

    # drop originals from table
    df_model = df_model.drop(columns=['Event', 'User Type', 'Gender', 'Fog', 'NoEvent', 'Rain',
                                      'Snow', 'Thunderstorm'])

    # Simply Scale Age, HighTemp, and Humidity by max
    df_model['Age'] = df_model['Age'] / df_model['Age'].max()
    df_model['HighTemp'] = df_model['HighTemp'] / df_model['HighTemp'].max()
    # df_model['AvgHumidity'] = df_model['AvgHumidity']/df_model['AvgHumidity'].max()

    # Create a column of median trip duration between any two start/stop points in data set
    # avg historic time is likely a good predictor of future travel time
    df_model['AvgTime'] = df_model.groupby(['Start Station ID', 'End Station ID'])['Trip Duration'].transform('median')

    df_model = df_model.drop(columns=['Start Station ID', 'End Station ID', 'Trip', 'Age', 'Start Time'])

    # print(df_model.dtypes, '\n')

    # For good model evaluation and quick iteration, want to select a random sample of available data
    df_model_sample = df_model.sample(frac=f, random_state=0)  # For consistent sampling

    # Predictor Variable
    X = df_model_sample.iloc[:, 1:]
    print(X.dtypes)

    # Target Variable
    y = df_model_sample['Trip Duration']

    # Split into test and train sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)  # set seed for consistent iteration

    # Fit Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr.score(X_test, y_test)

    y_pred = lr.predict(X_test)
    scores = cross_val_score(lr, X, y, cv=3)  # cv is number of partitions
    # returns array of scores from using each partition as testing data once
    # then we avg the scores to get a true estimate of error
    # predictions = cross_val_predict(lr, X, y, cv=3)  # same thing just outputs actual predictions

    # # Takes way too long to run SVR
    # clf = svm.SVR()
    # clf.fit(X, y)
    # clf.predict(X_test)
    # scores = cross_val_score(clf, X, y, cv=3)  # cv is number of partitions

    print('\n All scores: ', scores, '\n')
    print('Average score:', scores.mean(), '\n')

    # All scores: [0.68998182 0.68965507 0.69130367]
    # Average score: 0.6903135198702287

    # # # correlation plot
    # sns.regplot(x='AvgTime', y='Trip Duration', data=df_model_sample)  # strong positive
    # plt.ylim(0, )
    # plt.show()

    tm2 = time.time()
    print('Time to run model 1: ', round(tm2 - tm, 4))


### ------------------------------------------------------------------- ###
### --------------- Preliminary Functions to Create Data--------------- ###
### ------------------------------------------------------------------- ###

# # 1. Call save_data function (just once... it takes about 3 min) to create 12 csv's of the bike data
# save_data()
#
# # 2. Call combine_data function (just once... it takes about 5 min) to edit the january csv
# #   to include all 12 months of data
# combine_data()
#
# # 3. Call manhattan_data function (just once) to download and save a map of Manhattan roads
# manhattan_data()
#
# # 4. Call scrape_weather function (just once) to scrape weather data and save to csv file
# scrape_weather()


### ----------------------------------------------- ###
### --------------- Data Load Script--------------- ###
### ----------------------------------------------- ###

print('Started data load')
# Read in the file to be working with as pandas df
t0 = time.time()
df_bike = pd.read_csv('/Users/kevincrossley/Documents/DevUp/Capstone Data/2017-citibike-tripdata.csv')
                      # nrows=1000000)  # just first 10000 rows for quick testing

t1 = time.time()
print('Time to read in data frame: ', round(t1 - t0, 4))

# Load manhattan map data from disk once its been saved from Open Street Map (manhattan_data())
t2 = time.time()
mgraph = ox.save_load.load_graphml('ManhattanDrive.graphml', folder='/Users/kevincrossley/Documents/DevUp/Capstone Data')
t3 = time.time()
print('Time to load in manhattan map: ', round(t3 - t2, 4))

# Load in weather csv
# Loads weather_3.csv, a file I created using excel from the original, scraped weather.csv file
# It was easier and faster to rearrange columns/ values in excel, so I just did it there
df_weather = pd.read_csv('/Users/kevincrossley/Documents/DevUp/Capstone Data/weather_3.csv', na_values=' null')

### ------------------------------------------------------ ###
### --------------- Clean and Prepare Data --------------- ###
### ------------------------------------------------------ ###

t_clean = time.time()
print('Started Data Preparation')

# Convert start and end station IDs and user type to categorical variables
df_bike['Start Station ID'] = df_bike['Start Station ID'].astype('category')
df_bike['End Station ID'] = df_bike['End Station ID'].astype('category')
df_bike['User Type'] = df_bike['User Type'].astype('category')

# Convert bike time columns to pandas datetime type
df_bike['Start Time'] = pd.to_datetime(df_bike['Start Time'], format='%Y-%m-%d %H:%M:%S')
df_bike['Stop Time'] = pd.to_datetime(df_bike['Stop Time'], format='%Y-%m-%d %H:%M:%S')

# Create column of just the date, not the time
temp = pd.DatetimeIndex(df_bike['Start Time'])
df_bike['Start Date'] = temp.date

# Create set that drops all customers, keeping only subscribers (still 90% of data)
# only subscribers have a birth year
df_bike = df_bike[pd.notnull(df_bike['Birth Year'])]

## Change birth years in the 1800s to 1900s
# create dummy column of every year + 100
df_bike['by2'] = df_bike['Birth Year'] + 100

# create usable column that includes years+100 only for years less than 1900
df_bike['by3'] = np.where(df_bike['Birth Year'] < 1900, df_bike['by2'], df_bike['Birth Year'])

# Create a column of ages
df_bike['Age'] = 2017 - df_bike['by3']

# Drop calculation columns
df_bike = df_bike.drop(columns=['by2', 'by3'])

# Convert weather columns to correct type
df_weather['Date'] = pd.to_datetime(df_weather['Date'], format='%m/%d/%y')
df_weather['HighTemp'] = df_weather['HighTemp'].astype('float64')
df_weather['AvgHumidity'] = df_weather['AvgHumidity'].astype('float64')
df_weather['Event'] = df_weather['Event'].astype('category')

# Make date column just date, not time
temp = pd.DatetimeIndex(df_weather['Date'])
df_weather['Date'] = temp.date

# Create modelling dataframe by merging weather data (high temp, avg humidity, and any rain events) to bike data
df_model = pd.merge(df_bike, df_weather, left_on='Start Date', right_on='Date', how='left')

# Create column representing a trip
df_model['Trip'] = df_model['Start Station ID'].astype('str') + '-' + df_model['End Station ID'].astype('str')
df_model['Trip'] = df_model['Trip'].astype('category')

# Drop irrelevant columns from model df
# Drop Stop Time (same as start time + trip duration), Start/End Station Name, Lat, Long, Bike ID
df_model = df_model.drop(columns=['Start Date', 'Stop Time', 'Start Station Name', 'Start Station Latitude',
                                  'Start Station Longitude', 'End Station Name', 'End Station Latitude',
                                  'Date', 'End Station Longitude', 'Bike ID', 'Birth Year', 'AvgHumidity'])

t_clean2 = time.time()
print('Time to clean and prep data: ', round(t_clean2 - t_clean, 4))

### Other data set preparation occurs as needed in question functions ###


### ------------------------------------------------------------- ###
### --------------- Call Functions to Run Results --------------- ###
### ------------------------------------------------------------- ###

# basic_stats(df_bike)
#
# question1(df_bike, mgraph)
#
# question2(df_bike)
#
# question3(df_bike, mgraph)
#
# question4(df_bike)  # WARNING - My implementation of calculating distance takes EXTREMELY long to run
#
# question5(df_bike)
#
# fraction = 1.0  # fraction of total rows to include in random test sample (to aid run time)
# model1(df_model, fraction)

t5 = time.time()
print('Total run time: ', round(t5 - t4, 4))

