import math
import pandas as pd
import numpy as np

def generate_feature_matrix(raw_data, features):
    X_df = pd.DataFrame()
    if 'Longitude' in features:
        X_df['Longitude'] = raw_data['Longitude']
        
    if 'Latitude' in features:
        X_df['Latitude'] = raw_data['Latitude']
        
    if 'Speed' in features:
        X_df['Speed'] = raw_data['Speed']
        
    if 'Distance' in features:
        X_df['Distance'] = find_BS_distance(raw_data)
        
    if 'Distance_x' in features:
        X_df['Distance_x'] = derive_pos(raw_data,'x')
    
    if 'Distance_y' in features:
        X_df['Distance_y'] = derive_pos(raw_data,'y')
        
    if 'PCI' in features:
        X_df['PCI'] = raw_data['PCI']
        
    # Do one-hot encoding
    for s in features: 
        if 'PCI_' in s:
            X_df[s] = 0
            rows = raw_data.groupby('PCI').get_group(int(s[4:]))
            X_df.loc[rows.index, s] = 1
    
        
    return X_df

def filter_output(raw_data, outputs):
    Y_df = pd.DataFrame()
    
    for output in outputs:
        Y_df[output] = raw_data[output]
        
    return Y_df


def calculate_initial_compass_bearing(pointA, pointB):
    """
    Calculates the bearing between two points.
    The formulae used is the following:
        θ = atan2(sin(Δlong).cos(lat2),
                  cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))
    :Parameters:
      - `pointA: The tuple representing the latitude/longitude for the
        first point. Latitude and longitude must be in decimal degrees
      - `pointB: The tuple representing the latitude/longitude for the
        second point. Latitude and longitude must be in decimal degrees
    :Returns:
      The bearing in degrees
    :Returns Type:
      float
    """
    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])

    diffLong = math.radians(pointB[1] - pointA[1])

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
                                           * math.cos(lat2) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)

    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing

def flatten_list(list_to_flatten):
    flattened_list = []
    for sublist in list_to_flatten:
        for item in sublist:
            flattened_list.append(item)
    return flattened_list

def rotate_point(origio, degree, distance=0.05):
    
    R = 6378.1 #Radius of the Earth
    brng = np.deg2rad(degree) #Bearing is 90 degrees converted to radians.
    d = distance #Distance in km
    
    lat1 = math.radians(origio.lat) #Current lat point converted to radians
    lon1 = math.radians(origio.long) #Current long point converted to radians
    
    lat2 = math.asin( math.sin(lat1)*math.cos(d/R) +
         math.cos(lat1)*math.sin(d/R)*math.cos(brng))
    
    lon2 = lon1 + math.atan2(math.sin(brng)*math.sin(d/R)*math.cos(lat1),
                 math.cos(d/R)-math.sin(lat1)*math.sin(lat2))
    
    lat2 = math.degrees(lat2)
    lon2 = math.degrees(lon2)
    return lat2, lon2
