    
import math
import geopandas as gp
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
from descartes import PolygonPatch
import numpy as np
from utils.tools import flatten_list, rotate_point, calculate_initial_compass_bearing

class NavPoint:

    def __init__(self, coordinates):
        self.lat = coordinates[0]
        self.long = coordinates[1]

    def tolist(self):
        return [self.lat, self.long]


class Routev2:
    def __init__(self):
        self.nav_points = []
        self.route = []
        self.boundary_box = []

    def add_nav_point(self, nav):
        self.nav_points.append(nav)
    
    def get_nav_points(self):
        return self.nav_points

    def add_boundary_box(self, boundary_box):
        self.boundary_box.append(boundary_box)

    def draw_navs(self):
        nav_points = self.get_nav_points()
        color = iter(plt.cm.rainbow(np.linspace(0, 1, len(nav_points))))
        for idx, nav in enumerate(nav_points):
            c = next(color)
            plt.scatter(nav.long, nav.lat, s=100, c=c, label="Nav{}".format(idx))

    def draw_boundary_boxes(self):
        ax = plt.gca()
        for box in self.boundary_box:
            ax.add_patch(PolygonPatch(box[0], alpha=0.2))




def create_test_route():
    route = Routev2()
    nav = NavPoint([55.789091, 12.522860])
    route.add_nav_point(nav)

    nav = NavPoint([55.789320, 12.521457])
    route.add_nav_point(nav)
    
    nav = NavPoint([55.788192, 12.520838])
    route.add_nav_point(nav)
    
    nav = NavPoint([55.789320, 12.521457])
    route.add_nav_point(nav)

    nav = NavPoint([55.789667, 12.519422])
    route.add_nav_point(nav)
    
    nav = NavPoint([55.788577, 12.518768])
    route.add_nav_point(nav)
    
    nav = NavPoint([55.788440, 12.519598])
    route.add_nav_point(nav)

    return route

def create_boundary_box(start, end, box_width):
    bearing = calculate_initial_compass_bearing(start.tolist(),end.tolist())
    theta_x = bearing - 360
    theta_a = 90-(-theta_x)
    
    a_upper_lat, a_upper_long = rotate_point(start, theta_a, distance=box_width)
    a_lower_lat, a_lower_long = rotate_point(start, theta_a-180, distance=box_width)
    b_upper_lat, b_upper_long = rotate_point(end, theta_a, distance=box_width)
    b_lower_lat, b_lower_long = rotate_point(end, theta_a-180, distance=box_width)
    poly = gp.GeoSeries(Polygon([(a_upper_long, a_upper_lat), (a_lower_long, a_lower_lat), (b_lower_long,b_lower_lat), (b_upper_long,b_upper_lat)]))

    return poly

def sort_by_heading(start, end, X_df):
    lat_dir = end.lat - start.lat
    long_dir = end.long - start.long
    
    # Find largest variance of heading
    if math.fabs(lat_dir) > math.fabs(long_dir):

        if lat_dir > 0:
            X_df_sorted = X_df.sort_values('Latitude',ascending=True)
        else:
            X_df_sorted = X_df.sort_values('Latitude',ascending=False)
    else:
        if long_dir > 0:
            X_df_sorted = X_df.sort_values('Longitude',ascending=True)
        else:
            X_df_sorted = X_df.sort_values('Longitude',ascending=False)
            
    return X_df_sorted

def get_samples_test_route(X_df, Y_df, route):
    nav_points = route.get_nav_points()
    ss = gp.GeoSeries([Point(xy) for xy in zip(X_df['Longitude'], X_df['Latitude'])])
    geo_df = gp.GeoDataFrame(X_df, geometry=ss)
    route_index  = list()
    # This loops iterates nav points and applies logic between
    for idx, nav in enumerate(nav_points):
        if idx == len(nav_points)-1:
            break
        start = nav
        end = nav_points[idx+1]
        box_width = 0.02
        
        boundary_box = create_boundary_box(start, end, box_width)
        route.add_boundary_box(boundary_box)
        
        filtered_geo_df = geo_df.loc[geo_df.geometry.intersects(boundary_box.ix[0])]
        filtered_geo_df = sort_by_heading(start,end,filtered_geo_df)
        
        # Store indicies
        route_index.append(filtered_geo_df.index)
        
    route_index_flattened = flatten_list(route_index)
        
    X_df_test = X_df.loc[route_index_flattened]
    Y_df_test = Y_df.loc[route_index_flattened]
    return X_df_test, Y_df_test




def get_training_test_data(X_df, Y_df, draw=False):
    
    route = create_test_route()
    X_df_test, Y_df_test = get_samples_test_route(X_df, Y_df, route)

    X_df_train = X_df.drop(X_df_test.index, axis=0)
    Y_df_train = Y_df.drop(Y_df_test.index, axis=0)

    if draw:
        fig = plt.figure(figsize=(15, 20))
        ax = plt.subplot(2,1,1)
        plt.scatter(X_df_train['Longitude'], X_df_train['Latitude'])
        plt.scatter(X_df_test['Longitude'], X_df_test['Latitude'])
        route.draw_navs()
        route.draw_boundary_boxes()
        
        pcis = X_df.groupby('PCI')
        
    

        for idx, output in enumerate(Y_df.columns.tolist()):
            fig = plt.figure(figsize=(10, 10))
            ax = plt.subplot(4,1,idx+1)
            output_plt = list()
            for pci in pcis.groups.keys():
                X_data = X_df_test.loc[X_df_test['PCI'] == pci]
                Y_data = Y_df_test.loc[X_df_test['PCI'] == pci]
                x_axis = range(0,len(X_data))
                output_plt.append(plt.plot(x_axis, Y_data[output], label="PCI: {}".format(pci)))

            ax.legend(handles=[plt[0] for plt in output_plt])
            ax.set_xlabel("# Measurement")
            ax.set_ylabel(output)
        
        plt.show()
                
    X_df_train = X_df_train.drop('geometry',axis=1)
    X_df_test = X_df_test.drop('geometry',axis=1)

    return X_df_train, Y_df_train, X_df_test, Y_df_test

