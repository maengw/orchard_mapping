#!/usr/bin/env python

import pandas as pd
import csv
import matplotlib.pyplot as plt


if __name__ == "__main__":
    csv_file = '/home/woo/catkin_ws/src/orchard_mapping/data/vehicle_gps.csv'
    column_names = ['vehicle_gps_x', 'vehicle_gps_y']
    df = pd.read_csv(csv_file)

    x = df.vehicle_gps_x.tolist()
    y = df.vehicle_gps_y.tolist()

    gpsmap = plt.scatter(x, y, s=3, marker='o', color='red', label='vehicle gps')
    plt.legend(loc='lower right')
    plt.title('vehicle trajectory in orchard')
    plt.xlabel('latitude')
    plt.ylabel('longitude')
    plt.show()

