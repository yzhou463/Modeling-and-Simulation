
'''
# # Input Data Analysis
Author: Junghyun Kim
Created: March 24, 2019
Last modified: April 1, 2019
Affiliation*: 
- Aerospace Systems Design Lab, School of Aerospace Engineering, Georgia Institute of Technology
- Computational Science and Engineering, College of Computing, Georgia Institute of Technology
# ### Library
'''
# In[2]:


# Import all libraries needed for this Jupyter notebook
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ### User-defined

# In[3]:


# Define intersection number
Intersection_number = 5

# Define direction number 
Direction_number = 2  # We assume that there is only north bound direction (number 2)

# Define movement number 
Movement_number = 1  # We assume that there is neither left nor right turn of movement of the vehicle

# Specify intersection locations 
#Intersection_x = 2230553.380  # Convert lat/long to x/y coordinate using http://www.earthpoint.us/stateplane.aspx
#Intersection_y = 1375699.434  # Intersection 1

#Intersection_x = 2230599.801  
#Intersection_y = 1376244.052  # Intersection 2

#Intersection_x = 2230833.626
#Intersection_y = 1376615.802  # Intersection 3

#Intersection_x = 2230873.090  
#Intersection_y = 1377033.444  # Intersection 4

Intersection_x = 2230839.840  
Intersection_y = 1377404.299  # Intersection 5


# ### Data pre-processing
print('Selected analysis direction ID is {} \n'.format(Direction_number))
print('Selected analysis movement ID is {} \n'.format(Movement_number))
print('Selected analysis Intersection of Peachstreet for interarrival time is {} \n'.format(Intersection_number))
print('Waiting for data processing..... \n')


# Import NGSIM traffic data
NGSIM_traffic_data = pd.read_csv('trajectories-0400pm-0415pm_editted.csv')

# Filter the dataset based on the assumptions
NGSIM_traffic_data_filter = NGSIM_traffic_data[(NGSIM_traffic_data['Intersection']==Intersection_number) & (NGSIM_traffic_data['Direction']==Direction_number) & (NGSIM_traffic_data['Movement']==Movement_number)]

# List vehicle IDs based on the assumptions
vehicle_IDs = np.array(NGSIM_traffic_data_filter['Vehicle_ID'].drop_duplicates())  # Remove duplicates

# Define a dictionary to save the dataset
vehicle_level_data = dict()
vehicle_global_y = dict()
vehicle_monitoring_point = dict()
index_number = dict()

# Define an empty array (Otherwise, it's hard to do post-processing with dictionaries)
arrival_time_at_the_intersection = np.zeros(len(vehicle_IDs))
inter_arrival_time = np.zeros(len(vehicle_IDs)-1)
clean_index = np.zeros(len(inter_arrival_time))

# Do for-loop to rearrange the dataset according to each vehicle
for i in range(0,len(vehicle_IDs)):
    
    # Rearrange the data by each vehicle
    vehicle_level_data[i] = NGSIM_traffic_data_filter[(NGSIM_traffic_data_filter['Vehicle_ID']==vehicle_IDs[i])]
    
    # Specify global y data 
    vehicle_global_y[i] = np.array(vehicle_level_data[i]['Global_Y'].round(3))  # 3 floating points based on the dataset

    # Find the closest point to the intersection in y 
    vehicle_monitoring_point[i] = np.around(min(vehicle_global_y[i], key=lambda y:abs(y-Intersection_y)),3)  
    
    # Define the index number to parse time data
    index_number[i] = np.where(vehicle_global_y[i] == vehicle_monitoring_point[i])
    
    # Parse the time data
    arrival_time_at_the_intersection[i] = vehicle_level_data[i]['Epoch_ms'].iloc[index_number[i]]
    
# Sort the arrival_time_at_the_intersection array (Not consistent with data description but do it)
arrival_time_at_the_intersection_sorted = np.sort(arrival_time_at_the_intersection)

# Calculate inter-arrival time between two vehicles (unit: second)
for i in range(0,len(vehicle_IDs)-1):
    inter_arrival_time[i] = 0.001*(arrival_time_at_the_intersection_sorted[i+1] - arrival_time_at_the_intersection_sorted[i])

# Check the outliers (in case that we have negative values)
for i in range(0,len(inter_arrival_time)):
    if inter_arrival_time[i] < 0:
        clean_index[i] = i
        
# Record index for the outliers
outlier_index = np.unique(clean_index)[1:]  # Remove the first element because it's zero, which is not needed for us
outlier_index_int = outlier_index.astype(np.int64)  # Convert values to integer; otherwise, it's defined as a float 64

# Delete the outliers from the array
inter_arrival_time_clean_version_pre = np.delete(inter_arrival_time, outlier_index_int)
inter_arrival_time_clean_version = abs(inter_arrival_time_clean_version_pre)  # Just in case the first element is outlier; it gives 0 but not sorted


# ### CSV export

# In[5]:


# Export inter-arrival time information as csv 
np.savetxt('Inter_arrival_time_at_the_intersection_5.csv', inter_arrival_time_clean_version, delimiter=',')


# ### Probability Density Function
def count_range_in_list(li, minvalue, maxvalue):
	ctr = 0
	for x in li:
		if minvalue <= x < maxvalue:
			ctr += 1
	return ctr

num_interval = int(max(inter_arrival_time_clean_version)/5) + 1
minvalue = 0
maxvalue = 5
length_data = len(inter_arrival_time_clean_version)
for i in range(num_interval):
    prob = round(count_range_in_list(inter_arrival_time_clean_version, minvalue, maxvalue)/length_data, 3)
    print('Probability for interarrival time in interval [{},{}) is:{}\n'.format(minvalue, maxvalue, prob))
    minvalue += 5
    maxvalue += 5
print('You can also find the processed data in Inter_arrival_time_at_the_intersection_{}.csv, thank you!'.format(Intersection_number))


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Created on Tue Apr 23 19:38:38 2019
    
    @author: renwendi
    """

#import matplotlib
import numpy as np
import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt

# Define intersection number
Intersection_number = 1

# Define direction number
Direction_number = 2  # We assume that there is only north bound direction (number 2)

# Define movement number
Movement_number = 1  # We assume that there is neither left nor right turn of movement of the vehicle

NGSIM_traffic_data = pd.read_csv("trajectories-0400pm-0415pm_editted.csv")

# Filter the dataset based on the assumptions
NGSIM_traffic_data_filter = NGSIM_traffic_data[(NGSIM_traffic_data['Intersection']==1) & (NGSIM_traffic_data['Direction']==Direction_number) & (NGSIM_traffic_data['Movement']==Movement_number)]
vehicle_IDs_start = np.array(NGSIM_traffic_data_filter['Vehicle_ID'].drop_duplicates())  # Remove duplicates
NGSIM_traffic_data_filter_end = NGSIM_traffic_data[(NGSIM_traffic_data['Intersection']==5) & (NGSIM_traffic_data['Direction']==Direction_number) & (NGSIM_traffic_data['Movement']==Movement_number)]
vehicle_IDs_end = np.array(NGSIM_traffic_data_filter_end['Vehicle_ID'].drop_duplicates())  # Remove duplicates
same_car = list(set(vehicle_IDs_start) & set(vehicle_IDs_end))
same_car.sort()

start_time = NGSIM_traffic_data_filter["Epoch_ms"].groupby(NGSIM_traffic_data_filter['Vehicle_ID']).min().values
end_time = NGSIM_traffic_data_filter["Epoch_ms"].groupby(NGSIM_traffic_data_filter['Vehicle_ID']).max().values
vehicle_IDs_start = np.array(NGSIM_traffic_data_filter['Vehicle_ID'].drop_duplicates())  # Remove duplicates

totol_travel_time = []
for i in range(len(same_car)):
    select_car = NGSIM_traffic_data[NGSIM_traffic_data["Vehicle_ID"] == same_car[i]]
    travel_time = (select_car["Epoch_ms"].max() - select_car["Epoch_ms"].min()) * 0.001
    totol_travel_time.append(travel_time)
#    if(travel_time < 100):
#        print(same_car[i])

average_travel_time = np.mean(np.array(totol_travel_time))

#plt.figure()
#sns.distplot(np.array(totol_travel_time), color='g')
#plt.xlabel('Total travel time (sec)', fontsize = 14)
#plt.ylabel('Probability Density', fontsize = 14)
#plt.title('NGSIM result', fontsize = 14)
#plt.grid(True)
#plt.savefig('average_travel_time', dpi=150)
#plt.show()
