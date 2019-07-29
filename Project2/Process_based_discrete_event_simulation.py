#!/usr/bin/env python
# coding: utf-8

# # Process-based Discrete Event Simulation
#Author: Junghyun Kim
#Created: March 28, 2019
#Last modified: April 21, 2019
#Affiliation: 
#- Aerospace Systems Design Lab, School of Aerospace Engineering, Georgia Institute of #Technology
#- Computational Science and Engineering, College of Computing, Georgia Institute of #Technology
# ### Library

# In[20]:


# Import all libraries needed for this Jupyter notebook
import heapq
import time 
#import matplotlib
import numpy as np
import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt


# ### Input empirical distribution

# In[3]:


# Define intersection number
Intersection_number = 1  # We assume that all vehicles start from intersection 1

# Define direction number 
Direction_number = 2  # We assume that there is only north bound direction (number 2)

# Define movement number 
Movement_number = 1  # We assume that there is neither left nor right turn of movement of the vehicle

# Specify intersection 1 locations 
Intersection_x = 2230553.380  # Convert lat/long to x/y coordinate using http://www.earthpoint.us/stateplane.aspx
Intersection_y = 1375699.434  


# In[4]:


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

# Finalize the input empirical distribution with a list structure
input_empirical_distribution = inter_arrival_time_clean_version.tolist()


# ### Vehicle

# In[5]:


# Note that you can choose the following distributions:
# 1) Empirical distribution obtained from NGSIM data
# 2) Random number distribution 
# 3) Unifrom distribution


# In[6]:


# Define a heap structure for a priority queue purpose
def heapsort(iterable):
    h = []
    for value in iterable:
        heapq.heappush(h, value)
    return [heapq.heappop(h) for i in range(len(h))]


# In[7]:


# Create a queue for vehicle events using the input empirical distribution
vehicle_event_queue = np.zeros(len(inter_arrival_time_clean_version)+1)  # +1 is needed because we specify the first element
vehicle_event_queue[0] = 0
for i in range(0,len(inter_arrival_time_clean_version)):
    vehicle_event_queue[i+1] = np.round((vehicle_event_queue[i] + input_empirical_distribution[i]),1)


# In[8]:


# Create a queue for vehicle events using uniform distribution
#vehicle_event_queue = np.round((np.linspace(0,900,num=90)),1)


# In[9]:


# Create a queue for vehicle events using random number generator
#number_of_vehicles = 90
#random_number_generator = np.round((np.random.uniform(low=0, high=900, size=number_of_vehicles)),1)
#vehicle_event_queue = np.array(heapsort(random_number_generator))


# ### Traffic light

# In[10]:


# Define my time 
start_time = 0.0  
end_time = 1200.0  # Note that NGSIM collected vehicle trajectory data between 16:00 and 16:15 (During 900 sec)
time_interval = 0.1  # Unit: second
my_time = np.arange(start_time, end_time, time_interval)


# In[11]:


# Define signal timing (Note that the street doesn't have traffic light at the intersection 4)
intersection_1_northbound_signal = [34.7, 49.3]  # [Green through, Red through]
intersection_2_northbound_signal = [41.5, 55.4]
intersection_3_northbound_signal = [60.9, 35.7]
intersection_5_northbound_signal = [34.6, 46.1]


# In[12]:


# Define a function to creat a time-series green/red signal list for each intersection (Assume that all intersections start with green in the simulation)
def traffic_light(signal):
    intersection_signal = np.array(signal)
    intersection_signal_tile = np.tile(intersection_signal, int(end_time/(intersection_signal[0]+intersection_signal[1])))
    intersection_time = np.add.accumulate(intersection_signal_tile)
    intersection_thread = np.array((['G']*int(signal[0]*10) + ['R']*int(signal[1]*10))*int(end_time/(intersection_signal[0]+intersection_signal[1])+1))[0:len(my_time)]
    intersection_thread_list = intersection_thread.tolist()
    return intersection_thread_list


# In[13]:


# Generate time-series green/red signal lists for each intersection
intersection_1_northbound_traffic_light = traffic_light(intersection_1_northbound_signal)
intersection_2_northbound_traffic_light = traffic_light(intersection_2_northbound_signal)
intersection_3_northbound_traffic_light = traffic_light(intersection_3_northbound_signal)
intersection_5_northbound_traffic_light = traffic_light(intersection_5_northbound_signal)


# ### Process-based simulation

# In[14]:


# Define average velocity at each section (These are calculated based on NGSIM dataset)
section_2_average_velocity = 17.61  # unit: ft/s
section_3_average_velocity = 13.52
section_4_average_velocity = 32.19
section_5_average_velocity = 8.31

# Define average velocity at each intersection (These are calculated based on NGSIM dataset)
intersection_1_average_velocity = 28.95   # unit: ft/s
intersection_2_average_velocity = 20.38  
intersection_3_average_velocity = 14.87
intersection_4_average_velocity = 32.53
intersection_5_average_velocity = 18

# Define section lengths (These are from the reference document)
section_2_length = 417.976  # unit: ft 
section_3_length = 412.172
section_4_length = 351.511
section_5_length = 344.427

# Define intersection lengths (These are from the reference document)
intersection_1_length = 99.732  # unit: ft 
intersection_2_length = 129.875
intersection_3_length = 73.513
intersection_4_length = 66.602
intersection_5_length = 121.319


# In[15]:


# Create a queue to deal with intersections (Note that there is no traffic light at intersection 4)
intersection_1_queue = []
intersection_2_queue = []
intersection_3_queue = []
intersection_5_queue = []
intersection_1_stop_waiting_time_queue = []
intersection_2_stop_waiting_time_queue = []
intersection_3_stop_waiting_time_queue = []
intersection_5_stop_waiting_time_queue = []

# Specify time difference between two vehicles at each intersection (Calculated based on NGSIM data)
intersection_1_time_difference_between_two_vehicles = 3.2
intersection_2_time_difference_between_two_vehicles = 0.9
intersection_3_time_difference_between_two_vehicles = 1.2
intersection_5_time_difference_between_two_vehicles = 2.0


# In[16]:


# Define a function to process the simulation 
def simulation_process(vehicle_event):
    
    # ==================== Simulation starts ~ Intersection 2 ==================
    # Specify departure time of the vehicle (Assume that all vehicles depart from before-intersection 1)
    vehicle_start_time = vehicle_event
    
    # Match the time to the index of traffic light at intersection 1
    vehicle_intersection_1_traffic_light_index = int(vehicle_start_time * 10)
    
    # Specify traffic light signal of intersection 1 at the timestamp 
    intersection_1_traffic_light_signal = intersection_1_northbound_traffic_light[vehicle_intersection_1_traffic_light_index]

    # Calculate next timestamp of the vehicle (Check if either signal at intersection 1 is Green or Red)
    if intersection_1_traffic_light_signal == 'G':
        vehicle_intersection_2_time = vehicle_start_time + (intersection_1_length/intersection_1_average_velocity) + (section_2_length/section_2_average_velocity)
    elif intersection_1_traffic_light_signal == 'R':
        
        # Create a queue and append the time-stamp information
        intersection_1_queue.append(vehicle_start_time)
        
        # Do for-loop to calculate waiting time at stop of the intersection 1
        for stop_waiting_time in range (vehicle_intersection_1_traffic_light_index, len(intersection_1_northbound_traffic_light)):
            if intersection_1_northbound_traffic_light[stop_waiting_time] == 'G':
                break
        intersection_1_stop_waiting_time_queue.append(stop_waiting_time)
        vehicle_intersection_1_stop_waiting_time = stop_waiting_time/10 - vehicle_start_time  
        
        # Create a list to count how many vehicles are located in front of the current vehicle
        intersection_1_vehicle_count = []
        
        # Define global varialbe for intersection vehicle difference time (Otherwise, you will get unbound local error)
        #global intersection_1_vehicle_difference_time
        
        # Perform conditional statements to make sure that the vehicles are in a line when they are stuck in Red traffic light
        if len(intersection_1_stop_waiting_time_queue) == 1:
            for i in range(0,len(intersection_1_queue)):  
                if int(intersection_1_queue[i]*10) < stop_waiting_time:
                    intersection_1_vehicle_count.append(intersection_1_queue[i])
                    intersection_1_vehicle_difference_time = (len(intersection_1_vehicle_count)-1) * intersection_1_time_difference_between_two_vehicles  # -1 is needed because the vehicle doesn't count itself
        else:
            if intersection_1_stop_waiting_time_queue[len(intersection_1_stop_waiting_time_queue)-1] == intersection_1_stop_waiting_time_queue[len(intersection_1_stop_waiting_time_queue)-2]:
                if intersection_1_stop_waiting_time_queue[0] == intersection_1_stop_waiting_time_queue[len(intersection_1_stop_waiting_time_queue)-1]: 
                    for i in range(0,len(intersection_1_queue)):  
                        if int(intersection_1_queue[i]*10) < stop_waiting_time:
                            intersection_1_vehicle_count.append(intersection_1_queue[i])
                            intersection_1_vehicle_difference_time = (len(intersection_1_vehicle_count)-1) * intersection_1_time_difference_between_two_vehicles
       
                elif intersection_1_stop_waiting_time_queue[0] != intersection_1_stop_waiting_time_queue[len(intersection_1_stop_waiting_time_queue)-1]: 
                    for i in range(0,len(intersection_1_stop_waiting_time_queue)-1):
                        if intersection_1_stop_waiting_time_queue[i] != intersection_1_stop_waiting_time_queue[i+1]:
                            intersection_1_cutting_index = i

                    for i in range(intersection_1_cutting_index+1,len(intersection_1_queue)):
                        if int(intersection_1_queue[i]*10) < stop_waiting_time:
                            intersection_1_vehicle_count.append(intersection_1_queue[i])
                            intersection_1_vehicle_difference_time = (len(intersection_1_vehicle_count)-1) * intersection_1_time_difference_between_two_vehicles
    
            elif intersection_1_stop_waiting_time_queue[len(intersection_1_stop_waiting_time_queue)-1] != intersection_1_stop_waiting_time_queue[len(intersection_1_stop_waiting_time_queue)-2]:
                for i in range(0,len(intersection_1_queue)):  
                    if int(intersection_1_queue[i]*10) < stop_waiting_time and int(intersection_1_queue[i]*10) > intersection_1_stop_waiting_time_queue[len(intersection_1_stop_waiting_time_queue)-2]:  # -2 means looking at the previous step
                        intersection_1_vehicle_count.append(intersection_1_queue[i])
                        intersection_1_vehicle_difference_time = (len(intersection_1_vehicle_count)-1) * intersection_1_time_difference_between_two_vehicles 
        
        # Calculate arrival time of the vehicle at intersection 2
        vehicle_intersection_2_time = vehicle_start_time + vehicle_intersection_1_stop_waiting_time + intersection_1_vehicle_difference_time + (intersection_1_length/intersection_1_average_velocity) + (section_2_length/section_2_average_velocity)
    
    # ==================== Intersection 2 ~ Intersection 3 ====================
    # Match the time to the index of traffic light at intersection 2
    vehicle_intersection_2_traffic_light_index = int(vehicle_intersection_2_time * 10)
    
    # Specify traffic light signal of intersection 2 at the timestamp 
    intersection_2_traffic_light_signal = intersection_2_northbound_traffic_light[vehicle_intersection_2_traffic_light_index]
    
    # Calculate next timestamp of the vehicle (Check if either signal at intersection 2 is Green or Red)
    if intersection_2_traffic_light_signal == 'G':
        vehicle_intersection_3_time = vehicle_intersection_2_time + (intersection_2_length/intersection_2_average_velocity) + (section_3_length/section_3_average_velocity)
    elif intersection_2_traffic_light_signal == 'R':
        
        # Create a queue and append the time-stamp information
        intersection_2_queue.append(vehicle_intersection_2_time)
        
        # Do for-loop to calculate waiting time at stop of the intersection 2
        for stop_waiting_time in range (vehicle_intersection_2_traffic_light_index, len(intersection_2_northbound_traffic_light)):
            if intersection_2_northbound_traffic_light[stop_waiting_time] == 'G':
                break
        intersection_2_stop_waiting_time_queue.append(stop_waiting_time)
        vehicle_intersection_2_stop_waiting_time = stop_waiting_time/10 - vehicle_intersection_2_time  
        
         # Create a list to count how many vehicles are located in front of the current vehicle
        intersection_2_vehicle_count = []
        
        # Define global varialbe for intersection vehicle difference time (Otherwise, you will get unbound local error)
        global intersection_2_vehicle_difference_time
        
        # Perform conditional statements to make sure that the vehicles are in a line when they are stuck in Red traffic light
        if len(intersection_2_stop_waiting_time_queue) == 1:
            for i in range(0,len(intersection_2_queue)):  
                if int(intersection_2_queue[i]*10) < stop_waiting_time:
                    intersection_2_vehicle_count.append(intersection_2_queue[i])
                    intersection_2_vehicle_difference_time = (len(intersection_2_vehicle_count)-1) * intersection_2_time_difference_between_two_vehicles  # -1 is needed because the vehicle doesn't count itself
        else:
            if intersection_2_stop_waiting_time_queue[len(intersection_2_stop_waiting_time_queue)-1] == intersection_2_stop_waiting_time_queue[len(intersection_2_stop_waiting_time_queue)-2]:
                if intersection_2_stop_waiting_time_queue[0] == intersection_2_stop_waiting_time_queue[len(intersection_2_stop_waiting_time_queue)-1]: 
                    for i in range(0,len(intersection_2_queue)):  
                        if int(intersection_2_queue[i]*10) < stop_waiting_time:
                            intersection_2_vehicle_count.append(intersection_2_queue[i])
                            intersection_2_vehicle_difference_time = (len(intersection_2_vehicle_count)-1) * intersection_2_time_difference_between_two_vehicles
       
                elif intersection_2_stop_waiting_time_queue[0] != intersection_2_stop_waiting_time_queue[len(intersection_2_stop_waiting_time_queue)-1]: 
                    for i in range(0,len(intersection_2_stop_waiting_time_queue)-1):
                        if intersection_2_stop_waiting_time_queue[i] != intersection_2_stop_waiting_time_queue[i+1]:
                            intersection_2_cutting_index = i

                    for i in range(intersection_2_cutting_index+1,len(intersection_2_queue)):
                        if int(intersection_2_queue[i]*10) < stop_waiting_time:
                            intersection_2_vehicle_count.append(intersection_2_queue[i])
                            intersection_2_vehicle_difference_time = (len(intersection_2_vehicle_count)-1) * intersection_2_time_difference_between_two_vehicles
    
            elif intersection_2_stop_waiting_time_queue[len(intersection_2_stop_waiting_time_queue)-1] != intersection_2_stop_waiting_time_queue[len(intersection_2_stop_waiting_time_queue)-2]:
                for i in range(0,len(intersection_2_queue)):  
                    if int(intersection_2_queue[i]*10) < stop_waiting_time and int(intersection_2_queue[i]*10) > intersection_2_stop_waiting_time_queue[len(intersection_2_stop_waiting_time_queue)-2]:  # -2 means looking at the previous step
                        intersection_2_vehicle_count.append(intersection_2_queue[i])
                        intersection_2_vehicle_difference_time = (len(intersection_2_vehicle_count)-1) * intersection_2_time_difference_between_two_vehicles 
        
        # Calculate arrival time of the vehicle at intersection 3
        vehicle_intersection_3_time = vehicle_intersection_2_time + vehicle_intersection_2_stop_waiting_time + intersection_2_vehicle_difference_time + (intersection_2_length/intersection_2_average_velocity) + (section_3_length/section_3_average_velocity)
    
    # ==================== Intersection 3 ~ Intersection 4 ====================
    # Match the time to the index of traffic light at intersection 3
    vehicle_intersection_3_traffic_light_index = int(vehicle_intersection_3_time * 10)
    
    # Specify traffic light signal of intersection 3 at the timestamp 
    intersection_3_traffic_light_signal = intersection_3_northbound_traffic_light[vehicle_intersection_3_traffic_light_index]
    
    # Calculate next timestamp of the vehicle (Check if either signal at intersection 3 is Green or Red)
    if intersection_3_traffic_light_signal == 'G':
        vehicle_intersection_4_time = vehicle_intersection_3_time + (intersection_3_length/intersection_3_average_velocity) + (section_4_length/section_4_average_velocity)
    elif intersection_3_traffic_light_signal == 'R':
        
        # Create a queue and append the time-stamp information
        intersection_3_queue.append(vehicle_intersection_3_time)
        
        # Do for-loop to calculate waiting time at stop of the intersection 3
        for stop_waiting_time in range (vehicle_intersection_3_traffic_light_index, len(intersection_3_northbound_traffic_light)):
            if intersection_3_northbound_traffic_light[stop_waiting_time] == 'G':
                break
        intersection_3_stop_waiting_time_queue.append(stop_waiting_time)
        vehicle_intersection_3_stop_waiting_time = stop_waiting_time/10 - vehicle_intersection_3_time  
        
         # Create a list to count how many vehicles are located in front of the current vehicle
        intersection_3_vehicle_count = []
        
        # Define global varialbe for intersection vehicle difference time (Otherwise, you will get unbound local error)
        global intersection_3_vehicle_difference_time
        
        # Perform conditional statements to make sure that the vehicles are in a line when they are stuck in Red traffic light
        if len(intersection_3_stop_waiting_time_queue) == 1:
            for i in range(0,len(intersection_3_queue)):  
                if int(intersection_3_queue[i]*10) < stop_waiting_time:
                    intersection_3_vehicle_count.append(intersection_3_queue[i])
                    intersection_3_vehicle_difference_time = (len(intersection_3_vehicle_count)-1) * intersection_3_time_difference_between_two_vehicles  # -1 is needed because the vehicle doesn't count itself
        else:
            if intersection_3_stop_waiting_time_queue[len(intersection_3_stop_waiting_time_queue)-1] == intersection_3_stop_waiting_time_queue[len(intersection_3_stop_waiting_time_queue)-2]:
                if intersection_3_stop_waiting_time_queue[0] == intersection_3_stop_waiting_time_queue[len(intersection_3_stop_waiting_time_queue)-1]: 
                    for i in range(0,len(intersection_3_queue)):  
                        if int(intersection_3_queue[i]*10) < stop_waiting_time:
                            intersection_3_vehicle_count.append(intersection_3_queue[i])
                            intersection_3_vehicle_difference_time = (len(intersection_3_vehicle_count)-1) * intersection_3_time_difference_between_two_vehicles
       
                elif intersection_3_stop_waiting_time_queue[0] != intersection_3_stop_waiting_time_queue[len(intersection_3_stop_waiting_time_queue)-1]: 
                    for i in range(0,len(intersection_3_stop_waiting_time_queue)-1):
                        if intersection_3_stop_waiting_time_queue[i] != intersection_3_stop_waiting_time_queue[i+1]:
                            intersection_3_cutting_index = i

                    for i in range(intersection_3_cutting_index+1,len(intersection_3_queue)):
                        if int(intersection_3_queue[i]*10) < stop_waiting_time:
                            intersection_3_vehicle_count.append(intersection_3_queue[i])
                            intersection_3_vehicle_difference_time = (len(intersection_3_vehicle_count)-1) * intersection_3_time_difference_between_two_vehicles
    
            elif intersection_3_stop_waiting_time_queue[len(intersection_3_stop_waiting_time_queue)-1] != intersection_3_stop_waiting_time_queue[len(intersection_3_stop_waiting_time_queue)-2]:
                for i in range(0,len(intersection_3_queue)):  
                    if int(intersection_3_queue[i]*10) < stop_waiting_time and int(intersection_3_queue[i]*10) > intersection_3_stop_waiting_time_queue[len(intersection_3_stop_waiting_time_queue)-2]:  # -2 means looking at the previous step
                        intersection_3_vehicle_count.append(intersection_3_queue[i])
                        intersection_3_vehicle_difference_time = (len(intersection_3_vehicle_count)-1) * intersection_3_time_difference_between_two_vehicles 
        
        # Calculate arrival time of the vehicle at intersection 4
        vehicle_intersection_4_time = vehicle_intersection_3_time + vehicle_intersection_3_stop_waiting_time + intersection_3_vehicle_difference_time + (intersection_3_length/intersection_3_average_velocity) + (section_4_length/section_4_average_velocity)
    
    # ==================== Intersection 4 ~ Intersection 5 ====================
    # Calculate next timestamp of the vehicle (Note that there is no traffic light at intersection 4)
    vehicle_intersection_5_time = vehicle_intersection_4_time + (intersection_4_length/intersection_4_average_velocity) + (section_5_length/section_5_average_velocity)
 
    # ==================== Intersection 5 ~ Simulation ends ===================
    # Match the time to the index of traffic light at intersection 5
    vehicle_intersection_5_traffic_light_index = int(vehicle_intersection_5_time * 10)
    
    # Specify traffic light signal of intersection 5 at the timestamp 
    intersection_5_traffic_light_signal = intersection_5_northbound_traffic_light[vehicle_intersection_5_traffic_light_index]
    
    # Calculate next timestamp of the vehicle (Check if either signal at intersection 5 is Green or Red)
    if intersection_5_traffic_light_signal == 'G':
        vehicle_end_time = vehicle_intersection_5_time + (intersection_5_length/intersection_5_average_velocity)  # Assume that the simulation ends after intersection 5
    elif intersection_5_traffic_light_signal == 'R':
        
        # Create a queue and append the time-stamp information
        intersection_5_queue.append(vehicle_intersection_5_time)
        
        # Do for-loop to calculate waiting time at stop of the intersection 2
        for stop_waiting_time in range (vehicle_intersection_5_traffic_light_index, len(intersection_5_northbound_traffic_light)):
            if intersection_5_northbound_traffic_light[stop_waiting_time] == 'G':
                break
        intersection_5_stop_waiting_time_queue.append(stop_waiting_time)
        vehicle_intersection_5_stop_waiting_time = stop_waiting_time/10 - vehicle_intersection_5_time  
        
         # Create a list to count how many vehicles are located in front of the current vehicle
        intersection_5_vehicle_count = []
        
        # Define global varialbe for intersection vehicle difference time (Otherwise, you will get unbound local error)
        global intersection_5_vehicle_difference_time
        
        # Perform conditional statements to make sure that the vehicles are in a line when they are stuck in Red traffic light
        if len(intersection_5_stop_waiting_time_queue) == 1:
            for i in range(0,len(intersection_5_queue)):  
                if int(intersection_5_queue[i]*10) < stop_waiting_time:
                    intersection_5_vehicle_count.append(intersection_5_queue[i])
                    intersection_5_vehicle_difference_time = (len(intersection_5_vehicle_count)-1) * intersection_5_time_difference_between_two_vehicles  # -1 is needed because the vehicle doesn't count itself
        else:
            if intersection_5_stop_waiting_time_queue[len(intersection_5_stop_waiting_time_queue)-1] == intersection_5_stop_waiting_time_queue[len(intersection_5_stop_waiting_time_queue)-2]:
                if intersection_5_stop_waiting_time_queue[0] == intersection_5_stop_waiting_time_queue[len(intersection_5_stop_waiting_time_queue)-1]: 
                    for i in range(0,len(intersection_5_queue)):  
                        if int(intersection_5_queue[i]*10) < stop_waiting_time:
                            intersection_5_vehicle_count.append(intersection_5_queue[i])
                            intersection_5_vehicle_difference_time = (len(intersection_5_vehicle_count)-1) * intersection_5_time_difference_between_two_vehicles
       
                elif intersection_5_stop_waiting_time_queue[0] != intersection_5_stop_waiting_time_queue[len(intersection_5_stop_waiting_time_queue)-1]: 
                    for i in range(0,len(intersection_5_stop_waiting_time_queue)-1):
                        if intersection_5_stop_waiting_time_queue[i] != intersection_5_stop_waiting_time_queue[i+1]:
                            intersection_5_cutting_index = i

                    for i in range(intersection_5_cutting_index+1,len(intersection_5_queue)):
                        if int(intersection_5_queue[i]*10) < stop_waiting_time:
                            intersection_5_vehicle_count.append(intersection_5_queue[i])
                            intersection_5_vehicle_difference_time = (len(intersection_5_vehicle_count)-1) * intersection_5_time_difference_between_two_vehicles
    
            elif intersection_5_stop_waiting_time_queue[len(intersection_5_stop_waiting_time_queue)-1] != intersection_5_stop_waiting_time_queue[len(intersection_5_stop_waiting_time_queue)-2]:
                for i in range(0,len(intersection_5_queue)):  
                    if int(intersection_5_queue[i]*10) < stop_waiting_time and int(intersection_5_queue[i]*10) > intersection_5_stop_waiting_time_queue[len(intersection_5_stop_waiting_time_queue)-2]:  # -2 means looking at the previous step
                        intersection_5_vehicle_count.append(intersection_5_queue[i])
                        intersection_5_vehicle_difference_time = (len(intersection_5_vehicle_count)-1) * intersection_5_time_difference_between_two_vehicles 
        
        # Calculate vehicle end time
        vehicle_end_time = vehicle_intersection_5_time + vehicle_intersection_5_stop_waiting_time + intersection_5_vehicle_difference_time + (intersection_5_length/intersection_5_average_velocity)
    
    return vehicle_start_time, vehicle_intersection_2_time, vehicle_intersection_3_time, vehicle_intersection_4_time, vehicle_intersection_5_time, vehicle_end_time


# ### Simulation results

# In[17]:


# Create a dictionary to save simulation results 
simulation_results = dict() 

# Do for-loop to generate simulation results through the function for each vehicle 
for i in range(0,len(vehicle_event_queue)):
    simulation_results[i] = simulation_process(vehicle_event_queue[i])

# Do for-loop to convert the dictionary to an array 
simulation_results_array = np.array(list(simulation_results.values()))

# Create a dictionary to specify simulated inter-arrival results according to each intersection
simulation_results_intersection = dict()
simulation_inter_arrival_time_intersection = dict()

# Specify simulation results according to an intersection
simulation_results_intersection_1 = simulation_results_array[:,0]
simulation_results_intersection_2 = simulation_results_array[:,1]
simulation_results_intersection_3 = simulation_results_array[:,2]
simulation_results_intersection_4 = simulation_results_array[:,3]
simulation_results_intersection_5 = simulation_results_array[:,4]

# Creat an array to save simulation inter-arrival time results at each intersection
simulation_inter_arrival_time_intersection_1 = np.zeros(len(vehicle_event_queue)-1)
simulation_inter_arrival_time_intersection_2 = np.zeros(len(vehicle_event_queue)-1)
simulation_inter_arrival_time_intersection_3 = np.zeros(len(vehicle_event_queue)-1)
simulation_inter_arrival_time_intersection_4 = np.zeros(len(vehicle_event_queue)-1)
simulation_inter_arrival_time_intersection_5 = np.zeros(len(vehicle_event_queue)-1)

# Do for-loop to generate simulated inter-arrival results at all intersections
## Intersection 1
for i in range(0,len(vehicle_event_queue)-1):
    simulation_inter_arrival_time_intersection_1[i] = np.absolute(simulation_results_intersection_1[i+1] - simulation_results_intersection_1[i])
## Intersection 2
for i in range(0,len(vehicle_event_queue)-1):
    simulation_inter_arrival_time_intersection_2[i] = np.absolute(simulation_results_intersection_2[i+1] - simulation_results_intersection_2[i])
## Intersection 3
for i in range(0,len(vehicle_event_queue)-1):
    simulation_inter_arrival_time_intersection_3[i] = np.absolute(simulation_results_intersection_3[i+1] - simulation_results_intersection_3[i])
## Intersection 4
for i in range(0,len(vehicle_event_queue)-1):
    simulation_inter_arrival_time_intersection_4[i] = np.absolute(simulation_results_intersection_4[i+1] - simulation_results_intersection_4[i])
## Intersection 5
for i in range(0,len(vehicle_event_queue)-1):
    simulation_inter_arrival_time_intersection_5[i] = np.absolute(simulation_results_intersection_5[i+1] - simulation_results_intersection_5[i])


# ### Comparison between NGSIM and Process-based Simulation

# In[17]:


# Plot PDF for simulated inter-arrival time results at all intersections
# sns.distplot(simulation_inter_arrival_time_intersection_1)
# plt.xlabel('Inter-Arrival time (sec)', fontsize = 12)
# plt.ylabel('Probability Density', fontsize = 12)
# plt.title('Simulation result - Intersection 1', fontsize = 12)
# plt.grid(True)
# plt.savefig('Simulation_Result_Intersection_1', dpi=150)


# In[18]:


# Import NGSIM traffic data with the assumptions
## Note that these results were created by the Python script for input analysis
#NGSIM_inter_arrival_time_intersection_1 = pd.read_csv('Inter_arrival_time_at_the_intersection_1.csv', header=None)

# Plot PDF for simulated inter-arrival time results at all intersections
# sns.distplot(NGSIM_inter_arrival_time_intersection_1, color='g')
# plt.xlabel('Inter-Arrival time (sec)', fontsize = 12)
# plt.ylabel('Probability Density', fontsize = 12)
# plt.title('NGSIM result - Intersection 1', fontsize = 12)
# plt.grid(True)
# plt.savefig('NGSIM_Result_Intersection_1', dpi=150)


# In[19]:


# Plot PDF for simulated inter-arrival time results at all intersections
# sns.distplot(simulation_inter_arrival_time_intersection_2)
# plt.xlabel('Inter-Arrival time (sec)', fontsize = 12)
# plt.ylabel('Probability Density', fontsize = 12)
# plt.title('Simulation result - Intersection 2', fontsize = 12)
# plt.grid(True)
# plt.savefig('Simulation_Result_Intersection_2', dpi=150)


# In[20]:


# Import NGSIM traffic data with the assumptions
## Note that these results were created by the Python script for input analysis
#
# Plot PDF for simulated inter-arrival time results at all intersections
# sns.distplot(NGSIM_inter_arrival_time_intersection_2, color='g')
# plt.xlabel('Inter-Arrival time (sec)', fontsize = 12)
# plt.ylabel('Probability Density', fontsize = 12)
# plt.title('NGSIM result - Intersection 2', fontsize = 12)
# plt.grid(True)
# plt.savefig('NGSIM_Result_Intersection_2', dpi=150)


# In[21]:


# Plot PDF for simulated inter-arrival time results at all intersections
# sns.distplot(simulation_inter_arrival_time_intersection_3)
# plt.xlabel('Inter-Arrival time (sec)', fontsize = 12)
# plt.ylabel('Probability Density', fontsize = 12)
# plt.title('Simulation result - Intersection 3', fontsize = 12)
# plt.grid(True)
# plt.savefig('Simulation_Result_Intersection_3', dpi=150)


# In[22]:


# Import NGSIM traffic data with the assumptions
## Note that these results were created by the Python script for input analysis
#
# Plot PDF for simulated inter-arrival time results at all intersections
# sns.distplot(NGSIM_inter_arrival_time_intersection_3, color='g')
# plt.xlabel('Inter-Arrival time (sec)', fontsize = 12)
# plt.ylabel('Probability Density', fontsize = 12)
# plt.title('NGSIM result - Intersection 3', fontsize = 12)
# plt.grid(True)
# plt.savefig('NGSIM_Result_Intersection_3', dpi=150)


# In[23]:


# Plot PDF for simulated inter-arrival time results at all intersections
# sns.distplot(simulation_inter_arrival_time_intersection_5)
# plt.xlabel('Inter-Arrival time (sec)', fontsize = 12)
# plt.ylabel('Probability Density', fontsize = 12)
# plt.title('Simulation result - Intersection 5', fontsize = 12)
# plt.grid(True)
# plt.savefig('Simulation_Result_Intersection_5', dpi=150)


# In[24]:


# Import NGSIM traffic data with the assumptions
## Note that these results were created by the Python script for input analysis
#NGSIM_inter_arrival_time_intersection_5 = pd.read_csv('Inter_arrival_time_at_the_intersection_5.csv', header=None)

# Plot PDF for simulated inter-arrival time results at all intersections
# sns.distplot(NGSIM_inter_arrival_time_intersection_2, color='g')
# plt.xlabel('Inter-Arrival time (sec)', fontsize = 12)
# plt.ylabel('Probability Density', fontsize = 12)
# plt.title('NGSIM result - Intersection 5', fontsize = 12)
# plt.grid(True)
# plt.savefig('NGSIM_Result_Intersection_5', dpi=150)


# ### Warm-up period analysis

# In[18]:


# Specify simulated vehicle end time
simulation_results_vehicle_end_time = simulation_results_array[:,5]

# Calculate total travel time 
vehicle_total_travel_time = simulation_results_vehicle_end_time - vehicle_event_queue


# In[35]:


# Plot PDF for simulated total travel time results for all vehicles
# sns.distplot(vehicle_total_travel_time)
# plt.xlabel('Total travel time (sec)', fontsize = 14)
# plt.ylabel('Probability Density', fontsize = 14)
# plt.title('Total travel time for Process-oriented model', fontsize = 14)
# plt.grid(True)
# plt.savefig('Total travel time of simulated process', dpi=150)


# In[32]:


# Calculate and print average travel time 
vehicles_average_travel_time = np.average(vehicle_total_travel_time)
print("Average travel time", vehicles_average_travel_time)


# In[30]:


#print('Average travel time for simulated ' +str(number_of_vehicles)+' vehicles is:', vehicles_average_travel_time)


# In[ ]:




