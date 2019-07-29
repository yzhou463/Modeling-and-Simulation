#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 21:58:19 2019

@author: renwendi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 16:07:36 2019

@author: renwendi
"""
import numpy as np
import random
#import matplotlib.pyplot as plt
import pandas as pd
#import seaborn as sns

iuput_data = pd.read_csv("Inter_arrival_time_at_the_intersection_1.csv", header = None)
input_time = np.round(iuput_data.values) #get the discrete time stamp

# Section lengths (Table 3-3, A2, afternoon)
section_2_length = 417.976  # unit: ft 
section_3_length = 412.172
section_4_length = 351.511
section_5_length = 344.427

# Intersection lengths 
intersection_1_length = 99.732  # unit: ft 
intersection_2_length = 129.875
intersection_3_length = 73.513
intersection_4_length = 66.602
intersection_5_length = 121.319

# Calculate average car length from the dataset
car_length = 16  # unit: ft 

# Get from NGSIM-Data signalTiming 
red1 = 49.3
green1 = 34.7
red2 = 55.4
green2 = 51.4
red3 = 35.7
green3 = 60.9
red5 = 46.1
green5 = 34.6

green = [round(green1), round(green2), round(green3), round(green5)]
red = [round(red1), round(red2), round(red3), round(red5)]

s1 = intersection_1_length + section_2_length
s2 = s1 + intersection_2_length + section_3_length
s3 = s2 + intersection_3_length + section_4_length
s4 = s3 + intersection_4_length + section_5_length + intersection_5_length
road_length = [s1, s2, s3, s4]

# Convert lat/long to x/y coordinate using http://www.earthpoint.us/stateplane.aspx
Intersection_y1 = 1375699.434  # Intersection 1
Intersection_y2 = 1376244.052  # Intersection 2
Intersection_y3 = 1376615.802  # Intersection 3
Intersection_y4 = 1377033.444  # Intersection 4
Intersection_y5 = 1377404.299  # Intersection 5

inter_section_location_y = [0, Intersection_y2-Intersection_y1, Intersection_y3-Intersection_y1, Intersection_y5-Intersection_y1]
inter_section_location_localy = np.array(inter_section_location_y, dtype = int) / car_length
inter_section_location_localy = np.array(inter_section_location_localy, dtype = int)


def create_plaza(lane_number,plazalength):  
    plaza=np.zeros((plazalength,lane_number+2), dtype = int)
    v=np.zeros((plazalength,lane_number+2))
    plaza[:,0]=-1
    plaza[:,lane_number+1]=-1
    return plaza, v

def show_plaza(plaza,h,n,t):
    L = len(plaza)
    W = len(plaza[0])
    temp = np.zeros((120,W))#create the palza without any cars
    temp[:,0] = -1
    temp[:,-1] = -1
    plaza_draw=plaza[0:120,:]
    
    PLAZA = np.zeros((120,W,3))
    PLAZA[:,:,0]=plaza_draw
    PLAZA[:,:,1]=plaza_draw
    PLAZA[:,:,2]=temp
    plt.figure(figsize=(10,10)) 
    plt.imshow(PLAZA)
    plt.savefig(str(t)+".png")
#    plt.colorbar()
#    plt.show() 

def new_cars_input(input_time, plaza, t, input_wl, vmax, VMAX,v):
    for i in range(len(input_wl)):
        if(input_wl[i] == t):
            lane_No = np.random.randint(1,3)
            plaza[0, lane_No] = 1
            vmax[0,lane_No]=VMAX
            rand2 = random.random()
            v[0,lane_No]=rand2*vmax[0,lane_No] # Random initial speed for current position
            break
    return plaza, v, vmax

def random_new(new_car_number):
    import heapq
    def heapsort(iterable):
        h = []
        for value in iterable:
            heapq.heappush(h, value)
        return [heapq.heappop(h) for i in range(len(h))]
    
    # Create a queue for vehicle events using random number generator
    random_number_generator = np.round((np.random.uniform(low=0, high=900, size=new_car_number)),1)
#    random_number_generator = np.random.poisson(5, size = new_car_number)
    vehicle_event_queue = np.array(heapsort(random_number_generator), dtype = int)
    return vehicle_event_queue

def para_count(plaza,v,vmax):
    L = len(plaza) # the number of cells / the lenght of the road
    W = len(plaza[0])
    
    #  === rule1: speed up for the maximum speed ===
    #  IF v(i) != vd(i) THEN v(i) := v(i) + 1
    for lanes in range(1,W-1):
        # lanes = 1 or 2
        temp=np.where(plaza[:,lanes]==1)[0]
        for k in range(0,len(temp)):
            i=temp[k]
            v[i,lanes]=min(v[i,lanes]+1,vmax[i,lanes])
            
    #  === step2: gaps between current car and the front one === 
    # gap(i) := x(pred(i)) - x(i) - 1 the width of the gap to the predecessor
    gap=np.zeros((L,W),  dtype=int)
    for lanes in range(1,W-1):
        temp=np.where(plaza[:,lanes]==1)[0]
        nn=len(temp) # The number of cars in this lane
        for k in range(0,nn):
            i=temp[k]
            if(k==(nn-1)):
                gap[i,lanes]=L-(temp[k]-temp[0]+1) #periodic boundary 
                continue
            
            gap[i,lanes]=temp[k+1]-temp[k]-1  
    
    # === Compute the distance of the other lane ===
    LUP=np.zeros((L,W)) #forward gap on the other lane
    LDOWN=np.zeros((L,W)) #backward gap on the other lane
    
    for lanes in range(1,W-2):
        temp=np.where(plaza[:,lanes]==1)[0]
        nn=len(temp)  
        for k in range(0,nn):
            i=temp[k] 
            LDOWN[i,lanes]=(plaza[np.mod(i-2,L),lanes+1]==0)  
            if(k==(nn-1)):
                if(sum(plaza[i:L,lanes+1])==0 and sum(plaza[1:np.mod(i+gap[i,lanes],L),lanes+1])==0):
                    LUP[i,lanes]=1  

                continue  

            if(sum(plaza[i:i+gap[i,lanes],lanes+1])==0):
                LUP[i,lanes]=1  


    return v,gap,LUP,LDOWN

def switch_lane(plaza,v,vmax,gap,LUP,LDOWN):
    L = len(plaza)
    W = len(plaza[0])
    changeL=np.zeros((L,W))# can turn left
    changeR=np.zeros((L,W))# can turn right
    # can turn left?
    for lanes in range (1,W-2):
        # lane = 1
        temp=np.where(plaza[:,lanes]==1)[0]
        nn=len(temp)
        for k in range (0,nn):
            i=temp[k]
            if(v[i,lanes]>gap[i,lanes] and LUP[i,lanes]==1 and LDOWN[i,lanes]==1):
                  changeL[i,lanes]=1
    
    # can turn right?
    for lanes in range (2,W-1):
        # lane = 2
        temp=np.where(plaza[:,lanes]==1)[0]
        nn=len(temp)
        for k in range (0,nn):
            i=temp[k]
            if(plaza[i,lanes-1]==0 and plaza[np.mod(i-1-1,L),lanes-1]==0 and plaza[np.mod(i-2-1,L),lanes-1]==0 and plaza[np.mod(i,L),lanes-1]==0 and plaza[np.mod(i+1,L),lanes-1]==0):
                changeR[i,lanes]=1

    # turn right first
    for lanes in range (2,W-1):
        temp=np.where(changeR[:,lanes]==1)[0]
        nn=len(temp);
        for k in range (0,nn):
            i=temp[k]
            plaza[i,lanes-1]=1
            v[i,lanes-1]=max(v[i,lanes]-1,1)
            vmax[i,lanes-1]=vmax[i,lanes]
            plaza[i,lanes]=0
            v[i,lanes]=0
            vmax[i,lanes]=0          
            changeL[i,lanes]=0

    # turn left
    for lanes in range (1,W-2):
        temp=np.where(changeL[:,lanes]==1)[0]
        nn=len(temp)
        for k in range (0,nn):
            i=temp[k]
            plaza[i,lanes+1]=1
            v[i,lanes+1]=max(v[i,lanes]-1,1)
            vmax[i,lanes+1]=vmax[i,lanes]
            plaza[i,lanes]=0
            v[i,lanes]=0
            vmax[i,lanes]=0
    
    return plaza,v,vmax

def random_slow(plaza,v,vmax,probslow):
    # === IFv(i) >0 AND rand <p(i) THEN v(i) := v(i) - 1
    W = len(plaza[0])
    for lanes in range (1,W-1):
        temp=np.where(plaza[:,lanes]==1)[0]
        nn=len(temp)
        for k in range(0,nn):
            i=temp[k]
            rand = random.random()
            if(rand<=probslow):
                v[i,lanes]=max(v[i,lanes]-1,0)
    return plaza,v,vmax
    
def generate_traffice_lights(iterations, green, red):
    #start_light = np.random.randint(0,2,4)
    traffic_light_time = np.zeros((4, iterations))
    for i in range(4):
        traffic_change_time = 0
        iterations_time = iterations
        while(iterations_time/green[i]>0):
            if(traffic_change_time%2==0): #green
                traffic_light_time[i, green[i]*traffic_change_time: green[i]*(traffic_change_time+1)] = traffic_change_time%2
                traffic_change_time = traffic_change_time + 1
                iterations_time = iterations_time - green[i]
            if(traffic_change_time%2==1): #red
                traffic_light_time[i, red[i]*traffic_change_time: red[i]*(traffic_change_time+1)] = traffic_change_time%2
                traffic_change_time = traffic_change_time + 1
                iterations_time = iterations_time - red[i]
                
    return traffic_light_time

def traffic_light(inter_section_location_localy, plaza, traffic_light_time, t, v):
    W = len(plaza[0])
    for lanes in range (1,W-1):
        temp=np.where(plaza[:,lanes]==1)[0]
        nn=len(temp) # The number of the cars in the lane
        
        #Case 1: The traffic light is red in front of the nth vehicle min(vn, dn-1, sn-1)
        for i in range(nn):
            for j in range(len(inter_section_location_localy)):
                if temp[i] == inter_section_location_localy[j] - 1: #this car is at the intersection - 1
                    if(traffic_light_time[j, t] == 1.0): #red light
                        v[i,lanes] = 0 #brake
#                        v[:,:] = 0
                        print("red at "+ str(t) + " time.")
                        
    return v
        

def move_forward(plaza,v,vmax,probslow,t):
    move_t = 0
    L = len(plaza)
    W = len(plaza[0])
    # Compute values to get type of cars: turn left or stay
    gap=np.zeros((L,W), dtype=int)
    for lanes in range(1, W-1):
        temp=np.where(plaza[:,lanes]==1)[0] 
        nn=len(temp) # The number of the cars in the lane
        for k in range(0,nn):
            i=temp[k]
            if(k==nn-1):
                gap[i,lanes]=L-(temp[k]-temp[0]+1) # periodic boundary 
                continue

            gap[i,lanes]=temp[k+1]-temp[k]-1

    for lanes in range(1,W-1):
         temp=np.where(plaza[:,lanes]==1)[0]
         nn=len(temp)
         for k in range(0,nn):
             i=temp[k] #current location
#             print(i,pos)
             if(v[i,lanes]<=gap[i,lanes]):
                pos=int(i+v[i,lanes]-1) #next location
             
             if(v[i,lanes]>gap[i,lanes]): 
                pos=int(i+gap[i,lanes]-1) #next location
             if(pos!=i):
                 if(pos < L):
                     plaza[pos,lanes]=1
                     v[pos,lanes]=v[i,lanes]
                     vmax[pos,lanes]=vmax[i,lanes]
                     plaza[i,lanes]=0
                     v[i,lanes]=0
                     vmax[i,lanes]=0
                 elif(pos >= L):
                     plaza[i,lanes]=0
                     v[i,lanes]=0
                     vmax[i,lanes]=0
                     move_t = t
#                     print(t)
                     
    return plaza,v,vmax,move_t

#=============main=================
#=== parameters ===
lane_number=2       # The number of the lanes
probc=0.4         # Density of the cars
probslow=0.1        # The probability of random slow
Dsafe=1             # The safe gap distance for the car to change the lane
VMAX = 3       # Fixed max for any car, got it from the real dataset
iterations = 1000  # Number of iterations
plazalength = s4 / car_length
h=None              # The handle of the image

#=== initialization ===
#---use input_time to create the waiting list for plaza
input_wl = np.zeros(len(input_time) + 1)    
for i in range(len(input_time)):
    input_wl[i+1] = input_wl[i] + input_time[i]

#new_car_number = 91    
#input_wl = random_new(new_car_number)  #for random generator

number_cars = len(input_wl)   

plaza,v=create_plaza(lane_number,int(plazalength))
L = len(plaza)
W = len(plaza[0])
vmax=np.zeros((L,W))

traffic_light_time = generate_traffice_lights(iterations, green, red) #generate traffic time list
#=== initialization end ===

lane1_intersection = []
lane2_intersection = []
lane_intersection = [] #store the information at intersection for two lanes

move_t_list = []
car_numbers = [] #used for warm up
for t in range(0, iterations):
    #--- visualization ---
#    PLAZA=np.rot90(plaza,2);
#    show_plaza(PLAZA,h,0.1,t)
    
    # generate cars based on the distribution at intersection 1
    plaza,v,vmax = new_cars_input(input_time, plaza, t, input_wl, vmax, VMAX, v)
    
    # move forward based on the rules
    v,gap,LUP,LDOWN=para_count(plaza,v,vmax)
    v = traffic_light(inter_section_location_localy, plaza, traffic_light_time, t, v)
    plaza,v,vmax = switch_lane(plaza,v,vmax,gap,LUP,LDOWN)
    plaza,v,vmax = random_slow(plaza,v,vmax,probslow)
    plaza,v,vmax, move_t = move_forward(plaza,v,vmax,probslow,t)
    
    move_t_list.append(move_t) #store the cars which ends the simulation (pass through 14th street)
    
    position_lane1 = np.where(plaza[:,1]==1)[0]
    position_lane2 = np.where(plaza[:,2]==1)[0]
    for i in range(len(inter_section_location_localy)):
        for k in range(len(position_lane1)):
            if(position_lane1[k] == inter_section_location_localy[i]):
                if(t>20): #warm up period delete
                    lane_intersection.append([i+1, t])
        for j in range(len(position_lane2)):
            if(position_lane2[j] == inter_section_location_localy[i]):
                if(t>20): #warm up period delete
                    lane_intersection.append([i+1, t])
    
    #--- for warm up analysis ---
    car_number_at_road = len(position_lane1) +  len(position_lane2)
    car_numbers.append(car_number_at_road)

#get the time stamp for each car when they exit 14th
move_t_list_end = []
for t in move_t_list:
    if t!=0:
        move_t_list_end.append(t)

#---compute total travel time from 10th to 14th---
warmup_delete_car = 10
car_travel_time = move_t_list_end[warmup_delete_car:] - input_wl[warmup_delete_car:len(move_t_list_end)]     
average_time = np.mean(car_travel_time)
print('Average total travel time: ', average_time) 

#---compute the inter-arrival time---
lane_intersection_time = np.array(lane_intersection)        
lane_intersection_time.sort(axis = 0)
inter_arrival_time_section1 = np.diff(lane_intersection_time[lane_intersection_time[:,0]==1], axis = 0)[:,1]
inter_arrival_time_section2 = np.diff(lane_intersection_time[lane_intersection_time[:,0]==2], axis = 0)[:,1]
inter_arrival_time_section3 = np.diff(lane_intersection_time[lane_intersection_time[:,0]==3], axis = 0)[:,1]
inter_arrival_time_section5 = np.diff(lane_intersection_time[lane_intersection_time[:,0]==4], axis = 0)[:,1]

#---get the plots---

#plt.figure()
#sns.distplot(inter_arrival_time_section2, color='y')
#plt.xlabel('Inter-Arrival time (sec)', fontsize = 12)
#plt.ylabel('Probability Density', fontsize = 12)
#plt.title('CA_Simulation result - Intersection 2', fontsize = 12)
#plt.grid(True)
#plt.savefig('r_CA_Simulation_Result_Intersection_2', dpi=150)
#plt.show()
#
#plt.figure()
#sns.distplot(inter_arrival_time_section3, color='y')
#plt.xlabel('Inter-Arrival time (sec)', fontsize = 12)
#plt.ylabel('Probability Density', fontsize = 12)
#plt.title('CA_Simulation result - Intersection 3', fontsize = 12)
#plt.grid(True)
#plt.savefig('r_CA_Simulation_Result_Intersection_3', dpi=150)
#plt.show()
#
#plt.figure()
#sns.distplot(inter_arrival_time_section5, color='y')
#plt.xlabel('Inter-Arrival time (sec)', fontsize = 12)
#plt.ylabel('Probability Density', fontsize = 12)
#plt.title('Simulation result - Intersection 5', fontsize = 12)
#plt.grid(True)
#plt.savefig('r_CA_Simulation_Result_Intersection_5', dpi=150)
#plt.show()
#
#plt.figure()
#sns.distplot(np.array(car_travel_time), color='y')
#plt.xlabel('Total travel time (sec)', fontsize = 14)
#plt.ylabel('Probability Density', fontsize = 14)
#plt.title('CA Model total travel time', fontsize = 14)
#plt.grid(True)
#plt.savefig('r_CA_average_travel_time', dpi=150)
#plt.show()

#plt.figure()
#plt.plot(car_numbers[:1000], color='y',linewidth=2.0)
#plt.xlabel('Simulation time stamp', fontsize = 14)
#plt.ylabel('Number of cars at the street', fontsize = 14)
#plt.title('CA Model warm up analysis 1', fontsize = 14)
#plt.grid(True)
#plt.savefig('CA_warm_up_all.png', dpi=150)
#plt.show()
#
#plt.figure()
#plt.plot(car_numbers[:100], color='y',linewidth=2.0)
#plt.xlabel('Simulation time stamp', fontsize = 14)
#plt.ylabel('Number of cars at the street', fontsize = 14)
#plt.title('CA Model warm up analysis 2', fontsize = 14)
#plt.grid(True)
#plt.savefig('CA_warm_up.png', dpi=150)
#plt.show()
