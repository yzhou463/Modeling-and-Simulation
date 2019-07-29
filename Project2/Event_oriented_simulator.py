import numpy as np
import math
import pandas as pd

# check status of traffic lights
# decide vehicle whether wait or go through
# update traffic_lights_list and flag of traffic light
def RedTrafficLight(current_event, WUlocation, traffic_lights_list, flag_lights, signaltime):
    lights_time = traffic_lights_list[WUlocation][0]
    if(current_event <= lights_time and flag_lights[WUlocation] == 'G'):
        return True
    if(current_event > lights_time and flag_lights[WUlocation] == 'G'):
        time_redgreen = signaltime[WUlocation][0] + signaltime[WUlocation][1]
        num_lights = math.floor(current_event / time_redgreen)
        if(num_lights == 0):
            flag_lights[WUlocation] = 'R'
            traffic_lights_list[WUlocation].pop(0)
            traffic_lights_list[WUlocation].append(traffic_lights_list[WUlocation][0]+signaltime[WUlocation][0])
            return False
        else:
            diff = current_event - num_lights*time_redgreen
            low = num_lights*time_redgreen + signaltime[WUlocation][0]
            mid = low + signaltime[WUlocation][1]
            high = mid + signaltime[WUlocation][0]
            if(diff > signaltime[WUlocation][0]):
                flag_lights[WUlocation] = 'R'
                traffic_lights_list[WUlocation] = [mid, high]
                return False
            else:
                flag_lights[WUlocation] = 'G'
                traffic_lights_list[WUlocation] = [low, mid]
                return True
    
    if(current_event <= lights_time and flag_lights[WUlocation] == 'R'):
        return False
    if(current_event > lights_time and flag_lights[WUlocation] == 'R'):
        time_redgreen2 = signaltime[WUlocation][0] + signaltime[WUlocation][1]
        num_lights2 = math.floor(current_event / time_redgreen2)
        if(num_lights2 == 1):
            flag_lights[WUlocation] = 'G'
            traffic_lights_list[WUlocation].pop(0)
            traffic_lights_list[WUlocation].append(traffic_lights_list[WUlocation][0]+signaltime[WUlocation][1])
            return True
        else:
            diff2 = current_event - num_lights2*time_redgreen2
            low2 = num_lights2*time_redgreen2 + signaltime[WUlocation][0]
            mid2 = low2 + signaltime[WUlocation][1]
            high2 = mid2 + signaltime[WUlocation][0]
            if(diff2 > signaltime[WUlocation][0]):
                flag_lights[WUlocation] = 'R'
                traffic_lights_list[WUlocation] = [mid2, high2]
                return False
            else:
                flag_lights[WUlocation] = 'G'
                traffic_lights_list[WUlocation] = [low2, mid2]
                return True
        
# schedule future event
# ATstatus: False(traffic light is Red), True(traffic light is Green)
def AdvanceTime(current_event, ATstatus, future_event):
    if(ATstatus):
        return current_event+future_event
    else:
        return future_event
    
# event handler of arrival event
# calculate arrival event of vehicles at intersection 1 to 5
# AEstatus: False(traffic light is Red), True(traffic light is Green)
def ArrivalEvent(current_event, vehicle_num, AElocation, AEstatus):
    if(AElocation == 4):
        AElocation = 5
    if(AEstatus):
        print('Arrival Event: Vehicle {} arrives at Intersection {} '.format(
        vehicle_num, AElocation)+'on time {} s'.format(round(current_event,1)), '\n')
    else:
        print('Traffic Red light Event: Vehicle {} arrives at Intersection {} '.format(
        vehicle_num, AElocation)+'on time {} s'.format(round(current_event,1))+', under Red light', '\n')

# event handler of depart event
# calculate depart event of vehicles at intersection 1 to 5
def DepartureEvent(current_event, vehicle_num, DElocation):
    if(DElocation == 4):
        DElocation = 5
    print('Departure Event: Vehicle {} departs Intersection {} '.format(
    vehicle_num, DElocation)+'on time {} s'.format(round(current_event,1)), '\n')
    
# average service time of Intersection 1 to 5
Service_time_1 = 3.4
Service_time_2 = 6.3
Service_time_3 = 4.9
Service_time_4 = 2.0
Service_time_5 = 6.7

# average travel time of Section 2 to 5
travel_time_2 = 23.7
travel_time_3 = 30.5
travel_time_4 = 10.9
travel_time_5 = 41.4

# service and travel time list in location order
service_travel_time = [0, 3.4, 0, 23.7, 6.3, 0, 30.5, 4.9, 0, 54.3, 6.7]
service_travel_time2 = [0, 0, 3.4, 0, 0, 6.3, 0, 0, 4.9, 0, 0, 6.7]

# random generator of vehicle arrival time at intersection 1
# average interarrival time for intersection 1 is 9.9s
# interarrival rate is lamda
lamda = 1/9.9
    
# arrival time list for vehicles
data = pd.read_csv('Inter_arrival_time_at_the_intersection_1.csv', names = 'T').loc[:,'T'].tolist()
arrival_1 = [0]
for i in range(len(data)):
    arrival_1.append(round(arrival_1[i]+data[i],1))

# initilize future event list
FEL = arrival_1

# status list of each vehicle
# vehicle index range from 0 to num
# initialize with arrival time at first intersection 
vehicle_status_list = {}
for i in range(len(arrival_1)):
    vehicle_status_list[i]= [arrival_1[i]]

# status list of traffic lights of intersections
# initialize with signaltiming values
signaltime = {}
signaltime[1] = [34.7, 49.3]
signaltime[2] = [41.5, 55.4]
signaltime[3] = [60.9, 35.7]
signaltime[4] = [34.6, 46.1]

traffic_lights_list = {}
traffic_lights_list[1] = [34.7, 84]
traffic_lights_list[2] = [41.5, 96.9]
traffic_lights_list[3] = [60.9, 96.6]
traffic_lights_list[4] = [34.6, 80.7]

# initial traffic light
flag_lights = ['','G','G','G', 'G']

# arrival event index of each intersection
arrival_event_index = [0, 3, 6, 9]

# red light changes to green light index of each intersection
red_green_index = [1, 4, 7, 10]

# departure event index of each intersection
depart_event_index = [2, 5, 8, 11]

# event index recorder for each vehicle
vehicle_event_index = [0]*len(arrival_1)
print('initial number of vehicle arrives at intersection 1:',len(arrival_1))

# average length of cars, unit:feet
car_len = 16.5

# average car velocity of Section 1 to 5, unit:feet/s
ave_speed_1 = 5.17
ave_speed_2 = 17.61
ave_speed_3 = 13.52
ave_speed_4 = 32.19
ave_speed_5 = 8.31

# arrival time adjustment of cars of intersection from 1 to 5
# when traffic light is Red
# time_adjust = car_len / ave_speed
time_adjust = [0, 3.2, 0.9, 1.2, 2.0]

# queue recorder for each intersection when vehicle enters into the queue
# queue counter for each intersection when vehicle leaves the queue
queue_record = {}
flag = 0

# arrival time list for intersection 2, 3, 5
arrival_time = []
arrival_time2 = []
arrival_time3 = []
arrival_time5 = []

# event processing loop
while(len(FEL) !=0):
    # delete smallest timestamp in FEL
    current_event = FEL.pop(0)
    # find location, vehicle_num of current_event
    for item in vehicle_status_list:
        temp = vehicle_status_list[item]
        if current_event in temp:
            location = math.ceil((temp.index(current_event)+1)/3)
            vehicle_num = item
            if(temp.index(current_event) < vehicle_event_index[vehicle_num]):
                continue
            else:
                break

    # arrival event at each intersection [0, 3, 6, 9]
    if vehicle_event_index[vehicle_num] in arrival_event_index:
        # check status of traffic light
        status = RedTrafficLight(current_event, location, traffic_lights_list, flag_lights, signaltime)
    
        # output arrival event
        ArrivalEvent(current_event, vehicle_num, location, status)
        arrival_time.append(round(current_event,1))

        # collect arrival time list
        if(location == 2):
            arrival_time2.append(round(current_event,1))
        if(location == 3):
            arrival_time3.append(round(current_event,1))
        if(location == 4):
            arrival_time5.append(round(current_event,1))


        # schedule future event
        if(status):
            stt_index = len(vehicle_status_list[vehicle_num])
            future_event = AdvanceTime(current_event, status, service_travel_time[stt_index])
            # add future event into FEL
            FEL.append(round(future_event,1))
            vehicle_status_list[vehicle_num].append(-1)
            vehicle_status_list[vehicle_num].append(round(future_event,1))
            # update vehicle event index
            vehicle_event_index[vehicle_num] += 2
        else:
            future_event = AdvanceTime(current_event, status, traffic_lights_list[location][0])
            # add future event into FEL
            FEL.append(round(future_event,1))
            vehicle_status_list[vehicle_num].append(round(future_event,1))
            # update vehicle event index
            vehicle_event_index[vehicle_num] += 1
            # update queue_recorder
            Redlight_time = round(traffic_lights_list[location][0],1)
            if Redlight_time in queue_record:
                queue_record[Redlight_time] +=1
            else:
                queue_record[Redlight_time] = 1

        # keep the FEL in ascending order
        FEL.sort()
        continue

    # vehicle moves forward when red light changes to green light [1, 4, 7, 10]
    if vehicle_event_index[vehicle_num] in red_green_index:
        stt_index2 = len(vehicle_status_list[vehicle_num])
        # add time adjustment to vehicles under same time red light
        if(flag < queue_record[current_event]):
            addedtime = flag * time_adjust[location] + service_travel_time2[stt_index2]
            flag += 1
        if(flag >= queue_record[current_event]):
            flag = 0
            queue_record[current_event] = 0
        future_event2 = AdvanceTime(current_event, True, addedtime)
        # add future event into FEL
        FEL.append(round(future_event2,1))
        # keep the FEL in ascending order
        FEL.sort()
        vehicle_status_list[vehicle_num].append(round(future_event2,1))
        # update vehicle event index
        vehicle_event_index[vehicle_num] += 1
        continue

    # departure event at each intersection [2, 5, 8, 11]
    if vehicle_event_index[vehicle_num] in depart_event_index:
        DepartureEvent(current_event, vehicle_num, location)
        if(vehicle_event_index[vehicle_num] == 11):
            vehicle_event_index[vehicle_num] += 1
        stt_index3 = len(vehicle_status_list[vehicle_num])
        if(stt_index3 <= 10):
            future_event3 = AdvanceTime(current_event, True, service_travel_time[stt_index3])
            # add future event into FEL
            FEL.append(round(future_event3,1))
            # keep the FEL in ascending order
            FEL.sort()
            vehicle_status_list[vehicle_num].append(round(future_event3,1))
            # update vehicle event index
            vehicle_event_index[vehicle_num] += 1

