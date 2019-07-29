# Introduction
This project is to assess the average travel time for vehicles to traverse a portion of Peachtree Street, the corridor from 10th to 14th street, in midtown Atlanta.

Three different simulation model are developed for the road network.

## Event-oriented queueing model
The Peachtree corridor is modeled as a queueing network where each intersection is modeled as a server, and vehicles must queue while waiting to enter
the intersection.

## Cellular automata model
The corridor is modeled as a cellular automata where each section of each lane of the road is modeled as a cell that is either empty, or contains a single vehicle.
Vehicles move from cell to cell in traveling through the road network using certain movement rules that are encoded into the simulation.

## Process-oriented queueing model
This model is similar to the eventoriented queueing model described above, and should produce identical or nearly identical results, but is implemented using either the activity scanning or process-oriented world view.





