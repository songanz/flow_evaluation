#!/bin/bash
python "./randomTrips.py" -n sID_0.net.xml --fringe-factor 100 -p 0.030164 -o sID_0.passenger.trips.xml -e 3600 --vehicle-class passenger --vclass passenger --prefix veh --min-distance 300 --trip-attributes "departLane=\"best\" departSpeed=\"max\" departPos=\"random\"" --allow-fringe.min-length 1000 --lanes --validate
