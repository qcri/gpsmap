# gpsmap
Real time map creation and updating using GPS data

# Data:
The data will be available in MV3 shared NFS at: /export/sc1/urbancomp/data/qmic/gpsmap

# # GPS data:
The data is from 2016-07-10 to 2016-07-21, it encloses 2.5M tuples. 
Tuples have the following structure: 
- device id: id of the bluetooth device
- speed_kph: the speed of the device as reported by the GPS. 
- timestamp: 
- longitude: longitude of the devixposition
- latitude: latitude of the device position
- angle: angle of movement (I guess to the North!)
