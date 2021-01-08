#!/usr/bin/env python3
import csv
from sense_hat import SenseHat
from datetime import datetime, timedelta
from pathlib import Path
from time import sleep
from logzero import logger, logfile
from ephem import readtle, degree
from picamera import PiCamera

#create a CSV file, and write headers on top of it
def create_csv(file):
    with open(file,"w") as f:
        writer = csv.writer(f)
        header = ("Date/time", "Latitude", "Longitude", "Temperature", "Humidity", "Pressure")
        writer.writerow(header)

#append a single row of data to the CSV file
def add_csv_data(file,data):
    #open and close data01.csv file every time so that if something goes wrong we wont loose the previous data
    with open(file,"a") as f:
        writer = csv.writer(f)
        writer.writerow(data)


def convert(angle):
    #this function convert the angle in the right format in order to save it in photo's metadata
    """
    Convert an ephem angle (degrees:minutes:seconds) to
    an EXIF-appropriate representation (rationals)
    e.g. '51:35:19.7' to '51/1,35/1,197/10'
    Return a tuple containing a boolean and the converted angle,
    with the boolean indicating if the angle is negative.
    """
    degrees, minutes, seconds = (float(field) for field in str(angle).split(":"))
    exif_angle = f'{abs(degrees):.0f}/1,{minutes:.0f}/1,{seconds*10:.0f}/10'
    return degrees < 0, exif_angle

def setMetadata(lat, long):
    # convert the latitude and longitude to EXIF-appropriate representations
    south, exif_latitude = convert(iss.sublat)
    west, exif_longitude = convert(iss.sublong)

    # set the EXIF tags specifying the current location
    camera.exif_tags['GPS.GPSLatitude'] = exif_latitude
    camera.exif_tags['GPS.GPSLatitudeRef'] = "S" if south else "N"
    camera.exif_tags['GPS.GPSLongitude'] = exif_longitude
    camera.exif_tags['GPS.GPSLongitudeRef'] = "W" if west else "E"




sense = SenseHat()
# this TLE data are needed for computing ISS coordinates using ephem module
# IMPORTANT: make sure this data are updated to last version here -> http://www.celestrak.com/NORAD/elements/stations.txt
name = "ISS (ZARYA)"
line1 = "1 25544U 98067A   21008.41078817  .00001907  00000-0  42360-4 0  9993"
line2 = "2 25544  51.6453  53.0183 0000571 194.6399 179.4598 15.49269883263830"

iss = readtle(name,line1,line2)

#set up camera
camera = PiCamera()
camera.resolution = (1296,972) # TODO: check if this is the best resolution

# get the path of some usefull files
dir_path = Path(__file__).parent.resolve()
data_file = dir_path/'data01.csv'
logfile(dir_path/"teamname.log")

create_csv(data_file)

start_time = datetime.now()
last_catch = datetime.now()
iterationCounter = 0
rowCounter = 0
while(start_time + timedelta(minutes=1) > datetime.now() ):
    try:
        logger.info("{} iteration {} {}".format(datetime.now(),iterationCounter, rowCounter))
        if(last_catch+timedelta(seconds=10) < datetime.now() ):
            iss.compute() # compute lat/long data
            latitude, longitude = iss.sublat, iss.sublong
            row = (datetime.now(), latitude/degree, longitude/degree, sense.temperature, sense.humidity, sense.pressure)
            setMetadata(latitude ,longitude)
            camera.capture(str(dir_path)+"/image_{0:0=3d}.jpg".format(rowCounter+1)) #this line shot a photo and give the file a numeric unique name
            last_catch = datetime.now()
            add_csv_data(data_file, row)
            rowCounter+=1
    except Exception as e:
        logger.error('{}: {})'.format(e.__class__.__name__,e))
    iterationCounter+=1
    #sleep(60) sleep function should not be used when gathering data from sensehat
