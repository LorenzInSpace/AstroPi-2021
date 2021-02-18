#!/usr/bin/env python3
import csv
from sense_hat import SenseHat
from datetime import datetime, timedelta
from pathlib import Path
from time import sleep
from logzero import logger, logfile
from ephem import readtle, degree
from picamera import PiCamera
from PIL import Image
import os
import math


#create a CSV file, and write headers on top of it
def create_csv(file):
    with open(file,"w") as f:
        header = ("Date/time", "Latitude", "Longitude", "Roll", "Pitch", "Yaw","Photo")
        csv.writer(f).writerow(header)

#append a single row of data to the CSV file
def add_csv_data(file,data):
    #open and close data01.csv file every time so that if something goes wrong we wont loose the previous data
    with open(file,"a") as f:
        csv.writer(f).writerow(data)


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




def calculteBrightness(image):
    #calculate the brightness off an image
    #return value: 0 for dark image, 255 for bright image

    greyscale_image = image.convert('L')
    hw,hh = greyscale_image.size[0]/2, greyscale_image.size[1]/2
    r = min(hw,hh)/8
    mean = 0
    delta_theta = 2* math.pi / 8
    # we are sampling 8 pixel for 4 rings around the center of the image to check the brightness without wasting memory or time
    for i in range(32):
        theta = (i%8)*delta_theta
        l = r*(i//8+1)
        mean += greyscale_image.getpixel((int(hw+l*math.sin(theta)),int(hh+l*math.cos(theta))))
    mean/=32
    return mean


sense = SenseHat()
# this TLE data are needed for computing ISS coordinates using ephem module
# IMPORTANT: make sure this data are updated to last version here -> http://www.celestrak.com/NORAD/elements/stations.txt
name="ISS (ZARYA)"
line1 = "1 25544U 98067A   21049.73010417  .00001295  00000-0  31684-4 0  9992"
line2 = "2 25544  51.6438 208.6238 0003101  30.2592 155.9152 15.48974654270239"

iss = readtle(name,line1,line2)

#set up camera
camera = PiCamera()
#camera.resolution = (1296,972)
camera.resolution = (2000,1500)

# get the path of some usefull files
dir_path = Path(__file__).parent.resolve()
data_file = dir_path/'data01.csv'
logfile(dir_path/"lorenzinspace.log")

create_csv(data_file)

start_time = datetime.now()
last_catch = datetime.now()
iterationCounter = 0
PhotoCounter = 0
BrightnessArray = []
min_ = 257
while(start_time + timedelta(minutes=180) > datetime.now() ):
    try:
        """
        We save a bare minimum of 1000 images to be sure we have enough data, then we only save data that improves the brightness score of our dataset.
        """
        #logger.info("{} iteration {}".format(datetime.now(),iterationCounter))
        if(last_catch+timedelta(seconds=2) < datetime.now() ):
            iss.compute() # compute lat/long data
            latitude, longitude = iss.sublat, iss.sublong
            orientation = sense.get_orientation()
            setMetadata(latitude ,longitude)
            if PhotoCounter < 1000:
                camera.capture(str(dir_path)+"/image_{0:0=3d}.jpg".format(PhotoCounter)) # takes a photo and gives the file an indexed name
                print("/image_{0:0=3d}.jpg".format(PhotoCounter))
                last_catch = datetime.now()
                row = (last_catch, latitude/degree, longitude/degree, orientation['roll'], orientation['pitch'], orientation['yaw'], PhotoCounter)
                add_csv_data(data_file, row)
                # evaluate brightness of the picture and append the value to BrightnessArray
                with Image.open(str(dir_path)+"/image_{0:0=3d}.jpg".format(PhotoCounter)) as image:
                    brightness = calculteBrightness(image)
                BrightnessArray.append(brightness)
                if brightness < min_:
                    min_ = brightness
                PhotoCounter += 1
            else:
                camera.capture(str(dir_path)+"/temp.jpg")
                last_catch = datetime.now()
                with Image.open(str(dir_path)+"/temp.jpg") as image:
                    brightness = calculteBrightness(image)
                if brightness > min_:
                    i = 0
                    for b in BrightnessArray:
                        if b == min_:
                            break
                        i+=1
                    # substitute the darkest image with the new brighter one
                    BrightnessArray[i]=brightness
                    print("{}: {} -> {}".format(i,min_, brightness))
                    print(BrightnessArray)
                    os.remove(str(dir_path)+"/image_{0:0=3d}.jpg".format(i))
                    os.rename(str(dir_path)+"/temp.jpg", str(dir_path)+"/image_{0:0=3d}.jpg".format(i))
                    row = (last_catch, latitude/degree, longitude/degree, orientation['roll'], orientation['pitch'], orientation['yaw'], i)
                    add_csv_data(data_file, row)
                    min_ = min(BrightnessArray)
                else:
                    os.remove(str(dir_path)+"/temp.jpg")

    except Exception as e:
        logger.error('{}: {})'.format(e.__class__.__name__,e))
    iterationCounter+=1