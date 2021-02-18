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
    try:
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
    except Exception as e:
        logger.error('{}: {})'.format(e.__class__.__name__,e))
        return None

sense = SenseHat()
# this TLE data are needed for computing ISS coordinates using ephem module
# IMPORTANT: make sure this data are updated to last version here -> http://www.celestrak.com/NORAD/elements/stations.txt
name="ISS (ZARYA)"
line1 = "1 25544U 98067A   21047.51651748  .00000569  00000-0  18502-4 0  9991"
line2 = "2 25544  51.6430 219.5729 0002764  22.9843  51.4385 15.48965798269895"

iss = readtle(name,line1,line2)

#set up camera
camera = PiCamera()
camera.resolution = (1296,972) # TODO: check if this is the best resolution

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
min = 257
while(start_time + timedelta(minutes=180) > datetime.now() ):
    try:
        """
        We save a bare minimum of 1000 images to be sure we have enough data, then we only save data that improves the brightness score of our dataset.
        """
        logger.info("{} iteration {}".format(datetime.now(),iterationCounter))
        if(last_catch+timedelta(seconds=7.5) < datetime.now() ):
            iss.compute() # compute lat/long data
            latitude, longitude = iss.sublat, iss.sublong
            orientation = sense.get_orientation()
            setMetadata(latitude ,longitude)
            if PhotoCounter < 1000:
              camera.capture(str(dir_path)+"/image_{0:0=3d}.jpg".format(PhotoCounter)) # takes a photo and gives the file an indexed name
              last_catch = datetime.now()
              row = (last_catch, latitude/degree, longitude/degree, orientation['roll'], orientation['pitch'], orientation['yaw'], PhotoCounter)
              add_csv_data(data_file, row)
              # evaluate brightness of the picture and append the value to BrightnessArray
              with Image.open(str(dir_path)+"/image_{0:0=3d}.jpg".format(PhotoCounter)) as image:
                brightness = calculteBrightness(image)
              BrightnessArray.append(brightness)
              if brightness < min:
                min = brightness
              PhotoCounter += 1
            else:
              camera.capture(str(dir_path)+"/temp.jpg")
              last_catch = datetime.now()
              with Image.open(str(dir_path)+"/temp.jpg") as image:
                brightness = calculteBrightness(image)
              if brightness > min:
                i = 0
                for b in BrightnessArray:
                  if b == min:
                    break
                  i+=1
                # substitute the darkest image with the new brighter one
                BrightnessArray[i]=brightness
                os.remove(str(dir_path)+"/image_{0:0=3d}.jpg".format(i))
                os.rename(str(dir_path)+"/temp.jpg", str(dir_path)+"/image_{0:0=3d}.jpg".format(i)))
                row = (last_catch, latitude/degree, longitude/degree, orientation['roll'], orientation['pitch'], orientation['yaw'], i)
                add_csv_data(data_file, row)
                min = brightness
              else:
                os.remove(str(dir_path)+"/temp.jpg")
                
            """
            
            iss.compute() # compute lat/long data
            latitude, longitude = iss.sublat, iss.sublong
            orientation = sense.get_orientation()
            row = (datetime.now(), latitude/degree, longitude/degree, orientation['roll'], orientation['pitch'], orientation['yaw'])
            
            setMetadata(latitude ,longitude)
            camera.capture(str(dir_path)+"/image_{0:0=3d}.jpg".format(rowCounter+1)) #this line shot a photo and give the file a numeric unique name
            last_catch = datetime.now()
            add_csv_data(data_file, row)
            image = Image.open(str(dir_path)+"/image_{0:0=3d}.jpg".format(rowCounter+1))
            brightness = calculteBrightness(image)
            if(brightness != None and brightness < 65 ):
                os.remove(str(dir_path)+"/image_{0:0=3d}.jpg".format(rowCounter+1))
            image.close()
            rowCounter+=1
            """
    except Exception as e:
        logger.error('{}: {})'.format(e.__class__.__name__,e))
    iterationCounter+=1
