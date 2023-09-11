import numpy as np
import cv2 as cv
import math
import numpy as np
import csv
import scipy
from os import listdir
import matplotlib.pyplot as plt

distance = 0
worlddistance = 6.4008
origin = (0, 0)

def select_points(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDBLCLK:
        ix.append(x)
        iy.append(y)

def select_origin(event, x, y, flags, param):
    global origin
    if event == cv.EVENT_LBUTTONDBLCLK:
        origin = (x, y)

def get_background(video):
    frame_indices = video.get(cv.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=50)
    frames = []
    for i in frame_indices:
        video.set(cv.CAP_PROP_POS_FRAMES, i)
        ret, frame = video.read()
        frames.append(frame)
    background = np.median(frames, axis=0).astype(np.uint8)
    return background

def nothing(x):
    pass


def line_to_point_distance(p, q, r):
    d = np.linalg.norm(np.cross(q-p, p-r))/np.linalg.norm(q-p)
    return d
 
def get_mask(image):
    cv.namedWindow('image', cv.WINDOW_NORMAL)
    cv.createTrackbar('HMin', 'image', 0, 179, nothing)
    cv.createTrackbar('SMin', 'image', 0, 255, nothing)
    cv.createTrackbar('VMin', 'image', 0, 255, nothing)
    cv.createTrackbar('HMax', 'image', 0, 179, nothing)
    cv.createTrackbar('SMax', 'image', 0, 255, nothing)
    cv.createTrackbar('VMax', 'image', 0, 255, nothing)

    cv.setTrackbarPos('HMax', 'image', 179)
    cv.setTrackbarPos('SMax', 'image', 255)
    cv.setTrackbarPos('VMax', 'image', 255)

    hMin = sMin = vMin = hMax = sMax = vMax = 0
    phMin = psMin = pvMin = phMax = psMax = pvMax = 0

    while(1):
        hMin = cv.getTrackbarPos('HMin', 'image')
        sMin = cv.getTrackbarPos('SMin', 'image')
        vMin = cv.getTrackbarPos('VMin', 'image')
        hMax = cv.getTrackbarPos('HMax', 'image')
        sMax = cv.getTrackbarPos('SMax', 'image')
        vMax = cv.getTrackbarPos('VMax', 'image')

        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])

        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, lower, upper)
        result = cv.bitwise_and(image, image, mask=mask)

        if((phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
            print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
            phMin = hMin
            psMin = sMin
            pvMin = vMin
            phMax = hMax
            psMax = sMax
            pvMax = vMax

        cv.imshow('image', result)
        if cv.waitKey(10) & 0xFF == ord('q'):
            break
    cv.destroyAllWindows()

def getDistance(frame1):
    global distance
    cv.setMouseCallback('Window', select_points)
    cv.imshow('Window', frame1)
    k = cv.waitKey(0) & 0xff
    if k == 27:
        distance = math.dist((ix[1], iy[1]), (ix[0], iy[0]))
        
def getOrigin(frame1):
    global origin
    cv.setMouseCallback('Window', select_origin)
    cv.imshow('Window', frame1)
    k = cv.waitKey(0) & 0xff
    if k == 27:
        print(origin)

def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return (ang1 - ang2) % (2 * np.pi)

feature_params = dict(maxCorners=200, qualityLevel=0.2, minDistance=7, blockSize=5)
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03),
)
color = np.random.randint(0, 255, (100, 3))
cv.namedWindow('Window', cv.WINDOW_NORMAL)

for filename in listdir("E:/Image Tracking/videos/"):
    #Make sure the program only runs on video files, if we use a different format just change it here as
    #long as opencv supports it 
    if filename.endswith(".MP4") or filename.endswith(".mp4"):
        centers = []
        ix = []
        iy = []
        origin = (0, 0)


        #Create a window that is scaled to monitor size, not set to the actual size of the video being
        #used
        
        cap = cv.VideoCapture(f"videos/{filename}")
        ret, frame = cap.read()
        
        #Call function to set a distance inside the image to be equal to 21 feet in real-world units
        getDistance(frame)

        #Call function to get the origin (bottom left anchor)
        getOrigin(frame)

        # backSub = cv.createBackgroundSubtractorKNN()
        droneLocation = ((frame.shape[1]) / 2, (frame.shape[0]) / 2)
        print(droneLocation)

        #Create a MOG2 background subtractor object, initlizaing the thershold to 6000 to try and account for instability of the drone
        backSub = cv.createBackgroundSubtractorMOG2(varThreshold= 6000)
        while (1):
            ret, frame = cap.read()
            if not ret:
                print('No frames grabbed!')
                break

            #Utilize Gaussain Blurring to decrease noise but also minimize the effect of lighting on the algorithm
            frame = cv.GaussianBlur(frame, (15,15), 3)

            #Invert the BGR image, so now we want to mask for Cyan instead of red. Higher Contrast
            bgr_inv = ~frame

            #Convert to HSV colorspace for better discrimination between colors
            hsv_inv = cv.cvtColor(bgr_inv, cv.COLOR_BGR2HSV)

            #Apply the mask with the lower and upper bounds determined through the use of the get_mask function
            mask = cv.inRange(hsv_inv, np.array([68, 0, 100]), np.array([127, 108, 200]))

            #Old mask for masking for cyan
            # mask = cv.inRange(hsv_inv, np.array([90 - 10, 70, 50]), np.array([90 + 10, 255, 255]))
            #Apply mask created by the bounds to the original frame
            final = cv.bitwise_and(frame, frame, mask=mask)

            #Convert back to BGR colorspace (Might not be needed)
            prvs = cv.cvtColor(final, cv.COLOR_HSV2BGR)

            #Apply background subtraction using the new frame to update the mask
            fgMask = backSub.apply(prvs)

            #Puts a white rectangle to then put frame number in upper left hand corner of the screen
            cv.rectangle(frame, (10,2), (100,20), (255,255,255), -1)
            cv.putText(frame, str(cap.get(cv.CAP_PROP_POS_FRAMES)), (15,15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            keyboard = cv.waitKey(30)

            #Allow you to run the get_mask function on a frame where the boomerang has already been thrown
            # if cap.get(cv.CAP_PROP_POS_FRAMES) == 40:
            #     get_mask(bgr_inv)
          

            #Create contours over the remaining pieces of image, use this to get actual positional data
            contours, hierarchy = cv.findContours(
                fgMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            goodcontours = []

            #Here we try and do some calculations to only save contours that include the boomerang
            for c in contours:
                M = cv.moments(c)

                #This checks to see if the area of the contour is above a certain amount, helps reduce the noise
                #or not boomerang stuff that makes it through all the earlier image processing
                if M['m00'] > 120:
                    x, y, w, h = cv.boundingRect(c)

                    #Checks to see if the width and height of the contour are within an acceptable range before
                    #accepting it. Can be tweaked as well
                    if w < 50 and h < 60:
                        goodcontours.append(c)

                        #Compute the cetner of the contour with the moments, and then store those to the centers
                        #variable to eventually store in csv file. We also put a red dot where the computed centroid
                        #is
                        cx = int(M['m10']/M['m00'])
                        cy = int(M['m01']/M['m00'])
                        centers.append((cx, cy))
                        cv.circle(frame, (cx, cy), 7, (0, 0, 255), -1)
            for c in centers:
                cv.circle(frame, (c[0], c[1]), 7, (0, 0, 255), -1)

            #Draw all the good contours on the original image so that you can see how well the program is tracking
            #the boomerang.
            cv.drawContours(frame, goodcontours, -1, (0, 255, 0), 3)
            cv.imshow('Window', frame)
            if keyboard == 'q' or keyboard == 27:
                break

        normalizedcenters = []   

        #Initialize the origin for the data to be the first contour drawn (typically where the boomerang was thrown from)
        droneLocation = (int(droneLocation[0]), int(droneLocation[1]))
        a = list(map(int.__sub__, list(origin), list(droneLocation)))
        droneLocation = np.array(a) * worlddistance / distance
        
        droneLocation[0] = droneLocation[0] * -1
        print("Drone Location:")
        print(droneLocation)

        file = filename.split(".")

        with open(f'old results/UWB/{file[0] + "." + file[1] + ".txt"}') as file:
            lines = file.readlines()
        
        uwbData = []
        
        for line in lines:
            x = line.split(",")
            uwbData.append((x[1], x[2]))

        #Go through every data point in centers, and compute what the distance is in real-world units (meters)
        #from the approximation received from the getDistance function
        for c in centers:
            a = (c[0] - origin[0], c[1] - origin[1])
            # a = list(map(int.__sub__, list(c), list(origin)))
            b = np.array(a) * worlddistance / distance
            normalizedcenters.append(b)

        originpoint = [origin[0], origin[1] - 100]
        
        originpoint = list(map(int.__sub__, list(origin), list(originpoint)))
        originpoint = np.array(originpoint) * worlddistance / distance

        otherpoint = list(map(int.__sub__, list(origin), (ix[0], iy[0])))
        otherpoint = np.array(otherpoint) * worlddistance / distance
        print("Origin Point + 100px:")
        print(originpoint)
        print("First click: ")
        print(otherpoint)
        theta = angle_between(originpoint, otherpoint)
        rotation = [[np.cos(theta), -1 * np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        print("Angle: ")
        print((theta * 180 / math.pi))
        print("Rotation Matrix: ")
        print(rotation)
        
        normalizedcenterss = []
        for x in normalizedcenters:
            x[1] = -1 * x[1]
            # print(x)
            normalizedcenterss.append(rotation * x)


        #Plotting the UWBData to see what is going on
        for x in uwbData:
            x1 = (float(x[0]) * distance / worlddistance, float(x[1]) * distance / worlddistance)
            x1 = list(map(float.__add__, (float(origin[0]), float(origin[1])), (float(x1[0]), float(x1[1]))))
            x1 = (math.floor(x1[0] + origin[0]), math.floor(x1[1] + origin[1]))
            print(x1)
            frame = cv.circle(frame, (round(x1[0]), round(x1[1])), 7, (0, 0, 255), -1)
            
        # cv.imshow('Window', frame)
        
        val = input("Enter height of drone: ")
        heights = []
        # print(val)
        for x in uwbData:
            closest = []
            distance = []
            x = (float(x[0]), float(x[1]))
            for g in normalizedcenters:
                t = line_to_point_distance(np.array(droneLocation), np.array([x[0], x[1]]), np.array([g[0], g[1]]))
                distance.append(math.dist(x, g))
                closest.append(t)
            index = closest.index(min(closest))
            # if index == distance.index(min(distance)):
            #     point = normalizedcenters[index]
            # else:
            #     closest.pop(index)
            #     distance.pop(index)
            #     point = normalizedcenters[closest.index(min(closest))]
            point = normalizedcenters[index]
            R = math.dist(point, droneLocation)
            R1 = math.dist(point, x)
            height = float(val) * float(R1) / float(R)
            heights.append(height)
        print(heights)
    
        with open(f'results/{filename}.height.csv', 'w') as the_file:
            writer = csv.writer(the_file)
            writer.writerow(heights)
            the_file.flush()
        #Open a csv file in the results folder and write the data to the file, which has the same name as the video
        with open(f'results/{filename}.csv', 'w') as the_file:
            writer = csv.writer(the_file)
            for row in normalizedcenters:
                writer.writerow(row)
            the_file.flush()
        
cv.destroyAllWindows()
