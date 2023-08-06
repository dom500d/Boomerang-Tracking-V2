import numpy as np
import cv2 as cv
import math
import numpy as np
import csv
from os import listdir

distance = 0
worlddistance = 6.4008

def select_points(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDBLCLK:
        ix.append(x)
        iy.append(y)

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

feature_params = dict(maxCorners=200, qualityLevel=0.2, minDistance=7, blockSize=5)
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03),
)
color = np.random.randint(0, 255, (100, 3))

cv.namedWindow('Window', cv.WINDOW_NORMAL)
for filename in listdir("C:/Users/rorla/Desktop/Boomerang-Tracking-V2/videos/"):
    #Make sure the program only runs on video files, if we use a different format just change it here as
    #long as opencv supports it 
    if filename.endswith(".MP4") or filename.endswith(".mp4"):
        centers = []
        ix = []
        iy = []

        #Create a window that is scaled to monitor size, not set to the actual size of the video being
        #used
  
        cap = cv.VideoCapture(f"videos/{filename}")
        ret, frame = cap.read()
        
        #Call function to set a distance inside the image to be equal to 21 feet in real-world units
        getDistance(frame)

        # backSub = cv.createBackgroundSubtractorKNN()

        #Create a MOG2 background subtractor object, initlizaing the thershold to 6000 to try and account for instability of the drone
        backSub = cv.createBackgroundSubtractorMOG2(varThreshold= 6000)
        while (1):
            ret, frame = cap.read()
            if not ret:
                print('No frames grabbed!')
                break

            #Utilize Gaussain Blurring to decrease noise but also minimize the effect of lighting on the algorithm
            # frame = cv.GaussianBlur(frame, (15,15), 3)

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
            # cv.imshow('Window', final)

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
                # goodcontours.append(c)

                #This checks to see if the area of the contour is above a certain amount, helps reduce the noise
                #or not boomerang stuff that makes it through all the earlier image processing
                if M['m00'] > 30:
                    x, y, w, h = cv.boundingRect(c)

                    #Checks to see if the width and height of the contour are within an acceptable range before
                    #accepting it. Can be tweaked as well
                    if w < 30 and h < 30:
                        goodcontours.append(c)

                        #Compute the cetner of the contour with the moments, and then store those to the centers
                        #variable to eventually store in csv file. We also put a red dot where the computed centroid
                        #is
                        cx = int(M['m10']/M['m00'])
                        cy = int(M['m01']/M['m00'])
                        centers.append((cx, cy))
                        cv.circle(frame, (cx, cy), 7, (0, 0, 255), -1)

            #Draw all the good contours on the original image so that you can see how well the program is tracking
            #the boomerang.
            cv.drawContours(frame, goodcontours, -1, (0, 255, 0), 3)
            cv.imshow('Window', frame)
            if keyboard == 'q' or keyboard == 27:
                break

        normalizedcenters = []   

        #Initialize the origin for the data to be the first contour drawn (typically where the boomerang was thrown from)
        origin = centers[0]

        #Go through every data point in centers, and compute what the distance is in real-world units (meters)
        #from the approximation received from the getDistance function
        for c in centers:
            a = list(map(int.__sub__, list(origin), list(c)))
            b = np.array(a) * worlddistance / distance
            normalizedcenters.append(tuple(b))

        #Open a csv file in the results folder and write the data to the file, which has the same name as the video
        with open(f'results/{filename}.csv', 'w') as the_file:
            writer = csv.writer(the_file)
            for row in normalizedcenters:
                writer.writerow(row)
            the_file.flush()

cv.destroyAllWindows()

            # ret, frame2 = cap.read()
            # if not ret:
            #     print('No frames grabbed!')
            #     break

            # bgr_inv = ~frame2
            # hsv_inv = cv.cvtColor(bgr_inv, cv.COLOR_BGR2HSV);
            # mask = cv.inRange(hsv_inv, np.array([90 - 10, 70, 50]), np.array([90 + 10, 255, 255]))
            # final = cv.bitwise_and(frame2, frame2, mask=mask)
            # next = cv.cvtColor(final, cv.COLOR_HSV2BGR)
            # next = cv.cvtColor(next, cv.COLOR_BGR2GRAY)

            # #cv.imshow('Window', next)
            # p1, st, err = cv.calcOpticalFlowPyrLK(
            #     old_gray, next, p0, None, **lk_params
            # )
            # good_new = p1[st == 1]
            # good_old = p0[st == 1]
            # mask1 = np.zeros_like(frame2)
            # for i, (new, old) in enumerate(zip(good_new, good_old)):
            #     a, b = np.floor(new.ravel())
            #     c, d = np.floor(old.ravel())
            #     mask1 = cv.line(mask1, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            #     frame = cv.circle(frame2, (int(a), int(b)), 5, color[i].tolist(), -1)
 
            # img = cv.add(frame, mask1)
            # cv.imshow("Window", img)
            # k = cv.waitKey(25) & 0xFF
            # if k == 27:
            #     break
 
            # old_gray = next.copy()
            # p0 = good_new.reshape(-1, 1, 2)



            # p1, st, err = cv.calcOpticalFlowPyrLK(prvs, next, p0, None, **lk_params)
            # color = np.random.randint(0, 255, (100, 3))


            # if p1 is not None:
            #     good_new = p1[st==1]
            #     good_old = p0[st==1]
            # # draw the tracks
            # for i, (new, old) in enumerate(zip(good_new, good_old)):
            #     a, b = new.ravel()
            #     c, d = old.ravel()
            #     mask1 = cv.line(mask1, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            #     frame = cv.circle(frame2, (int(a), int(b)), 5, color[i].tolist(), -1)
            # img = cv.add(frame, mask1)
            # cv.imshow('Window', img)
            # k = cv.waitKey(30) & 0xff
            # if k == 27:
            #     break
            # # Now update the previous frame and previous points
            # prvs = next.copy()
            # p0 = good_new.reshape(-1, 1, 2)


    #         flow = cv.calcOpticalFlowFarneback(
    #             prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    #         mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    #         hsv[..., 0] = ang*180/np.pi/2
    #         hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    #         bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    #         bgr1 = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    #         x, bgr2 = cv.threshold(bgr1, 35, 255, cv.THRESH_BINARY)
    #         #cv.imshow('Window', bgr2)
    #         contours, hierarchy = cv.findContours(
    #             bgr2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    #         goodcontours = []
    #         for c in contours:
    #             if cv.contourArea(c) > 50:
    #                 x, y, w, h = cv.boundingRect(c)
    #                 cut_image = frame2[y:y+h, x:x+w]
    #                 cut_image1 = cv.cvtColor(cut_image, cv.COLOR_BGR2HSV)
    #                 bgr_inv = ~cut_image1
    #                 hsv_inv = cv.cvtColor(bgr_inv, cv.COLOR_BGR2HSV);
    #                 mask = cv.inRange(hsv_inv, np.array([90 - 10, 70, 50]), np.array([90 + 10, 255, 255]))
    #                 final = cv.bitwise_and(frame2, frame2, mask=mask)
    #                 mean = np.average(final[:, :, 0]).astype(np.uint8)
    #                 if mean != 0:
    #                     cv.rectangle(frame2, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #                     centers.append((x, y))
    #                     goodcontours.append(c)
    #                 # cv.rectangle(frame2, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #                 # centers.append((x, y))
    #                 # goodcontours.append(c)
    #         cv.drawContours(frame2, goodcontours, -1, (0, 255, 0), 3)
    #         cv.imshow('Window', frame2)

    #         k = cv.waitKey(1) & 0xff
    #         prvs = next.copy()
    #     
    #     cv.destroyAllWindows()
    #     continue
    # else:
    #     continue



#centers = [(1363, 636), (1368, 641), (1368, 644), (1367, 647), (1377, 650), (1383, 653), (1388, 659), (1390, 663), (1384, 667), (1381, 666), (1368, 667), (1351, 660), (1368, 723), (1373, 687), (1347, 688), (1361, 687), (1344, 737), (1354, 694), (1356, 891), (1452, 727), (1409, 694), (1354, 990), (1351, 919), (1418, 702), (1349, 988), (1414, 708), (1366, 1107), (1347, 1036), (1358, 1088), (1351, 1134), (1364, 1234), (1381, 1233), (1394, 1295), (1380, 1269), (1371, 1273), (1394, 1357), (1390, 1373), (1419, 1448), (1409, 1460), (1436, 1464), (1460, 1581), (1447, 1529), (1482, 1606), (1487, 1583), (1522, 1645), (1483, 1616), (1511, 1674), (1510, 1633), (1543, 1704), (1523, 1698), (1591, 1765), (1603, 1770), (1675, 1849), (2033, 1057), (2010, 1048), (1466, 744), (1705, 1875), (2009, 1049), (1462, 741), (1719, 1873), (1472, 752), (1731, 1883), (1469, 752), (1753, 1901), (1798, 1923), (1817, 1930), (1849, 1940), (1870, 1949), (1893, 1952), (1922, 1953), (1939, 1934), (1979, 1966), (1453, 744), (2017, 1959), (2040, 1960), (2070, 1945), (2088, 1945), (2116, 1943), (2011, 1021), (1427, 741), (2158, 1949), (2175, 1928), (2011, 1022), (1429, 737), (1464, 724), (2172, 1924), (2200, 1919), (2009, 1023), (2195, 1914), (2243, 1888), (2230, 1868), (2251, 1856), (2282, 1855), (2327, 1847), (2307, 1846), (1991, 1022), (1450, 755), (2301, 1841), (2331, 1834), (1430, 731), (1447, 755), (2371, 1786), (1996, 1020), (1435, 730), (2363, 1770), (2379, 1751), (2410, 1752), (2417, 1715), (2403, 1723), (2422, 1703), (2408, 1671), (2406, 1635), (2450, 1621), (2463, 1595), (2471, 1573), (2458, 1541), (2467, 1510), (2485, 1485), (2476, 1460), (2473, 1431), (2475, 1400), (2485, 1376), (2487, 1355), (2484, 1312), (2474, 1280), (2475, 1272), (2473, 1242), (2461, 1200), (2454, 1176), (1480, 717), (2443, 1125), (1486, 726), (2446, 1132), (2436, 1106), (1488, 726), (2421, 1084), (2400, 1049), (2365, 993), (2348, 956), (2319, 937), (2319, 921), (2302, 892), (2267, 857), (2256, 840), (2234, 822), (2206, 792), (2187, 758), (2156, 740), (2121, 727), (1985, 1019), (2092, 690), (1982, 1019), (2063, 668), (2031, 651), (1997, 637), (1981, 622), (1948, 603), (1923, 586), (1889, 575), (1862, 565), (1840, 550), (1808, 536), (1764, 519), (1738, 516), (1712, 488), (1691, 496), (1662, 490), (1630, 481), (1601, 437), (1564, 432), (1552, 455), (1520, 424), (1494, 414), (1450, 418), (1429, 410), (1398, 409), (1392, 409), (1361, 413), (1352, 407), (1317, 404), (1293, 405), (1288, 397), (1261, 401), (1249, 402), (1228, 398), (1187, 398), (1182, 411), (1156, 410), (1154, 412), (1144, 413), (1461, 656), (1123, 418), (1458, 654), (1112, 420), (1094, 425), (1070, 425), (1070, 425), (1073, 442), (1061, 438), (1050, 444), (1050, 448), (1460, 656), (1030, 449), (1454, 651), (1033, 458), (1014, 462), (1010, 463), (1013, 470), (1406, 642), (1013, 477), (1405, 641), (1445, 648), (1007, 475), (1404, 641), (1006, 492), (999, 475), (1007, 482), (1395, 630), (1013, 502), (1432, 666), (1392, 639), (993, 502), (1390, 626), (1007, 512), (994, 506), (1371, 625), (996, 520), (1372, 620), (1006, 531), (1018, 547), (1374, 619), (1006, 542), (1012, 544), (1006, 554), (982, 580), (978, 580), (973, 582), (2021, 1028), (970, 575), (968, 579), (2016, 1014), (1326, 614), (965, 584), (2019, 1025), (1321, 616), (963, 589), (2007, 1012), (960, 584), (2015, 1013), (959, 589), (1999, 1060), (2012, 1013), (978, 587), (2016, 1015), (2020, 1038), (2293, 170), (2009, 1015), (2008, 1015), (2010, 1016), (2011, 1017), (2012, 1017), (2013, 1016), (2014, 1016), (2009, 1014), (2007, 1013), (982, 569), (2007, 1015), (1991, 1016), (1992, 1018), (1992, 1022), (961, 568), (1997, 1019), (956, 569)]


# scatter = plt.scatter(*zip(*normalizedcenters))
# plt.xlabel('Feet', fontsize=20)
# plt.ylabel('Feet', fontsize=20)
# scatter.axes.invert_xaxis()
# print(normalizedcenters)
# with open('/output.csv', "w") as the_file:
#     writer = csv.writer(the_file)
#     for row in normalizedcenters:
#         writer.writerow(row)
#         print(row)
#     the_file.flush()
# plt.show()
# cv.destroyAllWindows()
