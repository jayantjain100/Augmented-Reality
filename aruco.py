#hepler functions to detect the aruco marker

import numpy as np 
import cv2 
import sys
  
def display(img, f = 1):
    #takes an image as input and scaling factor and displays
    img = scale(img, f)
    cv2.imshow('dummy', img) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def scale(image, f):
    #scales an image acc to f
    h = int(image.shape[0]*f)
    w = int(image.shape[1]*f)
    return cv2.resize(image, (w,h), interpolation = cv2.INTER_CUBIC )

def get_bit_sig(image, contour_pts, thresh = 127):
    ans = []

    a, b = contour_pts[0][0]
    c, d = contour_pts[1][0]
    e, f = contour_pts[3][0]
    g, h = contour_pts[2][0]

    for i in range(8):
        for j in range(8):
            #using bilinear interpolation to find the coordinate using fractional contributions of the corner 4 points
            f1 = float(i)/8 + 1./16
            f2 = float(j)/8 + 1./16

            upper_x = (1-f1)*a + f1*(c)
            lower_x = (1-f1)*e + f1*(g)
            upper_y = (1-f1)*b + f1*d
            lower_y = (1-f1)*(f) + f1*(h)

            x = int( (1-f2)*upper_x + (f2)*lower_x )
            y = int( (1-f2)*upper_y + (f2)*lower_y )

            # if patch_avg(image, y, x) >= 127:
            if image[y][x] >= 127:
                ans.append(1)
            else:
                ans.append(0)
    return ans

def match_sig(sig1, sig2, thresh = 62):
    # print(sum([ (1- abs(a - b)) for a, b in zip(sig1, sig2)]))
    if sum([ (1- abs(a - b)) for a, b in zip(sig1, sig2)]) >= 62:
        return True
    else:
        return False

def find_pattern_aruco(image, aruco_marker, sigs):
    #image is supposed to be the unflipped frame
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   
    threshold = cv2.adaptiveThreshold(gray, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,10)
    h, w = aruco_marker.shape
    _, contours ,_= cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    found = False
    for cnt in contours : 
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True) 
        if approx.shape[0]==4:
            x1 = approx[0][0][0] 
            x2 = approx[1][0][0]
            y1 = approx[0][0][1]
            y2 = approx[1][0][1]

            norm = (x1 - x2)**2 + (y1 - y2)**2
            if norm > 100:
                temp_sig = get_bit_sig(gray, approx)
                match1 = match_sig(sigs[0], temp_sig)
                match2 = match_sig(sigs[1], temp_sig)
                match3 = match_sig(sigs[2], temp_sig)
                match4 = match_sig(sigs[3], temp_sig)

                if (match1 or match2 or match3 or match4):
                    dst_pts = approx
                    found = True
                    if match1:
                        src_pts = np.array([[0,0],[0,w], [h,w], [h,0]])
                    if match2:
                        src_pts = np.array([[0,w], [h,w], [h,0], [0,0]])
                    if match3:
                        src_pts = np.array([[h,w],[h,0], [0,0], [0,w]])
                    if match4:
                        src_pts = np.array([[h,0],[0,0], [0,w], [h,w]])

                    cv2.drawContours(image, [approx], 0, (0, 0, 255), 2) #mark red outline for found marker 

                    return src_pts, dst_pts, True

    return None, None, False              

   
def find_homography_aruco(image, aruco_marker, sigs):
    src_pts, dst_pts, found = find_pattern_aruco(image, aruco_marker, sigs)
    H = None
    if found:
        H, mask = cv2.findHomography(src_pts.reshape(-1,1,2), dst_pts.reshape(-1,1,2), cv2.RANSAC,5.0)

    if H is None:
        return False, None
    else:
        return True, H
