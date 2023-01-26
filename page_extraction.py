import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def order_points(pts):
    '''Rearrange coordinates to order:
      top-left, top-right, bottom-right, bottom-left'''
    rect = np.zeros((4, 2), dtype='float32')
    pts = np.array(pts)
    s = pts.sum(axis=1)
    # Top-left point will have the smallest sum.
    rect[0] = pts[np.argmin(s)]
    # Bottom-right point will have the largest sum.
    rect[2] = pts[np.argmax(s)]
 
    diff = np.diff(pts, axis=1)
    # Top-right point will have the smallest difference.
    rect[1] = pts[np.argmin(diff)]
    # Bottom-left will have the largest difference.
    rect[3] = pts[np.argmax(diff)]
    # Return the ordered coordinates.
    return rect.astype('int').tolist()


"""
Given a mask on an image and the image, returns the rectangular region enclosing the mask. 
"""
def extract_box(img, mask):
    indices = np.argwhere(mask > 0)
    mins = np.min(indices, axis=0)
    min_x = mins[0]
    min_y = mins[1]

    maxs = np.max(indices, axis=0)
    max_x = maxs[0]
    max_y = maxs[1]

    return img[ min_x - 10 : max_x + 10, min_y - 10: max_y + 10,]
    

   #print(indices[0])

def snap_page(img):
    orig_img = img
    # Repeated Closing operation to remove text from the document.
    # Refer to: https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
    kernel = np.ones((5,5),np.uint8)
    img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel, iterations= 3)


    #Extract Foreground from Background. CV2.GrabCut algorithm is run to capture the document. 
    # Refer to: https://docs.opencv.org/4.6.0/d3/d47/group__imgproc__segmentation.html#ga909c1dda50efcbeaa3ce126be862b37f
    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (20,20,img.shape[1]-20,img.shape[0]-20)
    cv.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]


    #Edge detection after the background has been removed. 
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (11, 11), 0)
    canny = cv.Canny(gray, 0, 200)
    canny = cv.dilate(canny, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))

    # Blank canvas.
    con = np.zeros_like(img)
    # Finding contours for the detected edges.
    contours, hierarchy = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    # Keeping only the largest detected contour.
    page = sorted(contours, key=cv.contourArea, reverse=True)[:5]
    con = cv.drawContours(con, page, -1, (0, 255, 255), 3)
    cv.imshow('Image', con)

    # Wait for a key press
    cv.waitKey(0)

    # Close the display window
    cv.destroyAllWindows()  

    # Blank canvas.
    con = np.zeros_like(img)
    # Loop over the contours.
    for c in page:
    # Approximate the contour.
        epsilon = 0.02 * cv.arcLength(c, True)
        corners = cv.approxPolyDP(c, epsilon, True)
        # If our approximated contour has four points
        if len(corners) == 4:
            break
    cv.drawContours(con, c, -1, (0, 255, 255), 3)


    cv.drawContours(con, corners, -1, (0, 255, 0), 10)
    
    # Sorting the corners and converting them to desired shape.
    corners = sorted(np.concatenate(corners).tolist())
    
    # Displaying the corners.
    for index, c in enumerate(corners):
        character = chr(65 + index)
        cv.putText(con, character, tuple(c), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv.LINE_AA)


    (tl, tr, br, bl) = order_points(corners)

    # Finding the maximum width.
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # Finding the maximum height.
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # Final destination co-ordinates.
    destination_corners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]

    # Getting the homography.
    M = cv.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))
    # Perspective transform using homography.
    final = cv.warpPerspective(orig_img, M, (destination_corners[2][0], destination_corners[2][1]), flags=cv.INTER_LINEAR)
    cv.imshow('Image', final)

    # Wait for a key press
    cv.waitKey(0)

    # Close the display window
    cv.destroyAllWindows()  
    return final



img = cv.imread('page-3.jpeg')

img = cv.rotate(snap_page(img), cv.ROTATE_90_COUNTERCLOCKWISE)
# Convert BGR to HSV

hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# Range for upper range
yellow_lower = np.array([20, 100, 100])
yellow_upper = np.array([30, 255, 255])
mask_yellow = cv.inRange(hsv, yellow_lower, yellow_upper)


yellow_output = cv.bitwise_and(img, img, mask=mask_yellow)
final = extract_box(img, yellow_output)


#yellow_output = cv.dilate(yellow_output,cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))
cv.imshow('Image', final)

# Wait for a key press
cv.waitKey(0)

# Close the display window
cv.destroyAllWindows()

