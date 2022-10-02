
# imports
from tkinter import *
from PIL import Image, ImageTk
import cv2
import numpy as np
import requests
import imutils
import pytesseract
import json

url = "http://192.168.165.19:8080/shot.jpg"
pytesseract.pytesseract.tesseract_cmd ='C:/Program Files/Tesseract-OCR/tesseract.exe' 

data = {}

def is_float(n):
    try:
        float(n)
    except ValueError:
        return False
    return True

def blur_and_threshold(gray):
    gray = cv2.GaussianBlur(gray, (3, 3), 2)
    threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    threshold = cv2.fastNlMeansDenoising(threshold, 11, 31, 9)
    return threshold


# ## **Find the Biggest Contour**

# **Note: We made sure the minimum contour is bigger than 1/10 size of the whole picture. This helps in removing very small contours (noise) from our dataset**


def biggest_contour(contours, min_area):
    biggest = None
    max_area = 0
    biggest_n = 0
    approx_contour = None
    for n, i in enumerate(contours):
        area = cv2.contourArea(i)

        if area > min_area / 10:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
                biggest_n = n
                approx_contour = approx

    return biggest_n, approx_contour


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


# ## Find the exact (x,y) coordinates of the biggest contour and crop it out


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


# # Transformation the image

# **1. Convert the image to grayscale**

# **2. Remove noise and smoothen out the image by applying blurring and thresholding techniques**

# **3. Use Canny Edge Detection to find the edges**

# **4. Find the biggest contour and crop it out**


def transformation(image):
    image = image.copy()

    height, width, channels = image.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_size = gray.size

    threshold = blur_and_threshold(gray)
    # We need two threshold values, minVal and maxVal. Any edges with intensity gradient more than maxVal
    # are sure to be edges and those below minVal are sure to be non-edges, so discarded.
    #  Those who lie between these two thresholds are classified edges or non-edges based on their connectivity.
    # If they are connected to "sure-edge" pixels, they are considered to be part of edges.
    #  Otherwise, they are also discarded
    edges = cv2.Canny(threshold, 50, 150, apertureSize=7)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    simplified_contours = []

    for cnt in contours:
        hull = cv2.convexHull(cnt)
        simplified_contours.append(cv2.approxPolyDP(hull,
                                                    0.001 * cv2.arcLength(hull, True), True))
    simplified_contours = np.array(simplified_contours,dtype=object)
    biggest_n, approx_contour = biggest_contour(simplified_contours, image_size)

    threshold = cv2.drawContours(image, simplified_contours, biggest_n, (0, 255, 0), 1)

    dst = 0
    if approx_contour is not None and len(approx_contour) == 4:
        approx_contour = np.float32(approx_contour)
        dst = four_point_transform(threshold, approx_contour)
    croppedImage = dst
    return croppedImage


# **Increase the brightness of the image by playing with the "V" value (from HSV)**

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


# **Sharpen the image using Kernel Sharpening Technique**


def final_image(rotated):
    # Create our shapening kernel, it must equal to one eventually
    kernel_sharpening = np.array([[0, -1, 0],
                                  [-1, 5, -1],
                                  [0, -1, 0]])
    # applying the sharpening kernel to the input image & displaying it.
    sharpened = cv2.filter2D(rotated, -1, kernel_sharpening)
    sharpened = increase_brightness(sharpened, 30)
    return sharpened


fields = ('Name', 'Patient Number', 'Aadhaar Number')

patient_details = {}

def get_text():
    img = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    # converts the image to result and saves it into result variable
    result = pytesseract.image_to_string(img)

    with open('abc.txt',mode ='w') as file:     
        file.write(result)

    result = result.replace(',', '.')
    lines = result.split('\n')
    print(result)

    labels = [
        'Urea',
        'Creatinine',
        'Uric Acid',
        'Calcium, Total',
        'Phosphorous',
        'Alkaline Phosphatase (ALP)',
        'Total Protein',
        'Albumin',
        'A : G Ratio',
        'Sodium',
        'Potassium',
        'Chloride'
    ]


    def is_float(n):
        try:
            float(n)
        except ValueError:
            return False
        return True

    def is_special(line):
        #if '-' in line:
        #    return False
        words = line.split()
        for word in words:
            if is_float(word):
                return True
        return False

    special_lines = []

    for line in lines:
        if not len(line.strip()):
            continue
        if is_special(line):
            special_lines.append(line)
        else:
            special_lines = []
        if len(special_lines) == len(labels):
            break

    values = []
    for line in special_lines:
        print(line)
        for word in line.split():
            if is_float(word):
                values.append(float(word))
                break

    results = {}
    for i in range(len(labels)):
        results[labels[i]] = values[i]

    json_data = json.dumps(results, indent=4)
    print(json_data)

    frame.destroy()

def capture_form():
    global final_image

    retake_button = Button(frame, text = 'Retake',
        command=display_webcam)
    retake_button.pack()
    submit_button['text'] = 'Confirm'
    submit_button['command'] = get_text
    frame.after_cancel(image_update)

    #pil_image = ImageTk.getimage(imgtk)
    #image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    #cv2.imwrite("D:\hackathon\Silicon Rush" + "\letsImage2.jpg", image)
    image = original_img
    cv2.imshow("IMG",image)

    # blurred_threshold = transformation(image)
    # cv2.imshow("blurred",blurred_threshold)
    cleaned_image = final_image(image)
    cv2.imshow("cleaned", cleaned_image)

    final_image = cleaned_image
    cv2.imwrite("D:\hackathon\Silicon Rush" + "\letsssImage.jpg", cleaned_image)
    
    cleaned_image = cv2.resize(cleaned_image, (0, 0), fx = 0.3, fy = 0.3)
    cleaned_image= cv2.cvtColor(cleaned_image,cv2.COLOR_BGR2RGB)
    cleaned_image = Image.fromarray(cleaned_image)

    new_imgtk = ImageTk.PhotoImage(image = cleaned_image)

    label.imgtk = new_imgtk
    label.configure(image=new_imgtk)



def display_webcam():
    global frame
    global submit_button
    global label

    for widget in frame.winfo_children():
        widget.destroy()

    label = Label(frame)
    label.pack()
    cap = cv2.VideoCapture(0)

    def show_frames():  
        global imgtk
        global image_update
        global original_img
        # Get the latest frame and convert into Image
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        # img_arr = np.rot90(img_arr, axes = (0, 1))

        img = cv2.imdecode(img_arr, -1)
        img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)

        # img = imutils.resize(img, width=1200, height=1800)
        # print(img.shape)
        width, height,_ = img.shape
        # print(width,height)
        original_img = img
        img = cv2.resize(img, (0, 0), fx = 0.3, fy = 0.3)

        # img = imutils.resize(img, width = width/2,  height =height/2)

        img= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        # Convert image to PhotoImage
        imgtk = ImageTk.PhotoImage(image = img)
        label.imgtk = imgtk
        label.configure(image=imgtk)
        # Repeat after an interval to capture continiously
        image_update = frame.after(20, show_frames)
    show_frames()
    submit_button = Button(frame, text = 'Capture Form',
        command=capture_form)
    submit_button.pack()

def capture_details(entries):
    patient_details['name'] = entries['Name'].get()
    patient_details['patient_no'] = entries['Patient Number'].get()
    patient_details['aadhaar_no'] = entries['Aadhaar Number'].get()
    data['patient_detais'] = patient_details
    
    display_webcam()

def makeform(root, fields):
   entries = {}
   for field in fields:
      row = Frame(root)
      lab = Label(row, width=22, text=field+": ", anchor='w')
      ent = Entry(row)
      row.pack(side = TOP, fill = X, padx = 5 , pady = 5)
      lab.pack(side = LEFT)
      ent.pack(side = RIGHT, expand = YES, fill = X)
      entries[field] = ent
   return entries

def put_form():
    ents = makeform(frame, fields)
    capture_button = Button(frame, text = 'Submit Details',
        command=(lambda e = ents: capture_details(e)))
    capture_button.pack()


if __name__ == '__main__':
    global frame
    root = Tk()
    root.title('Report Details')
    frame = Frame(root)
    frame.pack() 


    root.mainloop()