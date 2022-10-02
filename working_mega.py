



# imports
from tkinter import *
import tkinter.filedialog as tk_file
from PIL import Image, ImageTk
import cv2
import numpy as np
import requests
import imutils
import pytesseract
import json
import csv

from pyexcel.cookbook import merge_all_to_a_book
import glob

url = "http://10.114.168.71:8080/shot.jpg"
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



patient_details = {}
hospital_details = {}

def get_text():
    #img = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
    #img_pil = Image.fromarray(img)
    # converts the image to result and saves it into result variable
    # final_image.show()
    result = pytesseract.image_to_string(final_image)

    lines = result.split('\n')
    lines = list(filter(None, lines))

    list_len = int(len(lines)/4)

    test_name = lines[:list_len]
    results = lines[list_len:list_len*2]
    units = lines[list_len*2:list_len*3]
    reference = lines[list_len*3:list_len*4]


        
    # field names 
    fields = ['Test', 'Test Value', 'Unit', 'Range'] 
    
    rows = []

    # name of csv file 
    filename = "data.csv"
    
    test_results = {}
    for i in range(list_len):
        rows.append([test_name[i], results[i], units[i], reference[i]])
        test_results[test_name[i]] = {
            'value': results[i],
            'unit': units[i],
            'reference': reference[i]
        }
        
        
    # writing to csv file 
    with open(filename, 'w') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
            
        # writing the fields 
        csvwriter.writerow(fields) 
            
        # writing the data rows 
        csvwriter.writerows(rows)

    merge_all_to_a_book(glob.glob("data.csv"), "output.xlsx")

    new_list = []
    for i in range(list_len):
        new_list.append( test_name[i] + " " + results[i] + " " + units[i] + " " + reference[i])
    
    print(test_results)


    patient_details['results'] = test_results
    json_data = json.dumps(patient_details, indent=4)

    print(json_data)
    with open('./node/db.json', 'w') as outfile:
        outfile.write(json_data)

    frame.destroy()
    root.destroy()

def capture_form():
    global final_image

    retake_button = Button(frame, text = 'Retake',
        command=display_webcam)
    retake_button.pack()
    submit_button['text'] = 'Confirm'
    submit_button['command'] = crop_window
    frame.after_cancel(image_update)

    #pil_image = ImageTk.getimage(imgtk)
    #image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    #cv2.imwrite("D:\hackathon\Silicon Rush" + "\letsImage2.jpg", image)
    image = original_img
    # cv2.imshow("IMG",image)

    #blurred_threshold = transformation(image)
    #cv2.imshow("blurred",blurred_threshold)
    cleaned_image = final_image(image)
    # cv2.imshow("cleaned", cleaned_image)

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

def authorize(entries):
    hospital_details['hospital_id'] = entries['Hospital_Id'].get()
    hospital_details['password'] = entries['Password'].get()

    print('authorized')

    for widget in frame.winfo_children():
        widget.destroy()
    put_form()

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
    
    fields = ('Name', 'Patient Number', 'Aadhaar Number')
    ents = makeform(frame, fields)
    capture_button = Button(frame, text = 'Submit Details',
        command=(lambda e = ents: capture_details(e)))
    capture_button.pack()


def put_authorization():
    fields = ('Hospital_Id','Password')
    ents = makeform(frame, fields)
    capture_button = Button(frame, text = 'Submit Details',
        command=(lambda e = ents: authorize(e)))
    capture_button.pack()




# def crop_image(left,top,right,bottom):
#     print(left, top, bottom, right)
#     global edited_images, opened_image

    

#     print('entered')ed

#     img = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)

#     edit_img = Image.fromarray(img)

#     image_width, image_height = edit_img.size

#     edited_images = edit_img.crop((left,top,image_width - right,image_height - bottom))
#     edited_images.save("edited.jpeg")
#     opened_images = ImageTk.PhotoImage(edited_images)


#     image_lb.config(image = opened_images)

def clear_image():
    global opened_image, image_path,image_height,image_width

    opened_image = ''
    image_path = ''
    image_height = 0
    image_width = 0

def confirm_image():
    global final_image

    #print(edited_image)
    #if not edited_image:
    #    edited_image = Image.open(image_path)
    if edited_image == '':
        img = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        final_image = img_pil
    else:
        final_image = edited_image
    get_text()


def controls_window(w,h):
    controls = Toplevel(root)
    controls.geometry('400x300')

    # print(w,h)
    # w=400
    # h=300

    w *= 2
    h *= 2

    left_lb = Label(controls, text='Left')
    left_lb.pack(anchor=W, pady=5)

    left_scale = Scale(controls, from_=0, to=w, orient=HORIZONTAL,
                          command = lambda x: crop_image(left_scale.get(),
                                   top_scale.get(),
                                   right_scale.get(),
                                   bottom_scale.get()))
    left_scale.pack(anchor=W, fill=X)

    right_lb = Label(controls, text='Right')
    right_lb.pack(anchor=W, pady=5)

    right_scale = Scale(controls, from_=0, to=w, orient=HORIZONTAL, 
                           command = lambda x: crop_image(left_scale.get(),
                                   top_scale.get(),
                                   right_scale.get(),
                                   bottom_scale.get()))
    right_scale.pack(anchor=W, fill=X)

    top_lb = Label(controls, text='Top')
    top_lb.pack(anchor=W, pady=5)

    top_scale = Scale(controls, from_=0, to=h, orient=HORIZONTAL, 
                         command = lambda x: crop_image(left_scale.get(),
                                   top_scale.get(),
                                   right_scale.get(),
                                   bottom_scale.get()))
    top_scale.pack(anchor=W, fill=X)

    bottom_lb = Label(controls, text='Bottom ')
    bottom_lb.pack(anchor=W, pady=5)

    bottom_scale = Scale(controls, from_=0, to=h, orient=HORIZONTAL,
                            command = lambda x: crop_image(left_scale.get(),
                                   top_scale.get(),
                                   right_scale.get(),
                                   bottom_scale.get()))
    bottom_scale.pack(anchor=W, fill=X)

def save_image():
    if image_path:
        file_name = tk_file.asksaveasfilename()

        if file_name:
            edited_image.save(file_name)

def crop_image(left,top,right,bottom):
    global edited_image, opened_image

    print(image_path)
    edit_img = Image.open(image_path)
    image_width = edit_img.width
    image_height = edit_img.height
    edited_image = edit_img.crop((left,top,image_width - right,image_height - bottom))
    
    h = edited_image.height
    w = edited_image.width 

    reduced_preview = edited_image.resize((int(w*0.3), int(h*0.3)))
    opened_image = ImageTk.PhotoImage(reduced_preview)


    image_lb.config(image = opened_image)

def open_image():
    global opened_image,image_path, image_height,image_width

    image_path = tk_file.askopenfilename()

    if image_path:
        img = Image.open(image_path)
        w = img.width 
        h = img.height
        img = img.resize((int(w*0.3), int(h*0.3)))
        opened_image = ImageTk.PhotoImage(img)
        image_lb.config(image=opened_image)

        image_height = opened_image.height()
        image_width = opened_image.width()

def load_image():
    global opened_image,image_path, image_height,image_width

    color_coverted = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(color_coverted)
    image_path = 'myfile.png'
    pil_image.save(image_path)

    w = pil_image.width
    h = pil_image.height
    pil_image = pil_image.resize((int(w*0.3), int(h*0.3)))
    opened_image = ImageTk.PhotoImage(pil_image)
    
    image_lb.config(image=opened_image)

    image_height = opened_image.height()
    image_width = opened_image.width()

def crop_window():
    global opened_image, edited_image, image_height, image_width, image_lb

    frame.destroy()

    root.geometry('800x600')
    root.config(bg="black")

    opened_image=''
    edited_image = ''

    image_height = 0
    image_width = 0

    menu_bar = Menu(root)
    menu_bar.add_command(label = "Open", command=open_image)
    menu_bar.add_command(label = "Controls", command=lambda: controls_window(w=image_width, h = image_height))
    menu_bar.add_command(label = "Confirm_Crop", command= confirm_image)
    menu_bar.add_command(label = "Clear", command=clear_image)

    root.config(menu = menu_bar)

    image_lb = Label(root, bg='gray')
    image_lb.pack(fill=BOTH)

    load_image()


if __name__ == '__main__':
    global frame, root
    root = Tk()
    root.title('Report Details')
    frame = Frame(root)
    frame.pack() 

    put_authorization()

    root.mainloop()