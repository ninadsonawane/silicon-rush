from PIL import Image
import pandas as pd
import numpy as np
import re,string,unicodedata

#Tesseract Library
import pytesseract

#Warnings
import warnings
warnings.filterwarnings("ignore")
import gc

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def convert_img_to_text(str): 

    pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
    img = cv2.imread(str) # image in BGR format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig = plt.figure(figsize = [10,10])

    text1 = pytesseract.image_to_string(img)
    text = text1.split("\n")
    print(text1)
    return text

def extract_values(lst):
    singleval=[]
    rangeval=[]
    
    for i in lst:
        str=''
        for j in i:
            if j.isnumeric() or j=='-' or j==" " or j==".":
                str+=j
            else:
                str=''
                break
        
        if str!='':
            singleval.append(str)

    rangeval=singleval[12:]
    singleval=singleval[0:12]
                

    dict={"values":singleval,"range":rangeval}


    return dict

str="D:\hackathon\Silicon Rush\letsImage.jpg"
lst = convert_img_to_text(str)
lst1 = extract_values(lst)

print(lst1)