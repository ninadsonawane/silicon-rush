import pytesseract      
  
# adds image processing capabilities
from PIL import Image    
  
 # converts the text to speech  
import pyttsx3           
  
#translates into the mentioned language
  
 # opening an image from the source path
img = Image.open('ab.jpeg')     
img.show()
# describes image format in the output
print(img)                          
# path where the tesseract module is installed
pytesseract.pytesseract.tesseract_cmd ='C:/Program Files/Tesseract-OCR/tesseract.exe' 
# converts the image to result and saves it into result variable
result = pytesseract.image_to_string(img)   
lines = result.split('\n')

print(lines)
lines = list(filter(None, lines))
print(lines)

list_len = int(len(lines)/4)

test_name = lines[:list_len]
results = lines[list_len:list_len*2]
units = lines[list_len*2:list_len*3]
reference = lines[list_len*3:list_len*4]

print(len(test_name), test_name)
print(len(results) ,results)
print(len(units), units)
print(len(reference) ,reference)

new_list = []
for i in range(list_len):
    new_list.append( test_name[i] + " " + results[i] + " " + units[i] + " " + reference[i])
print(new_list)

# write text in a text file and save it to source path   
with open('atee.txt',mode ='w') as file:     
      
                 file.write(result)
                #  print(result)
                   
