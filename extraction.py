
# imports
from tkinter import *


fields = ('Name', 'Patient Number', 'Aadhaar Number')

patient_details = {}

def capture_details(entries):
    patient_details['name'] = entries['Name'].get()
    patient_details['patient_no'] = entries['Patient Number']
    patient_details['aadhaar_no'] = entries['Aadhaar Number']
    
    frame.destroy()

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

if __name__ == '__main__':
    global frame
    root = Tk()
    root.title('Report Details')
    frame = Frame(root)
    frame.pack()    
    ents = makeform(frame, fields)
    capture_button = Button(frame, text = 'Capture Form',
    command=(lambda e = ents: capture_details(e)))
    capture_button.pack()
    root.mainloop()