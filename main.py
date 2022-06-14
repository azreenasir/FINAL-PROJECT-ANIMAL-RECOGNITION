from tkinter import *
import tkinter as tk
from tkinter import filedialog,Text,Label
import tkinter.font as tkFont
from PIL import ImageTk,Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pygame

# Music 
pygame.mixer.init()
pygame.mixer.music.load("Waves.mp3")
pygame.mixer.music.play()
pygame.mixer.music.set_volume(0.03)

# Create Model
new_model = tf.keras.models.load_model('datatrain/saved_model/my_model')
batch_size = 32
img_height = 300
img_width = 300
class_names = ['Cat', 'bear', 'dog', 'monkey', 'sheep']

# Title and so on
root = tk.Tk()
root.title('PROJECT ANIMAL RECOGNITION!')
root.geometry("1920x1080")
root.iconbitmap('icon.ico')

# Function

# Function for browse and display the image
def searchImage():
    global filename
    filename = filedialog.askopenfilename(initialdir="/dataset",title="Select Image",
    filetypes= (("JPG File","*.jpg*"),("PNG file","*.png"), ("All Files", "*.*")))
    img = Image.open(filename)
    img.thumbnail((400,350))
    img = ImageTk.PhotoImage(img)
    lblImg.configure(image=img)
    lblImg.image = img
    return filename

# Function to run the app to recognize the animal
def runApp():
    global filename
    img = keras.preprocessing.image.load_img(
    filename, target_size=(img_height, img_width))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = new_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    Output = (
    "This is the image of {} with a {:.2f} %"" confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score)))
    labeloutput = tk.Label(frame,text=Output, font="Arial 14 bold",bg="#957A52")
    labeloutput.place(relx=0.35, rely=0.8)

# End of All Function


# Create Frame
frame =tk.Frame(root,bg="#957A52")
frame.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)

# Title 
labelTitle = Label(frame, text="ANIMAL RECOGNITION SYSTEM", font="Arial 40 bold", background="#C4A484", borderwidth=3, relief=SOLID)
labelTitle.place(relx=0.222, rely=0.05)

# Label to display image that selected
lblImg = Label(root, bg="#957A52")
lblImg.place(relx=0.4, rely=0.25)

# Label Result 
lblResult = Label(frame, text="Result Prediction",font="Arial 20 bold",  background= "#C4A484", borderwidth=3, relief=SOLID)
lblResult.place(relx=0.42, rely=0.72)

# Logo
logo =ImageTk.PhotoImage(file= "logo.png")
labellogo = Label(frame ,image=logo, bd=0)
labellogo.place(relx=0.01, rely=0.01)

# Button for open file explorer
openFile = tk.Button(frame,text="Browse Image", font="Arial, 12",padx=20,pady=10,fg="white",bg="#263D42", command=searchImage)
openFile.place(relx= 0.4, rely=0.62)

# Run The recognition system
RunApp = tk.Button(frame,text="   Run App   ", font="Arial, 12",padx=20,pady=10,fg="white",bg="#263D42", command=runApp)
RunApp.place(relx=0.5, rely=0.62)

# Exit Button to close
Exitbutton = Button(frame, text = "Exit!",font="Arial, 12",padx=20,pady=10,fg="white",bg="#263D42", command = root.destroy)
Exitbutton.place(relx=0.9, rely=0.9)


root.mainloop()