import sys
import cv2
import PIL.Image
import PIL.ImageTk

if sys.version_info[0] == 2:
    from Tkinter import *
    import tkSimpleDialog as dialog
else:
    from tkinter import *
    import tkinter.simpledialog as dialog


# class for previewing plots as separate dialog (subclass of tkinter dialog widget)
class PreviewDialog(dialog.Dialog):

    # height of canvas in dialog (in pixels)
    CANVAS_HEIGHT = 500

    # constructor
    # @param parent: the widget's parent
    # @param title: the title to display
    # @param img: the numpy matrix image to display in the dialog
    def __init__(self, parent, title, img):
        # placeholder to store master, canvas and photoimage references
        self.master = None
        self.canvas = None
        self.photo = None
        # store the image
        self.img = img
        # constructor for parent class
        dialog.Dialog.__init__(self, parent, title)

    # this function is used to create the dialog
    def body(self, master):
        # set the master
        self.master = master
        # do not allow the dialog to be resized
        self.resizable(0, 0)
        # create the canvas widget
        self.canvas = Canvas(self.master, bg="gray", width=PreviewDialog.CANVAS_HEIGHT,
                             height=PreviewDialog.CANVAS_HEIGHT)
        # set canvas visual properties
        self.canvas.config(highlightthickness=1, bg="black", bd=0, highlightbackground="black")
        # make the canvas fill its parent
        self.canvas.grid(row=0, column=0, sticky=N+E+S+W)
        # get the height and width of the image to display
        h, w = self.img.shape[:2]
        # compute the aspect ratio
        ratio = w / float(h)
        # determine the resize width of the image based the canvas height
        new_width = int(round(ratio * PreviewDialog.CANVAS_HEIGHT))
        # resize the image so that it is the same height as the canvas
        img = cv2.resize(self.img, (new_width, PreviewDialog.CANVAS_HEIGHT), interpolation=cv2.INTER_CUBIC)
        # create image from numpy matrix
        pil_img = PIL.Image.fromarray(img)
        # store a reference the photoImage (required for canvas)
        self.photo = PIL.ImageTk.PhotoImage(image=pil_img)
        # put the image on the canvas
        img_obj = self.canvas.create_image(0, 0, image=self.photo, anchor=NW)
        self.canvas.itemconfig(img_obj, image=self.photo)
        # set the canvas width
        self.canvas.config(width=new_width)
        return None

    # override this method to remove okay and cancel buttons
    def buttonbox(self):
        pass
