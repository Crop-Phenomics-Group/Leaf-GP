import sys
import os
import PIL.Image
import PIL.ImageTk
from autosize_treeview import AutoSizeTreeView

if sys.version_info[0] == 2:
    from Tkinter import *
else:
    from tkinter import *
    import tkinter.ttk

# if we are running on windows then import this library (used to help resolve resource directory)
if sys.platform == 'win32':
    from ctypes import create_unicode_buffer, windll

# returns the path of the resources (the root directory of the application)
def get_resource_directory():
    # if the application has been packaged
    if hasattr(sys, 'frozen'):
        # if the application is a single executable file
        if hasattr(sys, '_MEIPASS'):
            # if we are on windows OS
            if sys.platform == 'win32':
                # if the application is packaged using PyInstaller as a single executable file on windows then
                # when the application is run, a temp folder is created that contains all of the program dependencies.
                # On windows this will be a folder with semi-randomly assigned name and if the name
                # is longer than 8 characters, sys._MEIPASS will only return the short name of the directory.
                # We need the full name and use this code to retrieve it.
                # Create a buffer of size 500
                buffer_size = 500
                buffer = create_unicode_buffer(buffer_size)
                # put long name of the resource directory into the buffer
                windll.kernel32.GetLongPathNameW(unicode(sys._MEIPASS), buffer, buffer_size)
                # return the buffer value (full long name of the resource directory)
                return buffer.value
            # if we are not on windows OS, then return the MEIPASS variable (have not encountered any issues with this)
            return sys._MEIPASS
        # if the application has not been packages as a single file then return the folder directory that stores the exe
        return os.path.dirname(sys.executable)
    # otherwise, we are probably running source, so return the directory that this file is in
    return os.path.dirname(os.path.realpath(__file__))


# class that subclasses AutoSizeTreeView (which itself subclasses Tkinter TreeView)
# this class is used to create treeview (table) that includes a selectable checkbox for each row
# NOTE: This class was designed to work when treeview is added to and will need to be developed further to support other functionality such as remove, reorder, etc.
class CheckBoxTreeView(AutoSizeTreeView):

    # this stores the maximum number of characters that are permitted in a cell before elipses should be used
    MAX_COLUMN_CHARS = 20

    # constructor
    def __init__(self, master, **kw):
        # stores the master
        self.master = master
        # stores whether the table is read-only (used when wish to stop user interaction)
        self.locked = False
        # an array of booleans that represent the check status of each row (True = checked, False = unchecked)
        self.tick_list = []
        # get the directory when we can load the tick and untick images from (resource directory)
        the_dir = get_resource_directory()
        # open images for tick and untick files
        image_tick = PIL.Image.open(os.path.join(the_dir, "tick.png"))
        image_untick = PIL.Image.open(os.path.join(the_dir, "untick.png"))
        # convert the tick images into tk photoimages so that they can be inserted into the GUI
        self.pic_tick = PIL.ImageTk.PhotoImage(image_tick)
        self.pic_untick = PIL.ImageTk.PhotoImage(image_untick)
        # constructor for the parent class
        AutoSizeTreeView.__init__(self, master, **kw)
        # bind mouse button 1 release event so that we can capture clicks on rows to toggle tick state
        self.bind('<ButtonRelease-1>', self.select_item)


    # overrides the default insert behaviour
    def insert(self, parent, index, iid=None, **kw):
        # iterate through table values and shorten and append elipses if necessary
        for i in range(len(kw["values"])):
            # get the text in the cell
            v = str(kw["values"][i])
            # if the length of the text exceeds the limit
            if len(v) > CheckBoxTreeView.MAX_COLUMN_CHARS:
                # cut the text short and append elipse so text length equals limit
                v = v[:CheckBoxTreeView.MAX_COLUMN_CHARS-3] + "..."
                # save the new text to the treeview cell
                kw["values"][i] = v
        # perform default insert behaviour of parent class
        item = AutoSizeTreeView.insert(self, parent, index, iid=iid, **kw)
        # add the tick image to the new row
        self.item(item, image=self.pic_tick)
        # update the array of tick statuses
        # if we are adding row to the end of the treeview then just append True
        if index is END:
            self.tick_list.append(True)
        # otherwise add True to appropriate position in tick status array
        else:
            self.tick_list.insert(index, True)

    # this is the event callback for mouse button 1 release (when user click row)
    def select_item(self, event):
        try:
            # if the user did not click on a valid row or column then exit function
            if self.identify_column(event.x) == '' or self.identify_row(event.y) == '':
                return
            # only respond to user events if the table is unlocked
            if self.locked is False:
                # get the row that the user clicked on
                item = self.selection()[0]
                # get the column the user clicked on
                column = self.identify_column(event.x)
                # if this is the first column (where tick box is shown)
                if column == "#0":
                    # get the row index
                    item_index = self.index(item)
                    # if the box is already ticked
                    if self.tick_list[item_index]:
                        # show the unticked image
                        self.item(item, image=self.pic_untick)
                        # update the corresponding status flag
                        self.tick_list[item_index] = False
                    # otherwise the box is unticked
                    else:
                        # show the ticked image
                        self.item(item, image=self.pic_tick)
                        # update the corresponding status flag
                        self.tick_list[item_index] = True
        # we catch any exceptions (this can occur when there is no data in table, or user click treeview but not on row)
        except:
            # do nothing - just ignore the click
            pass

    # clear the treeview
    def clear(self):
        # iterate through the rows
        for i in self.get_children():
            # delete the row
            self.delete(i)
        # clear the list of tick statuses
        self.tick_list = []
