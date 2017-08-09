import sys
import os
from threading import Lock
from gui_queue import GuiQueue
if sys.version_info[0] == 2:
    from Tkinter import *
    import tkMessageBox as messagebox
else:
    from tkinter import *



# returns the directory of the executable (used when creating log files)
def get_executable_directory():
    # if the application has been packaged
    if hasattr(sys, 'frozen'):
        # return the folder directory that stores the exe
        return os.path.dirname(sys.executable)
    # otherwise, we are probably running source, so return the directory that this file is in
    return os.path.dirname(os.path.realpath(__file__))


# class that represents a log widget (subclass of the Tkinter Frame)
class LogWidget(Frame):

    # stores the width of the log (based on characters)
    LOG_WIDTH = 25

    # constructor
    def __init__(self, parent, title, log_file, *args, **kwargs):
        # parent constructor
        Frame.__init__(self, parent, *args, **kwargs)
        # boolean flag used to determine whether this log widget is actively assigned to a processing task.
        self.is_assigned = False
        # stores the message line number (used when outputting message to the log)
        self.message_number = 0
        # create a label in the GUI showing the specified title
        log_label = Label(self, text=title)
        # put the label at the top of the frame
        log_label.grid(row=0, column=0)
        # create a textbox for the messages
        self.log = Text(self, width=LogWidget.LOG_WIDTH, wrap=NONE, borderwidth=1, highlightthickness=1,
                        highlightbackground="gray", highlightcolor="gray", font=("Courier", -10))
        # put the text box underneath the title
        self.log.grid(row=1, column=0, sticky=N+E+S+W)
        # make the textbox stretch vertically to fill its bounds
        self.rowconfigure(1, weight=1)
        # disable the textbox (this makes it read only - we don't want users to be able to enter text)
        self.log.config(state=DISABLED)
        # add vertical scrollbar to the text box
        log_scroll_vertical = Scrollbar(self, command=self.log.yview, orient=VERTICAL)
        # put the vertical scrollbar to the right of the text box
        log_scroll_vertical.grid(row=1, column=1, sticky=N+E+S)
        # link the scrollbar with the log
        self.log.configure(yscrollcommand=log_scroll_vertical.set)
        # add horizontal scrollbar
        log_scroll_horizontal = Scrollbar(self, command=self.log.xview, orient=HORIZONTAL)
        # put the horizontal scrollbar underneath the text box
        log_scroll_horizontal.grid(row=2, column=0, sticky=E+W)
        # link the scrollbar with the log
        self.log.configure(xscrollcommand=log_scroll_horizontal.set)
        # set the filename of the output file log
        self.log_file = log_file

    # print the specified message to the log
    # @param msg: the message to write to the log
    # @param *args: strings to be appended to the log (behaviour as in print function)
    # NOTE: as this function has calls to tkinter GUI, it must be called from main thread
    def print_to_log(self, msg, *args):
        # create the log message (message number followed by tab then the message)
        log_message = str(self.message_number + 1) + ":\t" + str(msg) # start from message number one
        # append arguments to the message string (separated by spaces)
        # iterate through optional arguments
        for arg in args:
            # append to log message
            log_message += " " + str(arg)
        # temporarily enable the log so that we can write to it
        self.log["state"] = NORMAL
        # add the log message to the end of the log. Also append new line
        self.log.insert(END, log_message + "\n")
        # disable the log so that it is read only again
        self.log["state"] = DISABLED
        # scroll to the bottom of the log so that the most recent message is visible
        self.log.see(END)
        # increment the message number for next message
        self.message_number += 1

    # print the specified message to the the log file
    # @param msg: the message to write to the log
    # @param *args: strings to be appended to the log (behaviour as in print function)
    def print_to_file_log(self, msg, *args):
        # create the log message (cast to string)
        log_message = str(msg)
        # append arguments to the message string (separated by spaces)
        # iterate through optional arguments
        for arg in args:
            # append to log message
            log_message += " " + str(arg)
        # open the log file
        with open(self.log_file, "a") as log_file:
            # write the message to file. Also append new line
            log_file.write(log_message + "\n")

    # clears the text in the GUI log
    # NOTE: as this function has calls to tkinter GUI, it must be called from main thread
    def clear_text(self):
        # reset the message number to 0
        self.message_number = 0
        # temporarily enable the log so that we can write to it
        self.log["state"] = NORMAL
        # delete all content
        self.log.delete('1.0', END)
        # disable the log so that it is read only again
        self.log["state"] = DISABLED

    # sets the assigned flag to false. This allows the log manager to assign the log to another process
    def detach(self):
        # sets the flag to false
        self.is_assigned = False


# class representing log manager (subclass of the Tkinter Frame)
# this class will create 3 logs stacked vertically in the GUI
# NOTE: this class can only be used in tkinter grid layout
class LogManagerWidget(Frame):

    # lock mechanism to prevent multiple threads of processes from executing code simultaneously
    lock = Lock()

    # constructor
    # @param number_of_logs: the number of logs to stack vertically
    def __init__(self, parent, number_of_logs, *args, **kwargs):
        # parent constructor
        Frame.__init__(self, parent, *args, **kwargs)
        # store the number of logs
        self.number_of_logs = number_of_logs
        # create a map to store tkinter layout options
        self.grid_information = dict()
        # boolean flag that stores whether the logs are visible or hidden (initially hidden)
        self.show = False
        # store an array of logs
        self.logs = [None] * self.number_of_logs
        # set the layout properties of the frame
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        # create a new frame within the log manager frame
        self.frame = Frame(self)
        self.frame.grid(row=0, column=0, sticky=N+E+S+W)
        # get the directory that the executable/file is in (path then used to save file log to)
        the_dir = get_executable_directory()
        # iterate through the log widgets
        for i in range(self.number_of_logs):
            # build filename for corresponding file log
            log_filename = os.path.join(the_dir, "leaf-GP_log_" + str(i+1) + ".txt")
            # creates a log widget
            log_widget = LogWidget(self, "Processing Log (Task " + str(i+1) + ")", log_filename)
            # position the log vertically based on the log number
            log_widget.grid(row=i, column=0)
            # the log can stretch with its parent container vertically
            self.rowconfigure(i, weight=1)
            # store a reference to the log widget
            self.logs[i] = log_widget

    # toggle show/hide for the log manager
    # NOTE: as this function has calls to tkinter GUI, it must be called from main thread
    def toggle_show_hide(self):
        # if the log is currently visible
        if self.show:
            # store the tkinter grid layout information for the log manager in the information map/dictionary
            # this is important as we will use this information when we re-show the log manager
            self.grid_information = self.grid_info()
            # remove the log from the GUI
            self.grid_forget()
        # if the log is currently hidden
        else:
            # add the manager log to the GUI
            self.grid(**self.grid_information)

    # overrides grid method for frame
    # NOTE: as this function has calls to tkinter GUI, it must be called from main thread
    def grid(self, **options):
        # ensure that the show flag has been set to true if this widget has been added to the GUI
        self.show = True
        # execute default behaviour
        Frame.grid(self, **options)

    # overrides grid method for frame
    # NOTE: as this function has calls to tkinter GUI, it must be called from main thread
    def grid_forget(self):
        # ensure that the show flag has been set to false if this widget has been removed
        self.show = False
        # execute default behaviour
        Frame.grid_forget(self)

    # search through the list of logs to find an unassigned log - this is used when assigning a log to a thread/process.
    # returns none if there are no unassigned logs
    def get_first_unassigned_log(self):
        # use the lock mechanism to ensure that only one thread is checking and assigning logs at any time
        # without the lock, there could potentially be an issue if 2+ threads find the same unassigned log
        # and both attempt to assign itself to it
        with LogManagerWidget.lock:
            # iterate through the list of logs
            for i in range(len(self.logs)):
                # if the log is unassigned
                if not self.logs[i].is_assigned:
                    # set the assigned flag for this log to true
                    self.logs[i].is_assigned = True
                    # clear the text (this is GUI task, so add to to GUI queue running in main thread)
                    GuiQueue.gui_queue.put(self.logs[i].clear_text)
                    # return the unassigned log (for assigning)
                    return self.logs[i]
        # if we reached the end of the list of logs and couldn't find an unassigned log then return none.
        return None

    # get a list of existing log files
    def get_existing_log_files(self):
        # list of found log files
        existing_log_file_list = []
        # iterate through the logs
        for log in self.logs:
            # does the log already exist?
            if os.path.isfile(log.log_file):
                # add to the list of found log files
                existing_log_file_list.append(log.log_file)
        # return the list
        return existing_log_file_list

    # clear all the logs
    # NOTE: as this function has calls to tkinter GUI, it must be called from main thread
    def clear_all(self):
        # iterate through the logs
        for log in self.logs:
            # clear the log
            log.clear_text()
