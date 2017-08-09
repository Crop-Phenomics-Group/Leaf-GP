import sys
if sys.version_info[0] == 2:
    from Tkinter import *
    import ttk
    import tkFileDialog as filedialog
    import tkMessageBox as messagebox
else:
    from tkinter import *
    import tkinter.ttk
    import tkinter.filedialog as filedialog
    import tkinter.messagebox as messagebox

from functools import partial
from gui_queue import GuiQueue
import Queue
import matplotlib
matplotlib.use('Agg')
from analysis import Analysis
from series import Series
import PIL.Image
import PIL.ImageTk
from preview import PreviewDialog
import plot_generator
from trait_ids import Trait
import glob
import os
from concurrent.futures import ThreadPoolExecutor
from checkbox_treeview import CheckBoxTreeView
from datetime import datetime
import subprocess
from enum import Enum

import logger_widget
from threading import Lock
from autosize_treeview import AutoSizeTreeView
import cv2
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


# enumeration class to represent application status
class ProcessingState(Enum):
    # state when user is entering input data
    DATA_INPUT = 0
    # state when analysis is running
    RUNNING_ANALYSIS = 1
    # state when analysis has ended
    ANALYSIS_ENDED = 2

# this class is the main application
class Application:
    # title to display in GUI
    WINDOW_TITLE = "Leaf-GP"
    # initial application window dimensions (width, height)
    INITIAL_WINDOW_DIMENSIONS = "1024x768"
    # GUI section heading font
    SECTION_HEADING_FONT = ("Helvetica", -16)
    # list of supported image file types
    SUPPORTED_FILE_TYPES = [".jpg", ".JPG", ".jpeg", ".png", ".PNG"]
    # GUI tkinter outer frame padding, border width, and border colour
    FRAME_PADDING = 10
    FRAME_BORDER_WIDTH = 1
    FRAME_BORDER_COLOUR = "dark gray"
    # GUI tkinter inner frame padding
    INNER_FRAME_PADDING_X = 10
    INNER_FRAME_PADDING_Y = 5
    # GUI section headings
    DATA_INPUT_TEXT                 = "1. DATA INPUT:"
    MACHINE_LEARNING_SETTINGS_TEXT  = "2. COLOUR CLUSTERING SETTING:"
    SERIES_PROCESSING_TEXT          = "3. SERIES PROCESSING:"
    RESULTS_TEXT                    = "4. RESULTS:"
    # input value options (plant species)
    PLANT_SPECIES_ARABIDOPSIS = "Arabidopsis"
    PLANT_SPECIES_WHEAT = "Wheat"
    # input value options (experiment data/meta source)
    EXPERIMENT_DATA_FROM_FOLDER_TEXT = "From Folder Name"
    EXPERIMENT_DATA_FROM_FILE_TEXT = "From Image Name"
    NO_EXPERIMENT_DATA_TEXT = "No Experiment Data Available"
    # default input values
    DEFAULT_IN_ROWS = 4
    DEFAULT_IN_COLS = 6
    DEFAULT_IN_REF_RADIUS = 4
    DEFAULT_IN_PLANT_SPECIES = PLANT_SPECIES_ARABIDOPSIS
    DEFAULT_IN_EXPERIMENTAL_DATA_SOURCE = EXPERIMENT_DATA_FROM_FILE_TEXT
    DEFAULT_IN_PIXEL_GROUPS = 4
    # width of machine learning setting section canvas (show k-means images)
    MACHINE_LEARNING_IMAGE_CANVAS_WIDTH = 120
    # set min and max machine learning pixel groups values
    MACHINE_LEARNING_PIXEL_GROUPS_MIN = 3
    MACHINE_LEARNING_PIXEL_GROUPS_MAX = 10
    # text to show above the machine learning sample image canvas
    MACHINE_LEARNING_SAMPLE_IMAGE_TEXT = "Sample Image"
    # text to show above the machine learning pixel clustering canvas
    MACHINE_LEARNING_PIXEL_CLUSTERING_TEXT = "Pixel Clustering"
    # text to show for pixel groups input label
    MACHINE_LEARNING_PIXEL_GROUPS_TEXT = "Pixel Groups:"
    # treeview/table column index of status field (required when updating processing status for an image series)
    STATUS_COL_INDEX = 1
    # status text to display for an image series before it has been processed
    NOT_PROCESSED_TEXT = "Not Processed"
    # text to display if value in table is unidentified (not sure if this is used anymore)
    UNIDENTIFIED_TEXT = "Unidentified"
    # text to display when an image series analysis has been cancelled
    CANCELLED_STATUS_TEXT = "Cancelled"
    # text to display in plot preview dialog title bar
    PLOT_PREVIEW_DIALOG_TITLE_TEXT = "Plot Viewer"
    # analysis button (run analysis) text
    ANALYSIS_BUTTON_TEXT_RUN = "Run Analysis"
    # number of parallel processes (same as number of logs)
    THREAD_POOL_SIZE = 3

    # constructor
    def __init__(self, root):
        # store a reference to the root of the program
        self.root = root
        # a hash map/dictionary that stores all of the data series information (key is unique based on experimental reference and tray number)
        self.data_series = dict()
        # list of 'future' processes used when running tasks in parallel
        self.futures = []
        # set font style for treeviews (tables)
        style = ttk.Style()
        style.configure('Treeview', font=AutoSizeTreeView.TREEVIEW_FONT)
        # value to store the processing state of the application (initial state is data input)
        self.processing_state = ProcessingState.DATA_INPUT
        # input directory text linked to entry box
        self.in_directory_text = StringVar()
        # input number of rows and columns text linked to respective entry boxes (stored as ints)
        self.in_rows_text = IntVar()
        self.in_rows_text.set(Application.DEFAULT_IN_ROWS)
        self.in_cols_text = IntVar()
        self.in_cols_text.set(Application.DEFAULT_IN_COLS)
        # reference marker radius
        self.in_ref_radius = IntVar()
        self.in_ref_radius.set(Application.DEFAULT_IN_REF_RADIUS)

        # set-up the main frame (split into 3 sections - 3 rows x 1 column)
        self.frame = Frame(root)
        # make the frame stick to all sides
        self.frame.grid(row=0, column=0, sticky=N+S+E+W)
        # there is only 1 column so make this expand with the parent
        self.frame.columnconfigure(0, weight=1)
        # the second and third rows will stretch with the window as these two frame will have tables (treeviews)
        self.frame.rowconfigure(1, weight=1)
        self.frame.rowconfigure(2, weight=1)

        # set-up the log frame (split into 3 sections - 3 rows x 1 column)
        self.log_frame = logger_widget.LogManagerWidget(root, Application.THREAD_POOL_SIZE, padx=Application.FRAME_PADDING, pady=Application.FRAME_PADDING, highlightthickness=Application.FRAME_BORDER_WIDTH)
        # make the frame stick to all sides
        self.log_frame.grid(row=0, column=1, sticky=N+S+E+W)
        # setup frame visual properties
        self.log_frame.config(highlightbackground=Application.FRAME_BORDER_COLOUR)
        self.log_frame.config(highlightcolor=Application.FRAME_BORDER_COLOUR)
        # the logs will be visible after creating them and we want them to be hidden when the application starts, so toggle their visibility
        self.log_frame.toggle_show_hide()
        # set-up data input frame
        self.data_input_frame = Frame(self.frame, padx=Application.FRAME_PADDING, pady=Application.FRAME_PADDING, highlightthickness=Application.FRAME_BORDER_WIDTH)
        # set-up frame appearance
        self.data_input_frame.config(highlightbackground=Application.FRAME_BORDER_COLOUR)
        self.data_input_frame.config(highlightcolor=Application.FRAME_BORDER_COLOUR)
        # the frame should stick to all sides of the parent
        self.data_input_frame.grid(row=0, sticky=N+E+S+W)
        # there is only 1 column so make this expand with the parent
        self.data_input_frame.columnconfigure(0, weight=1)
        # create the section heading
        self.data_input_label = Label(self.data_input_frame, font=Application.SECTION_HEADING_FONT, text=Application.DATA_INPUT_TEXT)
        # make this stick to the left of the widget
        self.data_input_label.grid(row=0, column=0, sticky=W)

        # an inner frame is used to position all of the widgets (offset so that it appears to be under parent section heading)
        self.data_input_inner_frame = Frame(self.data_input_frame, padx=Application.INNER_FRAME_PADDING_X, pady=Application.INNER_FRAME_PADDING_Y)
        # the frame should stick to all sides of the parent
        self.data_input_inner_frame.grid(row=1, column=0, sticky=N+E+S+W)
        # there are several columns. We wish to expand on the column that includes the input directory box (long string)
        self.data_input_inner_frame.columnconfigure(6, weight=1)

        # set-up machine learning settings frame
        self.machine_learning_frame = Frame(self.frame, padx=Application.FRAME_PADDING, pady=Application.FRAME_PADDING, highlightthickness=Application.FRAME_BORDER_WIDTH)
        # set-up frame appearance
        self.machine_learning_frame.config(highlightbackground=Application.FRAME_BORDER_COLOUR)
        self.machine_learning_frame.config(highlightcolor=Application.FRAME_BORDER_COLOUR)
        # the frame should stick to all sides of the parent
        self.machine_learning_frame.grid(row=0, column=1, sticky=N+E+S+W)
        # there is only 1 column so make this expand with the parent
        self.machine_learning_frame.columnconfigure(0, weight=1)
        # create the section heading
        self.machine_learning_label = Label(self.machine_learning_frame, text=Application.MACHINE_LEARNING_SETTINGS_TEXT, font=Application.SECTION_HEADING_FONT)
        # make this stick to the left of the widget
        self.machine_learning_label.grid(row=0, column=0, sticky=W)

        # an inner frame is used to position all of the widgets (offset so that it appears to be under parent section heading)
        self.machine_learning_inner_frame = Frame(self.machine_learning_frame, padx=Application.INNER_FRAME_PADDING_X, pady=Application.INNER_FRAME_PADDING_Y)
        # the frame should stick to all sides of the parent
        self.machine_learning_inner_frame.grid(sticky=N+W+E+S)
        # Expand the first row and column
        self.machine_learning_inner_frame.columnconfigure(0, weight=1)
        self.machine_learning_inner_frame.rowconfigure(0, weight=1)

        # set-up the 2 tkinter canvas objects to show 2 preview images in machine learning settings section
        # set-up the sample image label
        self.sample_image_canvas_label = Label(self.machine_learning_inner_frame, text=Application.MACHINE_LEARNING_SAMPLE_IMAGE_TEXT, justify=CENTER)
        # this label will go above the canvas (but will span 2 columns so that pixel groups input label and information icon will appear within the bounds of the canvas)
        self.sample_image_canvas_label.grid(row=0, column=0, columnspan=2, sticky=W+E)
        # set-up the sample image canvas (again this will also span 2 columns)
        self.sample_image_canvas = Canvas(self.machine_learning_inner_frame, bg="gray", width=Application.MACHINE_LEARNING_IMAGE_CANVAS_WIDTH, height=Application.MACHINE_LEARNING_IMAGE_CANVAS_WIDTH)
        # set-up canvas visual properties
        self.sample_image_canvas.config(highlightthickness=1, bg="black", bd=0, highlightbackground="black")
        # this will be positioned in 2nd row as 1st row is reserved to display label above
        self.sample_image_canvas.grid(row=1, column=0, columnspan=2, padx=1, pady=1, sticky=N+E+S+W)
        # set-up the pixel clustering image label
        self.pixel_clustering_canvas_label = Label(self.machine_learning_inner_frame, text=Application.MACHINE_LEARNING_PIXEL_CLUSTERING_TEXT, justify=CENTER)
        # this label will go above the canvas
        self.pixel_clustering_canvas_label.grid(row=0, column=2, sticky=W+E)
        # set-up the pixel clustering image canvas
        self.pixel_clustering_canvas = Canvas(self.machine_learning_inner_frame, bg="gray", width=Application.MACHINE_LEARNING_IMAGE_CANVAS_WIDTH, height=Application.MACHINE_LEARNING_IMAGE_CANVAS_WIDTH)
        # set-up canvas visual properties
        self.pixel_clustering_canvas.config(highlightthickness=1, bg="black", bd=0, highlightbackground="black")
        # this will be positioned in 2nd row as 1st row is reserved to display label above
        self.pixel_clustering_canvas.grid(row=1, column=2, padx=1, pady=1, sticky=N+E+S+W)

        # set the default 'no-image available' image to be shown in both canvases
        # in the machine learning settings section, we show image preview boxes
        # when there is no preview to show, we show a placeholder (no image available) image
        # get the filename of the no image available image resource
        no_image_available_img_filename = os.path.join(get_resource_directory(), "no_image.png")
        # read the image and store it
        self.no_image_available_img = cv2.imread(no_image_available_img_filename)
        self.set_machine_learning_sample_image(self.no_image_available_img, Application.MACHINE_LEARNING_IMAGE_CANVAS_WIDTH)
        self.set_machine_learning_pixel_clustering_image(self.no_image_available_img, Application.MACHINE_LEARNING_IMAGE_CANVAS_WIDTH)

        # create validation command callback (ensure that +ive integers are only allowed to be entered)
        positive_int_validate_command = self.frame.register(self.validate_positive_int)

        # set-up pixel groups label
        self.pixel_groups_label = Label(self.machine_learning_inner_frame, text=Application.MACHINE_LEARNING_PIXEL_GROUPS_TEXT)
        # position in the 3rd row (underneath the canvas img)
        self.pixel_groups_label.grid(row=2, column=0, sticky=W)
        # integer variable to store and link to pixel groups entry box
        self.pixel_groups_value = IntVar()
        # set its value to the default value
        self.pixel_groups_value.set(Application.DEFAULT_IN_PIXEL_GROUPS)
        # set-up the GUI entry text box for pixel groups
        self.pixel_groups_entry_box = Entry(self.machine_learning_inner_frame, textvariable=self.pixel_groups_value, width=3, validate="key", validatecommand=(positive_int_validate_command, '%P'))
        # position in the 3rd row but in the 3rd column so that it is underneath the pixel groups canvas image
        self.pixel_groups_entry_box.grid(row=2, column=2, sticky=W)
        # get the filename of the information icon image (active and disabled versions)
        info_icon_filename = os.path.join(get_resource_directory(), "information_icon.png")
        info_icon_disabled_filename = os.path.join(get_resource_directory(), "information_icon_disabled.png")
        # load these images
        self.info_icon = PIL.ImageTk.PhotoImage(file=info_icon_filename)
        self.info_icon_disabled = PIL.ImageTk.PhotoImage(file=info_icon_disabled_filename)
        # to show information icons, we use labels with images that can be clicked on to bring up pop-up with more info.
        self.pixel_groups_info_icon_label = Label(self.machine_learning_inner_frame, image=self.info_icon)
        # set the image for the label (by default it is the active/coloured version of the image)
        self.pixel_groups_info_icon_label.photo = self.info_icon
        # position on the right-hand side of the sample canvas image (therefore column 1)
        self.pixel_groups_info_icon_label.grid(row=2, column=1, sticky=W)
        # attach the mouse click release event to a function that displays the relevant information in pop-up
        self.pixel_groups_info_icon_label.bind('<ButtonRelease-1>', self.show_machine_learning_info)

        # create and position labels for the input boxes in data input section
        # input directory
        self.input_directory_entry_label = Label(self.data_input_inner_frame, text="Image Dir.:")
        self.input_directory_entry_label.grid(row=0, column=0, sticky=W)
        # number of rows
        self.input_rows_entry_label = Label(self.data_input_inner_frame, text="Rows No.:")
        self.input_rows_entry_label.grid(row=1, column=0, sticky=W)
        # number of columns
        self.input_columns_entry_label = Label(self.data_input_inner_frame, text="Columns No.:")
        self.input_columns_entry_label.grid(row=1, column=2, sticky=E)
        # reference radius
        self.input_ref_radius_entry_label = Label(self.data_input_inner_frame, text="Ref. Radius (mm):")
        self.input_ref_radius_entry_label.grid(row=1, column=4, sticky=E)
        # plant species
        self.plant_species_dropdown_label = Label(self.data_input_inner_frame, text="Plant Species:")
        self.plant_species_dropdown_label.grid(row=4, sticky=W)
        # experimental data/meta source
        self.experimental_data_dropdown_label = Label(self.data_input_inner_frame, text="Read Exp. Data:")
        self.experimental_data_dropdown_label.grid(row=5, column=0, sticky=W)
        # to show information icons, we use labels with images that can be clicked on to bring up pop-up with more info.
        self.experimental_data_info_icon_label = Label(self.data_input_inner_frame, image=self.info_icon)
        # set the image for the label (by default it is the active/coloured version of the image)
        self.experimental_data_info_icon_label.photo = self.info_icon
        # set its position
        self.experimental_data_info_icon_label.grid(row=6, column=6, sticky=W)
        # attach the mouse click release event to a function that displays the relevant information in pop-up
        self.experimental_data_info_icon_label.bind('<ButtonRelease-1>', self.show_experimental_data_info)

        # create and position the input widgets
        # input directory entry box
        self.input_directory_entry_box = Entry(self.data_input_inner_frame, textvariable=self.in_directory_text)
        self.input_directory_entry_box.grid(row=0, column=1, columnspan=6, sticky=E + W)
        # input rows entry box
        self.input_rows_entry_box = Entry(self.data_input_inner_frame, width=2, textvariable=self.in_rows_text, validate="key", validatecommand=(positive_int_validate_command, '%P'))
        self.input_rows_entry_box.grid(row=1, column=1, sticky=W)
        # input columns entry box
        self.input_columns_entry_box = Entry(self.data_input_inner_frame, width=2, textvariable=self.in_cols_text, validate="key", validatecommand=(positive_int_validate_command, '%P'))
        self.input_columns_entry_box.grid(row=1, column=3, sticky=W)
        # input reference entry box
        self.input_ref_radius_entry_box = Entry(self.data_input_inner_frame, width=2, textvariable=self.in_ref_radius, validate="key", validatecommand=(positive_int_validate_command, '%P'))
        self.input_ref_radius_entry_box.grid(row=1, column=5, sticky=W)
        # create and position button to show the directory input dialog
        self.show_input_directory_dialog_button = Button(self.data_input_inner_frame, text="...", command=self.show_directory_dialog)
        self.show_input_directory_dialog_button.grid(row=0, column=7)
        # create and position plant species dropdown box
        self.plant_species_value = StringVar()
        self.plant_species_value.set(Application.DEFAULT_IN_PLANT_SPECIES)  # default value
        self.plant_species_selector = OptionMenu(self.data_input_inner_frame, self.plant_species_value, Application.PLANT_SPECIES_ARABIDOPSIS, Application.PLANT_SPECIES_WHEAT)
        self.plant_species_selector.grid(row=4, column=1, columnspan=3, sticky=W)
        # create and position experimental data source dropdown box
        self.experimental_data_value = StringVar()
        self.experimental_data_selector = OptionMenu(self.data_input_inner_frame, self.experimental_data_value, Application.EXPERIMENT_DATA_FROM_FOLDER_TEXT, Application.EXPERIMENT_DATA_FROM_FILE_TEXT, Application.NO_EXPERIMENT_DATA_TEXT)
        self.experimental_data_selector.config(width=25)
        self.experimental_data_selector.grid(row=5, column=1, columnspan=4, sticky=W)
        self.experimental_data_value.set(Application.DEFAULT_IN_EXPERIMENTAL_DATA_SOURCE)
        # create and position the name convention label
        self.name_convention_label = Label(self.data_input_inner_frame, text="Image Naming Convention: 'YYYY-MM-DD_Exp-ID_Tray-No.jpg'")
        self.name_convention_label.grid(row=6, column=0, columnspan=6, sticky=W)
        # create and position the load button
        self.load_button = Button(self.data_input_inner_frame, width=15, text="Load", command=self.load)
        self.load_button.grid(row=7, column=0, columnspan=8, sticky=E + S)

        # set-up series processing frame
        self.series_processing_frame = Frame(self.frame, padx=Application.FRAME_PADDING, pady=Application.FRAME_PADDING, highlightthickness=Application.FRAME_BORDER_WIDTH)
        # set-up frame appearance
        self.series_processing_frame.config(highlightbackground=Application.FRAME_BORDER_COLOUR)
        self.series_processing_frame.config(highlightcolor=Application.FRAME_BORDER_COLOUR)
        # the frame should stick to all sides of the parent (spans 2 columns to sit under both data input and machine learning sections)
        self.series_processing_frame.grid(row=1, column=0, columnspan=2, sticky=N+E+S+W)
        # there is only 1 column and 1 row so make them expand with the parent
        self.series_processing_frame.columnconfigure(1, weight=1)
        self.series_processing_frame.rowconfigure(1, weight=1)
        # create the section heading
        self.series_process_label = Label(self.series_processing_frame, font=Application.SECTION_HEADING_FONT, text=Application.SERIES_PROCESSING_TEXT)
        # make this stick to the left of the widget
        self.series_process_label.grid(row=0, sticky=N+W)

        # to show information icons, we use labels with images that can be clicked on to bring up pop-up with more info.
        self.series_process_info_icon_label = Label(self.series_processing_frame, image=self.info_icon)
        # set the image for the label (by default it is the active/coloured version of the image)
        self.series_process_info_icon_label.photo = self.info_icon
        # set its position
        self.series_process_info_icon_label.grid(row=0, column=1, sticky=W)
        # attach the mouse click release event to a function that displays the relevant information in pop-up
        self.series_process_info_icon_label.bind('<ButtonRelease-1>', self.show_series_processing_info)

        # an inner frame is used to position all of the widgets (offset so that it appears to be under parent section heading)
        self.series_processing_inner_frame = Frame(self.series_processing_frame, padx=Application.INNER_FRAME_PADDING_X, pady=Application.INNER_FRAME_PADDING_Y)
        # the frame should stick to all sides of the parent
        self.series_processing_inner_frame.grid(row=1, column=0, columnspan=2, sticky=N+E+S+W)

        # create horizontal and vertical scrollbars for image series table in series processing section
        series_processing_table_v_scroll = Scrollbar(self.series_processing_inner_frame, orient=VERTICAL)
        series_processing_table_h_scroll = Scrollbar(self.series_processing_inner_frame, orient=HORIZONTAL)
        # create and position treeview to represent table of loaded image/experiment series
        self.series_processing_table = CheckBoxTreeView(self.series_processing_inner_frame, yscrollcommand = series_processing_table_v_scroll.set, xscrollcommand = series_processing_table_h_scroll.set)
        # set-up the table
        self.series_processing_table["columns"] = ("id", "status", "exp_id", "tray", "days", "images")
        self.series_processing_table.heading("id", text="ID")
        self.series_processing_table.heading("status", text="Status")
        self.series_processing_table.heading("exp_id", text="Exp. Ref.")
        self.series_processing_table.heading("tray", text="Tray No.")
        self.series_processing_table.heading("days", text="Duration (Days)")
        self.series_processing_table.heading("images", text="No. Images")
        self.series_processing_table.column("#0", stretch=False, width=35)
        self.series_processing_table.column("exp_id", stretch=True)
        self.series_processing_table.column("tray", stretch=True)
        self.series_processing_table.column("days", stretch=True)
        self.series_processing_table.column("images", stretch=True)
        # position the table
        self.series_processing_table.grid(row=0, column=0, columnspan=3, sticky=N + E + S + W)
        # now the table has been created resize the columns to fit the content
        self.series_processing_table.auto_size()
        # link the scrollbars to the table
        series_processing_table_v_scroll.configure(command=self.series_processing_table.yview)
        series_processing_table_h_scroll.configure(command=self.series_processing_table.xview)
        # position the scrollbars
        series_processing_table_v_scroll.grid(row=0,column=3, sticky=N+S+E)
        series_processing_table_h_scroll.grid(row=1, column=0, columnspan=3, sticky=N + S + E + W)

        # set-up the run analysis button
        self.analysis_button = Button(self.series_processing_inner_frame, width=20, text=Application.ANALYSIS_BUTTON_TEXT_RUN, command=self.analysis_button_callback)
        self.analysis_button.grid(row=2, column=2, sticky=N + E + S)
        # set-up the show/hide processing log button
        self.toggle_processing_log_visibility_button = Button(self.series_processing_inner_frame, text="Show/Hide Processing Log", command=self.log_frame.toggle_show_hide)
        self.toggle_processing_log_visibility_button.grid(row=2, column=1, sticky=N + E + S + W)

        # make the series processing widget stretch on first row and column (table)
        self.series_processing_inner_frame.columnconfigure(0, weight=1)
        self.series_processing_inner_frame.rowconfigure(0, weight=1)

        # set-up results frame
        self.results_frame = Frame(self.frame, padx=10, pady=10, highlightthickness=1)
        # set-up frame appearance
        self.results_frame.config(highlightbackground="dark gray")
        self.results_frame.config(highlightcolor="dark gray")
        self.results_frame.grid(row=2, column=0, columnspan=2, sticky=N+E+S+W)
        # stretch the first column and the 2nd row (inner frame - don't stretch 1st row as this is the section heading)
        self.results_frame.columnconfigure(0, weight=1)
        self.results_frame.rowconfigure(1, weight=1)

        # set-up section header
        self.results_label = Label(self.results_frame, font=Application.SECTION_HEADING_FONT, text=Application.RESULTS_TEXT)
        # position the section header in the first row
        self.results_label.grid(row=0, sticky=N + W)
        # an inner frame is used to position all of the widgets (offset so that it appears to be under parent section heading)
        self.results_inner_frame = Frame(self.results_frame, padx=Application.INNER_FRAME_PADDING_X, pady=Application.INNER_FRAME_PADDING_Y)
        # inner frame should stick to parent frame on all sides
        self.results_inner_frame.grid(sticky=N + W + E + S)
        # inner frame should stretch on first row and column (results table)
        self.results_inner_frame.columnconfigure(0, weight=1)
        self.results_inner_frame.rowconfigure(0, weight=1)

        # create horizontal and vertical scrollbars for results table in the results section
        results_table_v_scroll = Scrollbar(self.results_inner_frame, orient=VERTICAL)
        results_table_h_scroll = Scrollbar(self.results_inner_frame, orient=HORIZONTAL)
        # create and position treeview to represent table of processed image/experiment series results
        self.results_table = AutoSizeTreeView(self.results_inner_frame, xscrollcommand = results_table_h_scroll.set, yscrollcommand = results_table_v_scroll.set)
        # set-up the table (IMPORTANT: column ids must be the same as column names in CSV file in order to show plots)
        self.results_table['show'] = 'headings' # hides the 1st column
        self.results_table['selectmode'] = 'browse'  # do not allow selections
        # position the table
        self.results_table.grid(row=0, column=0, columnspan=2, sticky=N + E + S + W)
        # bind mouse button 1 release event so that we can capture clicks on rows to show result information
        self.results_table.bind('<ButtonRelease-1>', self.clicked_result_table_callback)
        # now the table has been created resize the columns to fit the content
        self.results_table.auto_size()
        # link the scrollbars to the table
        results_table_v_scroll.configure(command=self.results_table.yview)
        results_table_h_scroll.configure(command=self.results_table.xview)
        # position the scrollbars
        results_table_v_scroll.grid(row=0, column=2, sticky=N+S+E)
        results_table_h_scroll.grid(row=1, column=0, sticky=N+S+E+W)

        # enable/unlock the data input section
        self.unlock_data_input_section()
        # disable the processing series section
        self.lock_series_process_section()

        # create an instance of the GUI queue in the main thread for responding to GUI based method calls.
        # if other threads want to call a tkinter GUI function, then instead of calling the function in its thread,
        # pass the function to the GUI queue which is running in the main thread (this thread) and the queue system
        # will perform the GUI function call from the main thread. Function requests are added to a queue system.
        GuiQueue(self.root)

    def initialise_results_table(self, attributes_to_plot):

        column_ids = ["id", "one"]
        column_texts = ["ID", "Result Dir."]

        for trait_id in attributes_to_plot:
            text = Trait.TRAIT_IDS[trait_id]
            column_ids.append(trait_id)
            column_texts.append(text)

        self.results_table["columns"] = tuple(column_ids)
        for i in range(len(column_ids)):
            self.results_table.heading(column_ids[i], text=column_texts[i])
        # now the table has been created resize the columns to fit the content
        self.results_table.auto_size()

    # set a new image to a label (e.g. information icon)
    # @param label: the label to change the image of
    # @param photo: the tkinter photoimage object that represents the new image
    def update_label_photo(self, label, photo):
        # set the image to the photo
        label.configure(image=photo)
        # store the photo
        label.photo = photo

    # callback for click event on the results table
    # this function will open result directory of clicked row in OS if user clicked on the output directory column
    # this function will show plot of attribute for the column clicked on for image series of the clicked row
    def clicked_result_table_callback(self, event):
        # if the user did not click on a valid row or column then exit function
        if self.results_table.identify_column(event.x) == '' or self.results_table.identify_row(event.y) == '':
            return
        # if there is at least 1 row that has been selected in the results table
        if len(self.results_table.selection()) > 0:
            # get the first selected item (should only be one due to selection state of table)
            item = self.results_table.selection()[0]
            # get the column from the event
            column = self.results_table.identify_column(event.x)
            # get the id of the selected column
            column_id = self.results_table.column(column)["id"]
            # if the user clicked on the first column (series ID then do nothing)
            if column == "#1":
                return
            # if the user clicked on the second column (out directory)
            if column == "#2":
                # get the values of the row
                row_values = self.results_table.item(item).get("values")
                # get the output directory (value in 2nd column)
                directory = self.in_directory_text.get() + row_values[1]
                # open the directory via the OS
                self.open_directory(directory)
            # if any other column was clicked on (these will all be summary attributes)
            else:
                # if there was experimental data (meta information) then we are working with image series over time
                if self.experimental_data_value.get() is not Application.NO_EXPERIMENT_DATA_TEXT:
                    try:
                        # get the values of the row
                        row_values = self.results_table.item(item).get("values")
                        # get the output directory (value in 2nd column)
                        directory = self.in_directory_text.get() + row_values[1]
                        # create the path of the pre-generated plot image (output directory + column id)
                        # convert space and special characters (e.g. ^) to underscores as they won't be in the column name when used in filename
                        column_id_filename = column_id.replace(" ", "_").replace("^", "_")
                        plot_filename = os.path.join(directory, column_id_filename) + ".png"
                        # read the plot image
                        img = cv2.imread(plot_filename)
                        # convert to from BGR to RGB colour space
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        # show preview plot dialog with the loaded image
                        PreviewDialog(self.root, Application.PLOT_PREVIEW_DIALOG_TITLE_TEXT, img)
                    except:
                        messagebox.showerror("Cannot Show Plot", "Cannot show plot '" + os.path.basename(plot_filename) + "'")

    # when this method is called, show pop-up dialog box displaying information about experimental data input
    def show_experimental_data_info(self, event):
        # only show this pop-up when in data input mode
        if self.processing_state == ProcessingState.DATA_INPUT:
            # display the messagebox
            messagebox.showinfo("Meta Data Naming Convention", "Naming Convention:\n\n"
                                             "Image: 'YYYY-MM-DD_Exp-ID_Tray-No.jpg'\n"
                                             "Folder: 'Exp-ID_Tray-No'")

    # when this method is called, show pop-up dialog box displaying information about series processing section
    def show_series_processing_info(self, event):
        # only show this pop-up when in data input mode
        if self.processing_state == ProcessingState.DATA_INPUT:
            # display the messagebox
            messagebox.showinfo("Series Processing", "Check the box for each image series that you wish to process.")

    # when this method is called, show pop-up dialog box displaying information about machine learning pixel groups input
    def show_machine_learning_info(self, event):
        # only show this pop-up when in data input mode
        if self.processing_state == ProcessingState.DATA_INPUT:
            # display the messagebox
            messagebox.showinfo("Pixel Groups", "Please enter the number of pixel groups between " +
                                str(Application.MACHINE_LEARNING_PIXEL_GROUPS_MIN) + " and " +
                                str(Application.MACHINE_LEARNING_PIXEL_GROUPS_MAX) + " (inclusive).")

    # opens the specified directory via the OS
    # @param path: the path of the directory to open
    def open_directory(self, path):
        # as this is at OS level, we need to use different functions for different OS
        # if we are on windows
        if sys.platform == 'win32':
            # open the directory
            subprocess.Popen(['start', path], shell=True)
        # if we are OSX (mac)
        elif sys.platform == 'darwin':
            # open the directory
            subprocess.Popen(['open', path])
        # if we are on Linux
        # TODO: test on Linux
        elif sys.platform == 'linux2':
            # open the directory
            subprocess.Popen(['xdg-open', path])
        # if unrecognised OS then display error message
        else:
            messagebox.showinfo("Error", "Unrecognised OS: " + sys.platform + ". Unable to open directory.")

    # removes all image series analyses that are waiting to be processed in the queue.
    # this function will not affect any analysis currently running.
    def cancel_queued_analyses(self):

        # counter representing number of cancelled jobs
        cancelled_jobs = 0
        # counter representing index of 'future' processing task in future list (essentially the thread)
        future_counter = 0
        # once the user has pressed cancel, do not allow them to press cancel again (disable the button)
        self.analysis_button["state"] = 'disabled'
        # loop through the number of rows in the series processing table
        for row_index in range(len(self.series_processing_table.tick_list)):

            # NOTE: we know that the rows in the series processing table are ordered in the same way as the future
            # processing threads in the list of futures, as they the keys of the data series dictionary are first sorted
            # and added in this order to both the series processing table and the futures list.

            # if the series was selected for analysis in the series processing table
            if self.series_processing_table.tick_list[row_index]:
                # get the index of the 'future' object in the future list (essentially the thread)
                # this index is the number of selected image series we have encountered in the series processing table
                # so far in our iteration over all rows
                future = self.futures[future_counter]
                # if the thread/process is waiting to be processed (i.e. is currently not running and has not completed)
                if future.running() is False and future.done() is False:
                    # cancel the job
                    future.cancel()
                    # increment the number of cancelled jobs
                    cancelled_jobs += 1
                    # update the status column in the series processing table
                    self.update_treeview(self.series_processing_table, row_index, Application.STATUS_COL_INDEX, Application.CANCELLED_STATUS_TEXT)
                # increment the future counter
                future_counter += 1
        # if there were no cancelled jobs tell the user, otherwise they may not know what's happening
        if cancelled_jobs <= 0:
            # update the user
            messagebox.showinfo("Information", "There were no image series waiting in the queue to be cancelled.\n"
                                           "Image series that are currently being processed cannot be terminiated.")

    # return a dictionary/map of key value pairs which relate to experimental data extracted from specified folder name
    # @param full_folder_name: the path to the directory to extract experiment data/meta information from
    # returns none when there is an error
    # NOTE: the folder name should be 'exp-id_tray-number'
    @staticmethod
    def extract_experiment_data_from_folder_name(full_folder_name):
        # create dictionary to store extracted values
        meta = dict()
        try:
            # get the name of the folder
            name = os.path.basename(full_folder_name[:-1])
            # split the folder name based on experiment data delimiter (underscore)
            components = name.split("_")
            # at least 1 underscore must be present (between exp-id -> tray-number)
            if len(components) < 2:
                # if we don't have an underscore then return None to indicate error
                return None
            # we have at least 1 underscore, so we assume the data following the last underscore is the tray number
            # everything else can be the experiment-id
            # store the experiment id
            meta["exp_id"] = components[0]
            # iterate through the name components separated by underscores (except last as this will be tray number)
            for i in range(1, len(components) - 1):
                # as long as the component is not the penultimate component then append to experiment id
                # if it is the penultimate component and it is not 'Tray' then also append to experiment id
                # this is to ensure compatibility with folder names that contain the string 'Tray' before tray number
                if (i is len(components) -2 and components[i] == "Tray") == False:
                    # add the component to the experiment id
                    meta["exp_id"] += "_" + components[i]
            # there is no date information
            meta["date"] = "-"
            # get the tray number
            tray_number = components[-1]
            # convert tray number to integer - if this cast fails then an exception will be thrown and None returned
            meta["tray_number"] = int(tray_number)
            # directory is the folder name
            meta["directory"] = full_folder_name
            # key must be unique for each image series
            # when processing by folder, each folder is treated as 1 series, therefore foldername can be unique key
            meta["key"] = full_folder_name
            # return the dictionary
            return meta
        # there was an error, return None to indicate this
        except:
            return None

    # return a dictionary/map of key value pairs which relate to experimental data extracted from specified file name
    # @param full_file_name: the file name to extract experiment data/meta information from
    # returns none when there is an error
    # NOTE: the file name should be 'date_exp-id_tray-number', where date is in format (YYYY-MM-DD)
    @staticmethod
    def extract_experiment_data_from_file_name(full_file_name):
        # create dictionary to store extracted values
        meta = dict()
        try:
            # get the name of the file
            name = os.path.basename(full_file_name)
            # split the folder name based on experiment data delimiter (underscore)
            components = name.split("_")
            # at least 2 underscores must be present (between date -> exp-id and exp-id -> tray number)
            if len(components) < 3:
                # if we don't have an underscore then return None to indicate error
                return None
            # get the date
            date_str = components[0]
            # parse in required date format (to ensure compatibility)
            date = datetime.strptime(date_str, "%Y-%m-%d")
            # store the date
            meta["date"] = date_str
            # we have at least 2 underscores, so we assume the data following the last underscore is the tray number
            # everything else (except the first component, which is the date) can be the experiment-id
            # store the experiment id
            meta["exp_id"] = components[1]
            # iterate through the name components separated by underscores (from the 2nd component to the penultimate component, as the last will be tray number)
            for i in range(2, len(components) - 1):
                # as long as the component is not the penultimate component then append to experiment id
                # if it is the penultimate component and it is not 'Tray' then also append to experiment id
                # this is to ensure compatibility with folder names that contain the string 'Tray' before tray number
                if (i is len(components) - 2 and components[i] == "Tray") == False:
                    # add the component to the experiment id
                    meta["exp_id"] += "_" + components[i]
            # the directory will be the path of the file
            directory = os.path.dirname(full_file_name)
            # store the directory
            meta["directory"] = directory
            # get the tray component (as this is filename it will have extension following this, e.g. .png)
            tray_str = components[-1]
            # remove the file extension to leave just the tray number
            tray_number = tray_str[:tray_str.index('.')]
            # convert tray number to integer - if this cast fails then an exception will be thrown and None returned
            int(tray_number)
            # store the tray number
            meta["tray_number"] = tray_number
            # the key must be unique for each series
            # this series is the experiment id and tray number
            meta["key"] = meta["exp_id"] + "_" + meta["tray_number"]
            # return the dictionary of extracted experiment data
            return meta
        # there was an error, return None to indicate this
        except:
            return None

    # load images from the input directory and populate the series processing table
    def load(self):
        # if there is a input directory specified and the value is not an empty string
        if self.in_directory_text.get() is not None and self.in_directory_text.get() is not "":
            # clear the list of existing data series that have been loaded
            self.data_series.clear()
            # clear the series processing table
            self.series_processing_table.clear()
            # unlock the series processing section
            self.unlock_series_process_section()
            # the way that the series are loaded will depend upon the value selected for the experimental data/meta info

            # if experimental data is in the file name
            if self.experimental_data_value.get() == Application.EXPERIMENT_DATA_FROM_FILE_TEXT:
                # get all the files in the input directory
                for file in glob.glob(os.path.join(self.in_directory_text.get(), "*.*")):
                    # get the basename of the file
                    basename = os.path.basename(file)
                    # split the filename into its name and extension
                    file_parts = os.path.splitext(basename)
                    # get the extension
                    extension = file_parts[1]
                    # if the extension is one of the application's supported image file extensions
                    if extension in Application.SUPPORTED_FILE_TYPES:
                        # create a dictionary of experiment data extracted from the name
                        meta_dict = self.extract_experiment_data_from_file_name(file)
                        # if data extraction was successful (None if not successful)
                        if meta_dict is not None:
                            # get the id (will be used as dictionary key)
                            key = meta_dict["key"]
                            # have we already found a data series with the same id?
                            if key in self.data_series.keys():
                                # if so, then add this image filename and extracted date information to the series
                                self.data_series[key].add_file_date(meta_dict["date"], file)
                            # otherwise this is a new image series
                            else:
                                # create a new image series
                                s = Series()
                                # set the experimental reference
                                s.experiment_ref = meta_dict["exp_id"]
                                # set the tray number
                                s.tray_number = meta_dict["tray_number"]
                                # set the directory that the image is in
                                s.directory = meta_dict["directory"]
                                # set the root directory (chosen input directory)
                                # this will be the same when loading images by filename as it doesn't load sub directories
                                s.root_directory = self.in_directory_text.get()
                                # add this image filename and extracted date information to the series
                                s.add_file_date(meta_dict["date"], file)
                                # add the series to the list of series with its key set as the id
                                self.data_series[key] = s

            # if experimental data is in the folder name
            elif self.experimental_data_value.get() == Application.EXPERIMENT_DATA_FROM_FOLDER_TEXT:
                # get a list of all directories that are in specified input (root) directory
                for directory in glob.glob(os.path.join(self.in_directory_text.get(), "*", "")):
                    # we do not want to look in any folder that has the '_Processed_' in its name as
                    # this is likely to be an output result from this application
                    if "_Processed_" not in directory:
                        # get the meta data from the folder name
                        meta_dict = self.extract_experiment_data_from_folder_name(directory)
                        # if data extraction was successful (None if not successful)
                        if meta_dict is not None:
                            # get all of the file in the folder
                            for file in glob.glob(os.path.join(directory, "*.*")):
                                # split the filename into its name and extension
                                file_parts = os.path.splitext(file)
                                # get the extension
                                extension = file_parts[1]
                                # if the extension is one of the application's supported image file extensions
                                if extension in Application.SUPPORTED_FILE_TYPES:
                                    # when loading from folder there is no date information for each image file
                                    # therefore we use the modification date from the file as the image date
                                    # get the modification date
                                    m_time = os.path.getmtime(file)
                                    # convert it to YYYY-MM-DD format that we use
                                    date = datetime.fromtimestamp(m_time).strftime('%Y-%m-%d')
                                    # add this date to our experimental data/meta information dictionary
                                    meta_dict["date"] = date
                                    # get the id (will be used as dictionary key)
                                    key = meta_dict["key"]
                                    # have we already found a data series with the same id?
                                    if key in self.data_series.keys():
                                        # if so, then add this image filename and extracted date information to the series
                                        self.data_series[key].add_file_date(meta_dict["date"], file)
                                    # otherwise this is a new image series
                                    else:
                                        # create a new image series
                                        s = Series()
                                        # set the experimental reference
                                        s.experiment_ref = meta_dict["exp_id"]
                                        # set the tray number
                                        s.tray_number = meta_dict["tray_number"]
                                        # set the directory that the image is in (folder we are processing)
                                        s.directory = meta_dict["directory"]
                                        # set the root directory (chosen input directory)
                                        # this will be the parent of the folder we are processing
                                        s.root_directory = self.in_directory_text.get()
                                        # add this image filename and extracted date information to the series
                                        s.add_file_date(meta_dict["date"], file)
                                        # add the series to the list of series with its key set as the id
                                        self.data_series[key] = s
            # if there is no experimental data
            else:
                # each image will be processed separately but will belong to the same series (for processing purposes)
                # only images in the folder will be included, sub-directories are not processed
                # we create a key that will be used for all images for the series id when adding to data series map
                key = "All"
                # get all the files in the input directory
                for file in glob.glob(os.path.join(self.in_directory_text.get(), "*.*")):
                    # get the basename of the file
                    basename = os.path.basename(file)
                    # split the filename into its name and extension
                    file_parts = os.path.splitext(basename)
                    # get the extension
                    extension = file_parts[1]
                    # if the extension is one of the application's supported image file extensions
                    if extension in Application.SUPPORTED_FILE_TYPES:
                        # have we already found a data series with the same id?
                        if key in self.data_series.keys():
                            # if so, then add this image filename and extracted date information to the series
                            self.data_series[key].add_file_date("-", file)
                        # otherwise this is a new image series (will only happen when loading first image)
                        else:
                            # create a new image series
                            s = Series()
                            # there is no experimental reference
                            s.experiment_ref = "-"
                            # there is no tray number
                            s.tray_number = "-"
                            # set the directory that the image is in (same as root directory)
                            s.directory = self.in_directory_text.get()
                            # set the root directory (chosen input directory)
                            s.root_directory = self.in_directory_text.get()
                            # add the file
                            s.add_file_date("-", file)
                            # add the series to the list of series with its key set as the id
                            self.data_series[key] = s

            # now all valid images have been parsed and assigned to a series
            # counter representing the incremental unique series id assigned to a series
            counter = 0
            # get the series ids
            sorted_keys = self.data_series.keys()
            # sort them so that we have a known order (important for other processes)
            # also this presents series in alphabetical order (e.g. folder, experimental ref, and tray)
            sorted_keys.sort()
            # iterate through the ids (keys)
            for key in sorted_keys:
                # numerical series ids will start from 1, so increment here
                counter += 1
                # get the series associated with the key
                series = self.data_series[key]
                # set its numerical series id
                series.id = counter
                # if series information to the series processing table
                self.series_processing_table.insert("", END, values=[(series.id),  (Application.NOT_PROCESSED_TEXT),
                    (series.experiment_ref), (series.tray_number), (series.get_time_span()), (series.get_num_images())])

            # if there is at least 1 series then we generate preview image and clustering plot for machine learning settings section
            if len(self.data_series) > 0:
                # we want to choose an image which is somewhere in middle of growth
                # although not necessary the best way to do this, we first find the image series that has the most unique number of imaging dates
                # NOTE: possibly a better way would be to find the series with the greatest duration,
                # but this may cause problems when processing files with no experimental data as there is no date information.

                # store the greatest number of unique dates we have found in a series so far
                most_dates_in_a_series = None
                # store the series key associated with the greatest number of unique dates found so far
                series_with_most_dates = None
                # iterate through all of the series
                for key in self.data_series.keys():
                    # get the series
                    series = self.data_series[key]
                    # get the number of unique dates in this series (as dates as keys, we just get length of keys)
                    number_of_dates_in_series = len(series.date_file_dict.keys())
                    # if this is the greatest value encountered so far, or this is the first one
                    if most_dates_in_a_series is None or number_of_dates_in_series > most_dates_in_a_series:
                        # set the most dates and store the series key
                        most_dates_in_a_series = number_of_dates_in_series
                        series_with_most_dates = key

                # now we have found our sample series, retrieve its key
                sample_series = self.data_series[series_with_most_dates]
                # get a list of the dates of the images in the series
                dates = list(sample_series.date_file_dict.keys())
                # sort the dates (they are in YYYY-MM-DD format)
                dates.sort()
                # we assume that a date in the middle would be mid-growth.
                # again this is quite the assumption and this could be improved in the future.
                # get the index of the midpoint date
                n_dates = len(dates)
                middle_filename_index = int(n_dates/2)
                # get the associated image filename
                middle_filename = sample_series.date_file_dict[dates[middle_filename_index]][0]
                # read the image
                img = cv2.imread(middle_filename)
                # set this image as the sample image in the machine learning settings section
                self.set_machine_learning_images(img, Application.MACHINE_LEARNING_IMAGE_CANVAS_WIDTH)
            # if we have no loaded any image then we should remove any existing sample and clustering image
            # and show the default 'no image available' images
            else:
                # show the no image available images
                self.set_machine_learning_sample_image(self.no_image_available_img, Application.MACHINE_LEARNING_IMAGE_CANVAS_WIDTH)
                self.set_machine_learning_pixel_clustering_image(self.no_image_available_img, Application.MACHINE_LEARNING_IMAGE_CANVAS_WIDTH)
                # tell the user than no valid image series were found
                messagebox.showinfo("No Image Series Found",  "No valid image series were found in the input directory.")
        # no input directory specified
        else:
            # tell the user than no input directory was provided
            messagebox.showinfo("No Image Directory", "No valid input image directory was provided.")

    # this is the callback for when the user clicks the analysis button in the series processing section
    # it is multi-purpose and can function as run analysis, cancel queued analyses, or start new analysis
    def analysis_button_callback(self):
        # if the processing state of the application is at data input then the button function is run analysis
        if self.processing_state == ProcessingState.DATA_INPUT:
            # run the analyses for the selected image series
            self.run()
        # if the processing state of the application is running analyses then the button function is cancel
        elif self.processing_state == ProcessingState.RUNNING_ANALYSIS:
            # cancel all image series analyses that are currently waiting in the queue
            self.cancel_queued_analyses()
        # if the processing state of the application is analysis ended then the button function is to start a new analysis
        elif self.processing_state == ProcessingState.ANALYSIS_ENDED:
            # start a new analysis (reset GUI)
            self.new_analysis()

    # start a new analysis (reset GUI state)
    # this function must be run on main thread
    def new_analysis(self):

        # can we start a new analysis? (do not start a new analysis if there are threads running or waiting to run)
        # There are measures in place to stop this from happening (e.g. new analysis button has other state), but this is extra check
        # boolean flag to indicate whether we have found a running thread
        running = False
        # iterate through the list of futures (threads)
        for future in self.futures:
            # if the thread is not complete
            if future.done() is False:
                # then this is considered running or still waiting to run
                running = True
                # no need to continue searching, we can't start a new analysis
                break
        # if we found a thread that is waiting to run or is running
        if running:
            # show a message to the user informing them that a new analysis cannot be started
            messagebox.showinfo("Invalid Option", "Cannot start new analysis while current analysis is running.")
            # exit the function to stop new analysis
            return

        # set the processing state to the data input state
        self.processing_state = ProcessingState.DATA_INPUT
        # clear all of the logs
        self.log_frame.clear_all()
        # reset the analysis button to run analysis
        self.analysis_button["text"] = Application.ANALYSIS_BUTTON_TEXT_RUN
        # enable/unlock the data input section and machine learning sections
        self.unlock_data_input_section()
        self.unlock_machine_learning_section()
        # disable/lock the series processing section
        self.lock_series_process_section()
        # set inputs to default values
        self.in_directory_text.set("")
        self.in_rows_text.set(Application.DEFAULT_IN_ROWS)
        self.in_cols_text.set(Application.DEFAULT_IN_COLS)
        self.in_ref_radius.set(Application.DEFAULT_IN_REF_RADIUS)
        self.experimental_data_value.set(Application.DEFAULT_IN_EXPERIMENTAL_DATA_SOURCE)
        self.plant_species_value.set(Application.DEFAULT_IN_PLANT_SPECIES)
        self.pixel_groups_value.set(Application.DEFAULT_IN_PIXEL_GROUPS)
        # clear the series processing table
        self.series_processing_table.clear()
        # clear the results table
        # iterate through all of the rows
        for i in self.results_table.get_children():
            # delete the row
            self.results_table.delete(i)
        # clear the list of future instances
        self.futures = []
        # set machine learning image canvases to 'no image available' image
        self.set_machine_learning_sample_image(self.no_image_available_img, Application.MACHINE_LEARNING_IMAGE_CANVAS_WIDTH)
        self.set_machine_learning_pixel_clustering_image(self.no_image_available_img, Application.MACHINE_LEARNING_IMAGE_CANVAS_WIDTH)

    # run this function when the analysis has ended
    # this function must be run on main thread
    def completed_all_tasks(self):
        # show messagebox informing user that analysis has ended
        messagebox.showinfo("Analysis Completed", "The analysis has ended.")
        # enable the analysis button
        self.analysis_button["state"] = 'normal'
        # set its text to new analysis
        self.analysis_button["text"] = "New Analysis"

    # this function is called when a thread has completed processing an image series
    # this function must be run on main thread
    def task_complete(self, f):
        # check to see if all threads are finished
        # boolean flag that stores whether we have encountered an incomplete thread (initially set to true)
        all_tasks_completed = True
        # iterate through future instances (threads)
        for future in self.futures:
            # is the thread is not complete
            if future.done() is False:
                # there is still a thread processing an image series, so not all tasks are complete
                # update the flag
                all_tasks_completed = False
                # no need to continue searching
                break

        # if all image series have been processed
        if all_tasks_completed:
            # if the processing state has not already been set to end
            if self.processing_state is not ProcessingState.ANALYSIS_ENDED:
                # update the processing state to end
                self.processing_state = ProcessingState.ANALYSIS_ENDED
                # alert the user that all tasks have been completed and make GUI changes
                # as this function will run on non-main thread, we need to add this function to GUI queue
                GuiQueue.gui_queue.put(self.completed_all_tasks)

    # run the analysis
    def run(self):

        # check that the analysis can run (check for valid user inputs)
        try:
            # number of rows and columns must be > 0
            if self.in_cols_text.get() <= 0 or self.in_rows_text.get() <= 0:
                # tell the user than the analysis cannot start
                messagebox.showerror("Invalid Input", "The number of rows and columns must be greater than zero.")
                # do not continue with analysis
                return
        except:
            # tell the user than the analysis cannot start
            messagebox.showerror("Invalid Input", "The number of rows and columns must be greater than zero.")
            # do not continue with analysis
            return

        try:
            # check that the reference marker radius is > 0
            if self.in_ref_radius.get() <= 0:
                # tell the user than the analysis cannot start
                messagebox.showerror("Invalid Input", "The reference marker radius must be greater than zero.")
                # do not continue with analysis
                return
        except:
            # tell the user than the analysis cannot start
            messagebox.showerror("Invalid Input", "The reference marker radius must be greater than zero.")
            # do not continue with analysis
            return

        # check that number of rows and columns is equal to 1 if plant species is set to wheat
        # wheat code only supports a single pot
        if self.plant_species_value.get() == Application.PLANT_SPECIES_WHEAT and (self.in_cols_text.get() is not 1 or self.in_rows_text.get() is not 1):
            # tell the user than the analysis cannot start
            messagebox.showerror("Invalid Input", "When analysing wheat there must only be 1 row and 1 column.")
            # do not continue with analysis
            return

        # check that number of pixel groups in the machine learning settings section is within pre-determined range
        try:
            if self.pixel_groups_value.get() < Application.MACHINE_LEARNING_PIXEL_GROUPS_MIN or self.pixel_groups_value.get() > Application.MACHINE_LEARNING_PIXEL_GROUPS_MAX:
                # tell the user than the analysis cannot start
                messagebox.showerror("Invalid Input", "The number of pixel groups must be between " +
                                     str(Application.MACHINE_LEARNING_PIXEL_GROUPS_MIN) + " and " +
                                     str(Application.MACHINE_LEARNING_PIXEL_GROUPS_MAX) + " (inclusive).")
                # do not continue with analysis
                return
        except:
            # tell the user than the analysis cannot start
            messagebox.showerror("Invalid Input", "The number of pixel groups must be between " +
                                 str(Application.MACHINE_LEARNING_PIXEL_GROUPS_MIN) + " and " +
                                 str(Application.MACHINE_LEARNING_PIXEL_GROUPS_MAX) + " (inclusive).")
            # do not continue with analysis
            return

        # get the number of selected image series to process (rows ticked in series processing table)
        # counter storing the number of image series selected for processing
        selected_counter = 0
        # iterate through the series processing table tick list (list of booleans mapping to selected status)
        for i in range(len(self.series_processing_table.tick_list)):
            # if the current row (series id) has been selected for processing
            if self.series_processing_table.tick_list[i]:
                # increment the total counter
                selected_counter += 1
        # if no series have been selected for analysis
        if selected_counter <= 0:
            # tell the user than the analysis cannot start
            messagebox.showerror("Invalid Input", "No data series have been selected.")
            # do not continue with analysis
            return

        # before starting the analysis, check to see if the log files already exist
        # get a list of existing log files
        existing_log_files = self.log_frame.get_existing_log_files()
        # if there are existing log files
        if len(existing_log_files) > 0:
            # get a log filename list string
            log_str = ""
            # iterate through the list of log files
            for log_file in existing_log_files:
                # add the name of the log to the log filename list string
                log_str += os.path.basename(log_file) + "\n"
            # ask the user if they wish to continue (and overwrite the log files)
            if messagebox.askyesno("Overwrite Files?", "The following files already exist:\n\n"
                                + log_str + "\nDo you want to continue?\n\nWARNING: Continuing will overwrite these files.", default=messagebox.NO):
                # if the user wishes to continue then iterate through the log files
                for log_file in existing_log_files:
                    # clear the file
                    open(log_file, 'w').close()
            # if the user does not wish to continue then return and do not perform the analysis
            else:
                return

        # get a list of attributes to plot (will depend on plant species and experimental data source)
        # if there is no experimental data (we treat as single series but each image is independent)
        # therefore, we will not show plots in results table
        if self.experimental_data_value.get() == Application.NO_EXPERIMENT_DATA_TEXT:
            # the attributes to plot will be set to those required when no experimental data set
            attributes_to_plot = Trait.NO_EXPERIMENTAL_DATA_PLOT_IDS
        # otherwise it will depend upon plant species
        else:
            # set attributes to plot when analysing arabidopsis
            if self.plant_species_value.get() == Application.PLANT_SPECIES_ARABIDOPSIS:
                attributes_to_plot = Trait.ARABIDOPSIS_PLOT_IDS
            # otherwise set attributes to plot when analysing wheat
            elif self.plant_species_value.get() == Application.PLANT_SPECIES_WHEAT:
                attributes_to_plot = Trait.WHEAT_PLOT_IDS

        # analysis can commence - update the application processing state to running analysis
        self.processing_state = ProcessingState.RUNNING_ANALYSIS
        # set-up the results table based on the plant species and experimental data source
        self.initialise_results_table(attributes_to_plot)
        # update the analysis button to say cancel
        self.analysis_button["text"] = "Cancel Queued Jobs"
        # lock/disable the data input, machine learning, and series processing sections
        self.lock_data_input_section()
        self.lock_machine_learning_section()
        self.lock_series_process_section()

        # create a thread pool to perform image series analyses
        self.pool = ThreadPoolExecutor(Application.THREAD_POOL_SIZE)
        # list of future objects (threads)
        # the order will be the same as the order of the image series in the series processing table
        self.futures = []
        # get the data series keys and sort them to get a list of keys in same order as series processing table
        sorted_keys = self.data_series.keys()
        sorted_keys.sort()
        # counter representing row number in series processing table
        series_processing_table_row_index = 0
        # iterate through the sorted keys
        for key in sorted_keys:
            # get the corresponding image series
            series = self.data_series[key]
            # if the series has been selected for processing
            if self.series_processing_table.tick_list[series_processing_table_row_index]:
                # update the series processing table status column to say that the series is waiting in queue to be processed
                self.update_treeview(self.series_processing_table, series_processing_table_row_index, Application.STATUS_COL_INDEX, "In queue - waiting to process...")
                # store the row number (index) of the series in the series processing table
                series.result_index = series_processing_table_row_index
                # create an instance of analysis (core algorithm)
                analysis = Analysis(self, series, self.log_frame, attributes_to_plot)
                # run different algorithm depending upon plant species specified
                # if processing wheat then run wheat specific code
                if self.plant_species_value.get() == Application.PLANT_SPECIES_WHEAT:
                    # submit the job to the queue
                    future = self.pool.submit(analysis.perform_wheat_analysis)
                #  otherwise run the arabidopsis version of the algorithm
                else:
                    # submit the job to the queue
                    future = self.pool.submit(analysis.perform_arabidopsis_analysis)
                # add a callback to the thread so that it runs the task complete code
                # this is required to check whether all jobs have been completed
                future.add_done_callback(self.task_complete)
                # add this future to the list of futures
                self.futures.append(future)
            # if the row in the table has not been selected for analysis
            else:
                # update the series processing table to say that the series was not selected for analysis
                self.update_treeview(self.series_processing_table, series_processing_table_row_index, Application.STATUS_COL_INDEX, "Not selected for analysis")
            # increment the table row index (move to next row)
            series_processing_table_row_index += 1

    # disable/lock the data input section of the GUI
    # must be run on main thread
    def lock_data_input_section(self):
        # gray-out labels to make section appear disabled
        self.data_input_label.configure(fg='gray')
        self.input_directory_entry_label.configure(fg='gray')
        self.input_rows_entry_label.configure(fg='gray')
        self.input_columns_entry_label.configure(fg='gray')
        self.input_ref_radius_entry_label.configure(fg='gray')
        self.plant_species_dropdown_label.configure(fg='gray')
        self.experimental_data_dropdown_label.configure(fg='gray')
        self.name_convention_label.config(fg='gray')
        # disable input widgets
        self.input_directory_entry_box['state'] = 'disabled'
        self.input_rows_entry_box['state'] = 'disabled'
        self.input_columns_entry_box['state'] = 'disabled'
        self.input_ref_radius_entry_box['state'] = 'disabled'
        self.plant_species_selector['state'] = 'disabled'
        self.experimental_data_selector['state'] = 'disabled'
        # disable load and browse directory buttons
        self.load_button['state'] = 'disabled'
        self.show_input_directory_dialog_button['state'] = 'disabled'
        # change the information icon to the disabled version of the icon
        self.update_label_photo(self.experimental_data_info_icon_label, self.info_icon_disabled)

    # enable/unlock the data input section of the GUI
    # must be run on main thread
    def unlock_data_input_section(self):
        # change label font colour to black to indicate enabled
        self.data_input_label.configure(fg='black')
        self.input_directory_entry_label.configure(fg='black')
        self.input_rows_entry_label.configure(fg='black')
        self.input_columns_entry_label.configure(fg='black')
        self.input_ref_radius_entry_label.configure(fg='black')
        self.plant_species_dropdown_label.configure(fg='black')
        self.experimental_data_dropdown_label.configure(fg='black')
        self.name_convention_label.config(fg='black')
        # enable input widgets
        self.input_rows_entry_box['state'] = 'normal'
        self.input_columns_entry_box['state'] = 'normal'
        self.input_ref_radius_entry_box['state'] = 'normal'
        self.plant_species_selector['state'] = 'normal'
        self.experimental_data_selector['state'] = 'normal'
        # ensure that the input directory widget is still disabled (this can only be populated from open dialog)
        self.input_directory_entry_box['state'] = 'disabled'
        # enable load and browse directory buttons
        self.load_button['state'] = 'normal'
        self.show_input_directory_dialog_button['state'] = 'normal'
        # change the information icon to the enabled version of the icon
        self.update_label_photo(self.experimental_data_info_icon_label, self.info_icon)

    # disable/lock the machine learning section of the GUI
    # must be run on main thread
    def lock_machine_learning_section(self):
        # gray-out labels to make section appear disabled
        self.machine_learning_label.configure(fg='gray')
        self.sample_image_canvas_label.configure(fg='gray')
        self.pixel_clustering_canvas_label.configure(fg='gray')
        self.pixel_groups_label.configure(fg='gray')
        # disable input widgets
        self.pixel_groups_entry_box['state'] = 'disabled'
        # change the information icon to the disabled version of the icon
        self.update_label_photo(self.pixel_groups_info_icon_label, self.info_icon_disabled)

    # enable/unlock the machine learning section of the GUI
    # must be run on main thread
    def unlock_machine_learning_section(self):
        # change label font colour to black to indicate enabled
        self.machine_learning_label.configure(fg='black')
        self.sample_image_canvas_label.configure(fg='black')
        self.pixel_clustering_canvas_label.configure(fg='black')
        self.pixel_groups_label.configure(fg='black')
        # enable input widgets
        self.pixel_groups_entry_box['state'] = 'normal'
        # change the information icon to the enabled version of the icon
        self.update_label_photo(self.pixel_groups_info_icon_label, self.info_icon)

    # disable/lock the series processing section of the GUI
    # must be run on main thread
    def lock_series_process_section(self):
        # do not allow rows to be selected in the series processing table
        self.series_processing_table['selectmode'] = 'none'
        # lock the table so that checkbox cannot be changed
        self.series_processing_table.locked = True

    # enable/unlock the series processing section of the GUI
    # must be run on main thread
    def unlock_series_process_section(self):
        # allow a single row to be selected in the series processing table
        self.series_processing_table['selectmode'] = 'browse'
        # unlock the table so that checkbox can be changed
        self.series_processing_table.locked = False

    # update the content of a cell in a treeview/table
    # @param treeview: the treeview object to update
    # @param row_index: the row to update
    # @param column_index: the column to update
    # @param value: the new value of the cell at the specified row and column indices
    @staticmethod
    def update_treeview(treeview, row_index, column_index, value):
        # ensure that only 1 thread can make a change to the table at once
        with AutoSizeTreeView.lock:
            # get the row at the row index
            item = treeview.get_children()[row_index]
            # get the values of the cells in the row
            row_values = treeview.item(item).get("values")
            # update the value in the array at the specified column index
            row_values[column_index] = value
            # reassign the updated row to the treeview to make the change
            treeview.item(item, values=row_values)

    # show the input directory dialog
    def show_directory_dialog(self):
        # ask the user to specify the input image directory
        in_directory = filedialog.askdirectory(parent=self.root, mustexist=True, title='Please select an INPUT directory.')
        # if the user has not pressed cancel
        if in_directory != "":
            # update the input directory textbox with the chosen directory path
            self.in_directory_text.set(in_directory)

    # validation function that gets called on input boxes to ensure that entry is positive integer
    # @param input: the input value
    # returns true if valid input, otherwise false
    @staticmethod
    def validate_positive_int(input):
        # if the input is blank then this is okay
        if not input:
            return True
        # otherwise attempt to convert to integer
        try:
            int(input) # this will throw an exception if cannot convert to integer
            # if we have reached here then no exception thrown therefore valid integer
            return True
        # if a value error exception is thrown then not valid input, so return false
        except ValueError:
            return False

    # set the machine learning section canvas images bason on the provided BGR image
    # @param img: the BGR format image
    # @param canvas_width: the width of the canvas object that the image is to fit
    def set_machine_learning_images(self, img, canvas_width):
        # convert the BGR image to RGB format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # get the height and width of the image
        h, w = img.shape[:2]
        # get the aspect ratio of the image
        ratio = w / float(h)
        # compute the new height of the image based on canvas width and aspect ratio
        new_height = int(round(1.0 / float(ratio) * canvas_width))
        # generate pixel clustering plot and determine suggested number of pixel groups based on k-mean clustering
        pixel_clustering_figure, suggested_pixel_groups = Analysis.generate_machine_learning_pixel_clustering_figure(img)
        # resize the returned image to fit the canvas
        pixel_clustering_figure = cv2.resize(pixel_clustering_figure, (canvas_width, new_height))
        # set the sample image to the provided image
        self.set_machine_learning_sample_image(img, canvas_width)
        # set the pixel clustering image to the generated graph
        self.set_machine_learning_pixel_clustering_image(pixel_clustering_figure, canvas_width)
        # clamp the suggested number of pixel groups to within the limits of pre-determined bounds
        # clamp to minimum value
        if suggested_pixel_groups < Application.MACHINE_LEARNING_PIXEL_GROUPS_MIN:
            suggested_pixel_groups = Application.MACHINE_LEARNING_PIXEL_GROUPS_MIN
        # clamp to maximum value
        if suggested_pixel_groups > Application.MACHINE_LEARNING_PIXEL_GROUPS_MAX:
            suggested_pixel_groups = Application.MACHINE_LEARNING_PIXEL_GROUPS_MAX
        # set the number of pixel groups to the suggested number of pixel groups
        self.pixel_groups_value.set(suggested_pixel_groups)

    # set the machine learning section sample image based on the specified image
    # @param img: the RGB format image
    # @param canvas_width: the width of the canvas object that the image is to fit
    def set_machine_learning_sample_image(self, img, canvas_width):
        # get the height and width of the image
        h, w = img.shape[:2]
        # get the aspect ratio
        ratio = w / float(h)
        # compute the new height of the image based on canvas width and aspect ratio
        new_height = int(round(1.0/float(ratio) * canvas_width))
        # resize the image
        img = cv2.resize(img, (canvas_width, new_height))
        # convert the image to PIL format for photoimage
        pil_img = PIL.Image.fromarray(img)
        # store the photoImage (required for canvas)
        self.machine_learning_sample_image_photo = PIL.ImageTk.PhotoImage(image=pil_img)
        # create the canvas image from the photo object
        img_obj = self.sample_image_canvas.create_image(0, 0, image=self.machine_learning_sample_image_photo, anchor=NW)
        self.sample_image_canvas.itemconfig(img_obj, image=self.machine_learning_sample_image_photo)
        # update the canvas width and height
        self.sample_image_canvas.config(height=new_height, width=canvas_width)

    # set the machine learning pixel clustering image based on the specified image
    # @param img: the RGB format image
    # @param canvas_width: the width of the canvas object that the image is to fit
    def set_machine_learning_pixel_clustering_image(self, img, canvas_width):
        # get the height and width of the image
        h, w = img.shape[:2]
        # get the aspect ratio
        ratio = w / float(h)
        # compute the new height of the image based on canvas width and aspect ratio
        new_height = int(round(1.0/float(ratio) * canvas_width))
        # resize the image
        img = cv2.resize(img, (canvas_width, new_height))
        # convert the image to PIL format for photoimage
        pil_img = PIL.Image.fromarray(img)
        # store the photoImage (required for canvas)
        self.pixel_clustering_image_photo = PIL.ImageTk.PhotoImage(image=pil_img)
        # create the canvas image from the photo object
        img_obj = self.pixel_clustering_canvas.create_image(0, 0, image=self.pixel_clustering_image_photo, anchor=NW)
        self.pixel_clustering_canvas.itemconfig(img_obj, image=self.pixel_clustering_image_photo)
        # update the canvas width and height
        self.pixel_clustering_canvas.config(height=new_height, width=canvas_width)

# create the window
root = Tk()
# set the size of the window
root.geometry(Application.INITIAL_WINDOW_DIMENSIONS)
root.minsize(1024, 768)
# set the window title
root.title(Application.WINDOW_TITLE)

# check python version compatibility
# if we are not running python 2 then show compatibility info to user and then quit
if sys.version_info[0] is not 2:
    messagebox.showinfo("Python Version", "This application is designed to work with Python version 2.\n\n"
                                          "The installed version is Python " + str(sys.version_info[0]) + ".\n\n"
                                          "The application will now quit.")
else:
    # create the application
    Application(root)
    # set-up window layout
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    # set the main loop
    root.mainloop()
# exit application
root.quit()