import sys
if sys.version_info[0] == 2:
    from Tkinter import *
else:
    from tkinter import *


from gui_queue import GuiQueue
from scipy.signal import argrelextrema, argrelmax
from scipy import signal
import random
import plot_generator
import time
import os
import numpy as np
import csv
from skimage import img_as_float, img_as_ubyte
from skimage.transform import rescale
from skimage.morphology import dilation, erosion, remove_small_objects
from skimage.morphology import disk, remove_small_holes
from scipy.ndimage import binary_hit_or_miss
from scipy import ndimage
import matplotlib
import matplotlib.pyplot as plt
from skimage.measure import label, find_contours
from skimage.measure import regionprops
from skimage.draw import line_aa
from skimage import color
from skimage import filters
from skimage import exposure
from skimage import measure
from skimage.measure import label, find_contours
from skimage.measure import regionprops
import math
import cv2
from skimage.morphology import skeletonize
import gc # Garbage collection
import logging
from threading import Lock
from sklearn.neighbors import NearestNeighbors
from skimage.draw import circle
from skimage import io
from skimage.morphology import convex_hull_image
import findpeaks
import datetime
from trait_ids import Trait

from functools import partial
import clustering_plot
from series import Series


# class that wraps-up the core algorithm code to perform analysis of a single image series
class Analysis:

    # text to show in GUI when processing image series
    PROCESSING_LABEL = "Processing..."
    # text to show in GUI when image series has completed processing
    COMPLETED_LABEL = "Completed"
    # lock mechanism used to ensure that only 1 thread executes protected code block simultaneously (used for non-thread safe matplotlib code)
    lock = Lock()
    # this is the column index of the status column in the series processing table (used to know which column to use when updating status)
    STATUS_COL_INDEX = 1

    # constructor
    # @param app: a reference to the main application
    # @param series: the image series to analyse
    # @param logger_manager: a reference to the log manager widget
    def __init__(self, app, series, logger_manager, attributes_to_plot):
        # store a reference to the main application
        self.app = app
        # store the image series to process
        self.series = series
        # store the working root folder
        self.workingRootFolder = self.series.directory
        # use the result index of the series to store a reference to the correspondiong row in the series processing table
        self.item = self.app.series_processing_table.get_children()[self.series.result_index]
        # store the result directory
        self.Result_Directory = None
        # store a reference to the log manager widget
        self.logger_manager = logger_manager
        # store a reference to individual log widget that this series will write to (initially None)
        # this will be set when the main algorithm begins execution
        self.log = None
        # store the attributes to plot (graph)
        self.attributes_to_plot = attributes_to_plot



    # Function_1
    # Return the current date and time based on the OS
    @staticmethod
    def LocalTimeStamp():
        """Detect curret data and time"""
        # Current OS date and time
        currentDate = time.localtime(time.time())
        return currentDate

    # Function_0
    # Cross-platform delimiter
    @staticmethod
    def PlatformDelimiter():
        """Generate Delimiter based on Platforms"""
        # Current OS
        OS_Delimiter = ''
        if sys.platform == 'win32':
            # windows codes
            OS_Delimiter = '\\'
        elif sys.platform == 'darwin':
            # mac codes
            OS_Delimiter = '/'
        else:
            # other code here (linux)
            OS_Delimiter = '/'
            # Return the delimiter
        return OS_Delimiter

    # core algorithm for performing wheat analysis
    def perform_wheat_analysis(self):
        # use the log manager to retrieve an unassigned log to print messages to GUI
        self.log = self.logger_manager.get_first_unassigned_log()
        # set the progress status to 0, as this is the beginning of the analysis
        self.set_progress(0)
        try:
            self.log.print_to_file_log("Started the analysis")
            Platform_Delimiter = Analysis.PlatformDelimiter()

            # we need to get a list of files to process for this series (use date/file dictionary)
            # get all the image dates
            dates = list(self.series.date_file_dict.keys())
            # sort the dates (strings but in YYYY-MM-DD format)
            dates.sort()
            # get the total number of files to process (used for progress display)
            total_num_files = 0
            # iterate through the dates
            for date in dates:
                # get the number of images in the series for the current date (add this to the total number of images to process)
                total_num_files += len(self.series.date_file_dict[date])
            # get the GUI inputs
            # get the reference marker radius
            sticker_radius = int(self.app.in_ref_radius.get())
            # get the number of input rows
            input_rows = int(self.app.in_rows_text.get())
            # get the number of input columns
            input_cols = int(self.app.in_cols_text.get())
            # get the number of pixel cluster groups
            k_value = int(self.app.pixel_groups_value.get())
            # output the retrieved input parameters to the log file (useful when debugging)
            self.log.print_to_file_log("Root directory:", self.series.root_directory)
            self.log.print_to_file_log("Experiment reference:", self.series.experiment_ref)
            self.log.print_to_file_log("Tray number:", self.series.tray_number)
            self.log.print_to_file_log("Number of images:", total_num_files)
            self.log.print_to_file_log("Reference marker radius:", sticker_radius)
            self.log.print_to_file_log("Number of rows:", input_rows)
            self.log.print_to_file_log("Number of columns:", input_cols)
            self.log.print_to_file_log("Number of pixel groups:", k_value)
            self.log.print_to_file_log("Plant species:", self.app.plant_species_value.get())
            self.log.print_to_file_log("Meta source:", self.app.experimental_data_value.get())

            gc.enable()
            ##STEP 2.4: Set up a result folder to contain the processing results


            # Get the pre-processing date
            timeStamp = Analysis.LocalTimeStamp()
            curYear = timeStamp[0]
            curMonth = timeStamp[1]
            curDay = timeStamp[2]

            # Assemble the result folder
            Result_Folder = '_Processed_%d' % curDay + '-%d' % curMonth + '-%d' % curYear
            Result_Folder += "_" + str(self.series.experiment_ref) + "_Tray_" + str(self.series.tray_number)
            self.Result_Directory = self.workingRootFolder + Result_Folder
            self.print_to_log("Created output result directory:", self.Result_Directory)

            # If repeating the selection again with different criteria
            if not os.path.exists(self.Result_Directory):
                os.makedirs(self.Result_Directory)
            # Folder for processed results

            ImgDirectory = self.workingRootFolder

            # Loop based on the index of image files
            ##STEP 4.2: Create an empty csv file to save analysis results
            # chris changed ----------------
            output_csv_name = time.strftime("%Y-%m-%d") + "_LeafMeasure.csv"
            output_csv_path = os.path.join(self.Result_Directory, output_csv_name)
            # ------------------------------
            Leaf_file = open(output_csv_path, 'wb')

            result_List = [Trait.IMAGE_NAME, Trait.EXP_DATE, Trait.EXPERIMENT_REF, Trait.TRAY_NO, Trait.PROJECTED_LEAF_AREA,
                           Trait.CANOPY_LENGTH, Trait.CANOPY_WIDTH, Trait.LEAF_CANOPY_SIZE, Trait.LEAF_STOCKINESS,
                           Trait.LEAF_COMPACTNESS, Trait.GREENNESS, Trait.PIX2MM_RATIO]
            # Close the file object as the format of the csv is finished
            write_line = csv.writer(Leaf_file, delimiter=',', quoting=csv.QUOTE_NONE)
            write_line.writerow(result_List)
            Leaf_file.close()
            self.print_to_log("A CSV file called", output_csv_name, "is created in:", self.Result_Directory)

            ##STEP 4.3: Batch process the image series
            # Set up counters for processing stats
            pass_count, fail_count = (0, 0)

            # we need to iterate through all of the image files in the image series
            # so iterate through the list of sorted dates
            for date in dates:
                # for each date, iterate through the list of image files
                for file_to_process in self.series.date_file_dict[date]:
                    # this line is used to bridge the GUI code with the core algorithm code (used different variable names)
                    tmp_IMG_File = file_to_process
                    # calculate the progress (how many images processed vs total) and update GUI
                    self.set_progress((pass_count + fail_count)/float(total_num_files) * 100)
                    # An image used for analysis
                    # pattern match jpg or png files, make sure images are .jpg, .jpeg or .png
                    Image_FullName = tmp_IMG_File.split(Platform_Delimiter)[-1]
                    # Other parts of an image name
                    ImageName_Length = len(Image_FullName)
                    ImageDirectory = tmp_IMG_File[:(ImageName_Length * -1)]
                    ImageName = Image_FullName[:-4]
                    self.print_to_log("Start reading image: ", ImageName)

                    # ***Step_4.3.1***#
                    # Buffer the image file to the memory
                    img_tmp = io.imread(tmp_IMG_File)
                    Resize_Ratio = 1.0 / (img_tmp.shape[0] / 1024.0)  # dynamically transfer the original resolution
                    image_resized = img_as_ubyte(rescale(img_tmp.copy(), Resize_Ratio))
                    # The length (y-axis) of the image has been transferred to 1024 pixels
                    # Standardise the image size to improve processing efficiency and accuracy
                    # Set up a blank image for carrying following image objects in the process
                    Blank_Img = np.zeros((image_resized.shape[0], image_resized.shape[1]), dtype=np.uint8)
                    # Buffer only one blak image in the memory using numpy


                    # Start to process the image file
                    try:  # try...except is used to handle excpetional cases
                        # ***Step_4.3.2***# Detect Red reference points on the image
                        # The reference points mask is only generated for presentation purpose
                        Ref_Point_Array_Ref, Ref_Point_Areas, RefPoint_image = Analysis.RefPoints_Wheat(image_resized, Blank_Img)
                        # print "Ref points have been detected"

                        # ***Step_4.3.3***# Find the ratio of pixels in the detected red circular reference points
                        ## radius of the red sticker used in the wheat experiment is 5mm
                        # ..! This value shall be input from the GUI
                        pix2mm_ratio = Analysis.convert_pixels_to_mm(Ref_Point_Areas, sticker_radius)

                        # ***Step_4.3.4***# Perspective Transformation based on reference points
                        Transformed_img, New_Ref_Points_Array = Analysis.PerspectiveTrans_2D_Wheat(image_resized, Ref_Point_Array_Ref)
                        Blank_Img = np.zeros((Transformed_img.shape[0], Transformed_img.shape[1]), dtype=np.uint8)

                        #######################################
                        row_no = 1;
                        column_no = 1  # for wheat only
                        #######################################
                        Pot_Image, Pot_Segment_Refine = Analysis.PotSegmentation_Wheat(Transformed_img, row_no, column_no,
                                                                              New_Ref_Points_Array)
                        # print "Pots have been detected"

                        # ***Step_4.3.6***# Generate segmented pot sections, for a rows x cols tray
                        # Denoising using OpenCV to smooth the leaf surface so that shadows can be presented
                        # in a linear manner. Skimage functions are too slow for this process
                        Transformed_img_Denoised = cv2.fastNlMeansDenoisingColored(Transformed_img, None, 10, 10, 5,
                                                                                   searchWindowSize=15)
                        dilated_lab_Img, leaf_LAB_img = Analysis.LAB_Img_Segmentation_Wheat(Transformed_img_Denoised, pix2mm_ratio)
                        # Rescale lab image so that it can be used for leaf intensity check
                        # LAB colour spacing deals with green-related lights in uniform chromaticity diagram
                        # This function is designed for select features from the mature leaves
                        p0, p100 = np.percentile(leaf_LAB_img[:, :, 1], (0, 100))
                        leaf_LAB_img_rescale = exposure.rescale_intensity(leaf_LAB_img[:, :, 1], in_range=(p0, p100))
                        # print "LAB have been detected"

                        # ***Step_4.3.7***# Use Excessive greenness and kmeans to define pixel clustering
                        # Four groups were set: green leaves, soil/compost, reference points and others such as reflection
                        img_EGreen = Analysis.compute_greenness_img(Transformed_img.copy())
                        # Reduce the RGB to excessive greenness image for ML performance
                        img_ref = img_EGreen.copy()
                        # Use kmeans to cluster pixel groups
                        kmeans_img_final = Analysis.kmeans_cluster(img_ref, k_value)  # the value of k can be changed, maximum k = 10
                        kmeans_mask = kmeans_img_final > np.median(kmeans_img_final) * 1.025  # ONLY Five Pixel Groups
                        # print "kmeans clustering has been detected"

                        # ***Step_4.3.8***# Finish finding the leaf mask
                        dilated_kmeans_Img = dilation(kmeans_mask, disk(3))
                        leaf_mask_final = np.logical_and(dilated_lab_Img, dilated_kmeans_Img)  # Leaf ROI
                        leaf_mask_ref_1 = np.logical_and(leaf_mask_final, dilation(kmeans_mask, disk(1)))
                        leaf_mask_ref = remove_small_holes(leaf_mask_ref_1, 125)  # remove small holes in the mask
                        Leaf_size_overall = int(np.sum(leaf_mask_ref) * 0.01)  # Calculate 1% of the total leaf area

                    except IOError:
                        self.print_to_log("Error: can\'t find file or read data")
                        fail_count += 1
                    except ValueError:
                        self.print_to_log('Non-numeric data found in the file.')
                        fail_count += 1
                    except ImportError:
                        self.print_to_log("Import module(s) cannot be found")
                        fail_count += 1
                    # Finish the section for running functions
                    except Exception as e:
                        self.print_to_log("Process failed due to analysis issues, see the log file for more information...")
                        self.print_to_log("\t\t Continuing with the next image!")
                        self.log.print_to_file_log("Exception:", e)
                        fail_count += 1
                        gc.collect()
                        continue

                        ##STEP 4.4: Quantify traits from the images
                    try:
                        # use the locking mechanism here as we are using matplotlib
                        with Analysis.lock:
                            # Start to extract the quantification
                            if leaf_mask_ref.mean() > 0:  # The leaf mask contains information
                                # ***Step_4.4.1***# Prepare leaf measurements
                                ## Step_4.4.1.1 Prepare lists for carrying measurements
                                pot_array = []  # collect the pot location
                                label_array = []  # Collect labels for objects that have been selected
                                no_Count = 0  # used for counting the iteration

                                ## Step_4.4.1.2 Prepare images for carrying processed images
                                Final_Leaf_img = Blank_Img.copy()
                                Final_Leaf_Skeleton = Blank_Img.copy()
                                Final_Tip_img = Blank_Img.copy()
                                Leaf_hull_Final = Blank_Img.copy()
                                Leaf_hull_Outline_Final = Blank_Img.copy()
                                (Leaf_Region_Width, Leaf_Region_Length) = (0, 0)

                                # Prepare images for carry leaf tip detection
                                blank_h, blank_w = Blank_Img.shape[:2]
                                Final_Leaf_Sweep_img = np.zeros((blank_h, blank_w, 3), np.uint8)
                                hull_region = None

                                ## Step_4.4.1.3 For segmenting pots, use ndimage to extract pot level features
                                Labelled_Pot_Img, num_features = ndimage.measurements.label(Pot_Segment_Refine)
                                ## Step_4.4.1.4 Prepare the processed image
                                fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(15, 10))
                                # The background of the processed image
                                ax1.imshow(Transformed_img)
                                ax2.imshow(Transformed_img)

                                # ***Step_4.4.2***# Start to generate leaf measurements
                                # Go through every recognised pot
                                for pot_region in regionprops(Labelled_Pot_Img, cache=True):

                                    ## Step_4.4.2.1 skip too small or too large objects
                                    if (pot_region.area > (0.925 * Transformed_img.size / (row_no * column_no))) or (
                                        pot_region.area <
                                        (0.075 * Transformed_img.size / (row_no * column_no))):
                                        continue  # Ignore the pot as to big/small***")

                                    ## Step_4.4.2.2 Prepare temporary pot image base
                                    pot_tmp_img = Blank_Img.copy()
                                    pot_coord = pot_region.coords.astype(int)
                                    # coordinates based on labelled outlines
                                    pot_tmp_img[pot_coord[:, 0], pot_coord[:, 1]] = 1
                                    # expand the pot region to include leaves growing outside the pot
                                    kernel = np.ones((18, 18), np.uint8)  # Use OpenCV to increase the dilation performance
                                    pot_tmp_img_Ref = cv2.dilate(pot_tmp_img, kernel, iterations=1)

                                    ## Step_4.4.2.3 Generate temporary leaf image base
                                    Overlap_Leaf_tmp = Blank_Img.copy()
                                    # Use leaf_mask_final to leaf level detection, use leaf_mask_ref for leaf area measurement
                                    Overlap_Leaf_tmp = np.logical_and(leaf_mask_ref, pot_tmp_img_Ref)
                                    Overlap_Leaf_tmp_Ref = Blank_Img.copy()
                                    Leaf_Area_tmp = 0.0

                                    ## Step_4.4.2.4 Start to trim leaves in every pot
                                    # Normalised LAB is used as the intensity template for feature seleciton
                                    Labelled_leaf_Img = measure.label(Overlap_Leaf_tmp)
                                    for leaf_region in regionprops(Labelled_leaf_Img, intensity_image=leaf_LAB_img_rescale,
                                                                   cache=True):
                                        # Remove stones and other non-green small objects
                                        if leaf_region.area <= 25:  # the resolution is fixed
                                            continue
                                        Intensity_Diff = leaf_region.max_intensity - leaf_region.mean_intensity
                                        if Intensity_Diff < 0.15 and leaf_region.area < 125:  # Deviation from the scaled difference,
                                            continue
                                            # After rescaling, the most green part has a value -1, 1SD round to 65% -> 35%
                                        if leaf_region.mean_intensity > -0.3 or leaf_region.min_intensity > -0.6:
                                            # the LAB has been rescaled between -1.0 and 0 for presenting greenness
                                            continue
                                        # The position of the leaf object
                                        if leaf_region.centroid[0] < pot_region.bbox[0] * 1.025 or leaf_region.centroid[0] > \
                                                        pot_region.bbox[2] * 0.975:
                                            continue  # Column, y-axis
                                        if leaf_region.centroid[1] < pot_region.bbox[1] * 1.025 or leaf_region.centroid[1] > \
                                                        pot_region.bbox[3] * 0.975:
                                            continue  # Row, x-axis
                                        leaf_contrast = Intensity_Diff * 1.0 / (
                                        leaf_region.max_intensity + leaf_region.min_intensity)
                                        if (leaf_contrast > -0.075 and leaf_region.area < 525):
                                            continue
                                        # Remove based on contrast levels and intensity values
                                        # print (no_Count+1), ': ', leaf_contrast, " - ", Intensity_Diff, " - ", leaf_region.area
                                        # print leaf_region.mean_intensity, "_", leaf_region.min_intensity
                                        leaf_coord = leaf_region.coords.astype(int)
                                        Overlap_Leaf_tmp_Ref[leaf_coord[:, 0], leaf_coord[:, 1]] = 1
                                        # Final refinement
                                    Final_Leaf_Clean = remove_small_objects(dilation(leaf_mask_ref, disk(3)),
                                                                            Leaf_size_overall * 0.15)
                                    Overlap_Leaf_tmp_Ref = np.logical_and(Final_Leaf_Clean, Overlap_Leaf_tmp_Ref)
                                    Overlap_Leaf_tmp_Ref = remove_small_objects(Overlap_Leaf_tmp_Ref, Leaf_size_overall * 0.32)
                                    Final_Leaf_img = np.logical_or(Overlap_Leaf_tmp_Ref, Final_Leaf_img)
                                    Leaf_Area_tmp = np.sum(Overlap_Leaf_tmp_Ref) * 1.0 / (pix2mm_ratio * pix2mm_ratio)
                                    # print "Finish processing leaf surface in pot %d!" %(no_Count + 1)
                                    ## Finish trimming leaves

                                    ## Step_4.4.2.5 Label pots
                                    no_Count = no_Count + 1
                                    ax1.text(pot_region.bbox[1] + 25, pot_region.bbox[0] + 25, '%d' % (no_Count), ha='center',
                                             va='bottom', color='lime', size=24)  # altered from +25 and +25
                                    ax2.text(pot_region.bbox[1] + 25, pot_region.bbox[0] + 25, '%d' % (no_Count), ha='center',
                                             va='bottom', color='lime', size=24)  # altered from +25 and +25

                                    ## Step_4.4.2.6 Extract Convex Hull for leaf objects
                                    # Add convex hull to the leaf objects
                                    Leaf_hull_tmp = Blank_Img.copy()
                                    Leaf_hull_Outline = Blank_Img.copy()
                                    if np.sum(Overlap_Leaf_tmp_Ref) > 0:  # Plants found in a pot
                                        Leaf_hull_tmp = convex_hull_image(Overlap_Leaf_tmp_Ref)
                                        Leaf_hull_Outline = np.logical_xor(Leaf_hull_tmp, erosion(Leaf_hull_tmp, disk(2)))
                                        Leaf_hull_Outline_Final = np.logical_or(Leaf_hull_Outline_Final, Leaf_hull_Outline)
                                        Leaf_hull_Final = np.logical_or(Leaf_hull_Final, Leaf_hull_tmp)
                                    # Leaf canopy
                                    Hull_area = np.sum(Leaf_hull_tmp)
                                    # generate a circle to present the centroid of convex hull
                                    Leaf_Centroid_TMP = Blank_Img.copy()
                                    if (np.sum(Leaf_hull_tmp) > 0):  # Plant canopy found in a pot
                                        # Measure the region properties of the leaf canopy
                                        label_hull_img = label(Leaf_hull_tmp, connectivity=Leaf_hull_tmp.ndim)
                                        hull_region = measure.regionprops(label_hull_img, intensity_image=leaf_LAB_img_rescale,
                                                                          cache=True)
                                        # Measure the full length and width of the canopy
                                        Leaf_Region_Width = round(hull_region[0].minor_axis_length, 0)
                                        Leaf_Region_Length = round(hull_region[0].major_axis_length, 0)
                                        Leaf_Region_Radius = (Leaf_Region_Length * 1.0) / 2
                                        # to remove early small leaves, as the calculation is dynamic
                                        Leaf_Region_Hull = (2.75 * pix2mm_ratio) * sticker_radius  # diameter is 4 mm, 50% of final size
                                        rr, cc = circle(hull_region[0].centroid[0], hull_region[0].centroid[1], Leaf_Region_Hull)
                                        Leaf_Centroid_TMP[rr, cc] = 1  # A filled circle is generated
                                    else:  # an empty canopy region
                                        Leaf_Region_Width = 0
                                        Leaf_Region_Length = 0
                                        Leaf_Region_Radius = 0
                                    # Finish detecting leaf canopy
                                    # Overlap_Leaf_tmp_Ref
                                    # print "Finish processing leaf canopy in pot %d!" %(no_Count)


                                    # ***Step_4.4.3***# Append the quantification to the csv file
                                    # chris changed --------------
                                    Leaf_file = open(output_csv_path, 'a')

                                    # get the series information from the series instance
                                    Imaging_Date = date
                                    Genotype_Treatment_tmp = self.series.experiment_ref
                                    Tray_No_tmp = self.series.tray_number

                                    #Leaf_file = open(Result_Directory + "/" + time.strftime("%Y-%m-%d") + "_LeafMeasure.csv", 'a')
                                    #if ImageName.count('_') < 2:  # At least two underscores need to be found
                                    #    # wrong format of the image name
                                    #    Imaging_Date = ''
                                    #    Genotype_Treatment_tmp = ''
                                    #    Tray_No_tmp = ''
                                    #elif ImageName.count('_') >= 4:
                                    #    # Proper annotation
                                    #    Imaging_Date = ImageName.split("_")[0]
                                    #    Genotype_Treatment_tmp = ImageName.split("_")[1] + "_" + ImageName.split("_")[2]
                                    #    Tray_No_tmp = ImageName.split("_")[4]
                                    #else:
                                    #    # Format with two-three '_' in the image name
                                    #    Imaging_Date = ImageName.split("_")[0]
                                    #    Genotype_Treatment_tmp = ''
                                    #    for elem in ImageName.split("_")[1:-2]:
                                    #        Genotype_Treatment_tmp = Genotype_Treatment_tmp + elem
                                    #    Tray_No_tmp = ImageName.split("_")[-1]
                                    #---------------------------------------------------------------------------

                                    if np.sum(Overlap_Leaf_tmp_Ref) < Leaf_size_overall * 0.2 and Leaf_Area_tmp > 1000:
                                        # only applying when the leaf area is over 1000 mm^2
                                        Leaf_Region_Radius = 0  # not counted for trait analysis

                                    # The string to be appended
                                    if Leaf_Region_Radius > 0:
                                        Leaf_Stockiness = 4.0 * math.pi * (np.sum(Overlap_Leaf_tmp_Ref)) / math.pow(
                                            (2 * math.pi * Leaf_Region_Radius), 2)
                                        result_List = [ImageName, Imaging_Date, Genotype_Treatment_tmp, Tray_No_tmp,
                                                       round(np.sum(Overlap_Leaf_tmp_Ref) * 1.0 / (pix2mm_ratio * pix2mm_ratio), 1),
                                                       round(Leaf_Region_Length * 1.0 / pix2mm_ratio, 1),
                                                       round(Leaf_Region_Width * 1.0 / pix2mm_ratio, 1),
                                                       round(Hull_area * 1.0 / (pix2mm_ratio * pix2mm_ratio), 1),
                                                       round(np.sum(Overlap_Leaf_tmp_Ref) * 100.0 / Hull_area, 1),
                                                       round(Leaf_Stockiness * 100, 1),
                                                       round(abs(hull_region[0].mean_intensity * 1.125) * 255.0, 1),
                                                       round(pix2mm_ratio, 1)
                                                       ]
                                    else:  # No leaf found in the pot
                                        result_List = [ImageName, Imaging_Date, Genotype_Treatment_tmp, Tray_No_tmp,
                                                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, round(pix2mm_ratio, 1)
                                                       ]

                                    write_line = csv.writer(Leaf_file, delimiter=',', quoting=csv.QUOTE_NONE)
                                    write_line.writerow(result_List)
                                    Leaf_file.close()
                                    self.print_to_log("\tFinish exporting leaf measurements in pot", str(no_Count))
                                    # Finish exporting to the csv file for leaf measurements


                                    ##STEP 5: Export processed images
                                ## Step_5.1 Generate contour images for leaf outlines and pot ID
                                contours_image = np.logical_and(Final_Leaf_Clean, dilation(leaf_mask_ref, disk(0)))
                                contours2 = find_contours(Leaf_hull_Outline_Final, 0.9)
                                for n2, contour2 in enumerate(contours2):
                                    ax1.plot(contour2[:, 1], contour2[:, 0], linewidth=2, color='yellow')

                                ax2.imshow(np.logical_and(np.invert(Final_Leaf_img), Pot_Segment_Refine), cmap='gray')
                                # Prepare the processed figure
                                ax2.set_title('Refined Projected Leaf Regions in pots', fontsize=12)
                                ax2.axis([-25, Final_Leaf_img.shape[1] + 25, Final_Leaf_img.shape[0] + 25, -25])
                                # The following can handle different platforms
                                # Prepare the processed figure
                                ax1.set_title('Leaf Canopy', fontsize=12)
                                ax1.axis([-25, Final_Leaf_img.shape[1] + 25, Final_Leaf_img.shape[0] + 25, -25])
                                plt.tight_layout()
                                # The following can handle different platforms
                                fig.savefig(os.path.join(self.Result_Directory + Platform_Delimiter,
                                                         (ImageName + '_04_Leaf_Analysis.jpg')), bbox_inches='tight')
                                plt.close(fig)  # Close the figure

                                ## Step_5.2 Output processed images - calibration
                                # Produce processed images
                                fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(15, 10))
                                ax1.set_title('Original Image', fontsize=12)
                                ax1.imshow(image_resized)  # The background of the processed image
                                ax2.set_title('Transformed Image', fontsize=12)
                                ax2.imshow(Transformed_img_Denoised)  # Transformed image
                                # Prepare the processed figure
                                plt.axis([-25, Final_Leaf_img.shape[1] + 25, Final_Leaf_img.shape[0] + 25, -25])
                                plt.tight_layout()
                                # The following can handle different platforms
                                fig.savefig(os.path.join(self.Result_Directory + Platform_Delimiter,
                                                         (ImageName + '_01_Image_Calibration.jpg')), bbox_inches='tight')
                                plt.close(fig)  # Close the figure

                                ## Step_5.3 Output processed images - ML clustering
                                # Produce processed images
                                fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(15, 10))
                                ax1.set_title('LAB Colour Space with Green-Red Colour Opponents', fontsize=12)
                                ax1.imshow(leaf_LAB_img[:, :, 2])  # The non-linear LAB image
                                ax2.set_title('Colour Clustering Groups Using Kmeans', fontsize=12)
                                ax2.imshow(kmeans_mask, cmap='rainbow')
                                # Prepare the processed figure
                                ax1.axis([-25, Final_Leaf_img.shape[1] + 25, Final_Leaf_img.shape[0] + 25, -25])
                                ax2.axis([-25, Final_Leaf_img.shape[1] + 25, Final_Leaf_img.shape[0] + 25, -25])
                                plt.tight_layout()
                                # The following can handle different platforms
                                fig.savefig(os.path.join(self.Result_Directory + Platform_Delimiter,
                                                         (ImageName + '_02_Colour_Clustering.jpg')), bbox_inches='tight')
                                plt.close(fig)  # Close the figure

                                ## Step_5.4 Output processed images - Leaf extraction
                                # Produce processed images
                                fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(15, 10))
                                # The background of the processed image
                                ax2.imshow(Transformed_img_Denoised)  # The denoised image

                                contours1 = find_contours(erosion(remove_small_holes(Final_Leaf_Clean, min_size=525), disk(3)), 0.9)
                                for n1, contour1 in enumerate(contours1):
                                    ax2.plot(contour1[:, 1], contour1[:, 0], linewidth=2, color='yellow')
                                ax2.set_title('Outline of Leaves of Interest', fontsize=12)
                                # Crop leaf for presentation
                                Leaf_Crop = Transformed_img_Denoised.copy()
                                for layer in range(Leaf_Crop.shape[-1]):
                                    Leaf_Crop[np.where(np.invert(erosion(Final_Leaf_Clean, disk(3))))] = 1
                                ax1.set_title('Leaf Extraction using Lab and Kmeans', fontsize=12)
                                ax1.imshow(Leaf_Crop)
                                # Prepare the processed figure
                                ax1.axis([-25, Final_Leaf_img.shape[1] + 25, Final_Leaf_img.shape[0] + 25, -25])
                                ax2.axis([-25, Final_Leaf_img.shape[1] + 25, Final_Leaf_img.shape[0] + 25, -25])
                                plt.tight_layout()
                                # The following can handle different platforms
                                fig.savefig(os.path.join(self.Result_Directory + Platform_Delimiter,
                                                         (ImageName + '_03_Leaf_Extraction.jpg')), bbox_inches='tight')
                                plt.close(fig)  # Close the figure

                                ##STEP 6: Processing stats and garbage collection
                                pass_count += 1
                                gc.collect()
                                self.print_to_log("Successfully processed: ", ImageName)

                            # If an image does not contain any leaf information
                            else:
                                self.print_to_log("Error: can\'t find any leaf information!")
                                fail_count += 1
                    except Exception as e:
                        # In case we failed in the process
                        self.print_to_log("Process failed due to image analysis issues... Continuing with the next image!")
                        self.log.print_to_file_log("Exception:", e)
                        fail_count += 1
                        gc.collect()
                        # Clear memory
                        del img_tmp, image_resized, Transformed_img, Transformed_img_Denoised
                        del leaf_LAB_img, leaf_LAB_img_rescale, img_EGreen
                        del img_ref, kmeans_img_final, kmeans_mask
                        continue

                    # One image has been processed
                    gc.collect()
                    # Clear memory for the batch processing
                    del img_tmp, image_resized, Transformed_img, Transformed_img_Denoised
                    del leaf_LAB_img, leaf_LAB_img_rescale, img_EGreen
                    del img_ref, kmeans_img_final, kmeans_mask
                    self.log.print_to_file_log("garbage collected")
                    # Plot results of the processed image

            # chris added -----------
            attribute_button_text = ["-"] * len(self.attributes_to_plot)

            # if this is truly an image series (i.e. there is date information)
            if date is not "-":
                # we want to produce plots for each plot attribute
                # use the locking mechanism to ensure that no other thread is using matplotlib functionality whilse we are using it here
                with Analysis.lock:
                    # iterate through the attributes to plot (these are the plots we will show in the GUI in the results table)
                    for attribute in self.attributes_to_plot:
                        # some of the attribute names include characters that we shouldn't or can't use in filename
                        # therefore, we replace these characters with underscores
                        attribute_without_spaces = attribute.replace(" ", "_").replace("^", "_")
                        # create suitable plot output filename
                        output_plot_filename = os.path.join(self.Result_Directory,
                                                            str(attribute_without_spaces + ".png"))
                        # generate the plot for this attribute for this image series
                        img = plot_generator.generate_plot(output_csv_path,
                                                           self.series.experiment_ref, self.series.tray_number,
                                                           attribute)
                        # before we save, we need to convert from RGB to BGR format when using OpenCV functionality
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        # save the plot to file
                        cv2.imwrite(output_plot_filename, img)


                # in the GUI we also display the minimum and maximum mean values for each plotted attribute
                # this is the min and max of the mean value from each image analysed

                # counter storing the number of attributes we have got min and max mean values for
                attribute_counter = 0
                # iterate through the list of attributes to plot
                for attribute in self.attributes_to_plot:
                    # get the min and max mean value
                    min_value, max_value = Analysis.get_trait_min_max_means(output_csv_path, attribute)
                    # the total leaf number should be represented as integer, so round to int if we are processing this attribute
                    if attribute == Trait.LEAF_TOTAL_NO:
                        # round the min
                        min_value = round(min_value)
                        # round the max
                        max_value = round(max_value)
                    # otherwise, just round to 1 decimal place (for display purposes)
                    else:
                        # round the min
                        min_value = round(min_value, 1)
                        # round the max
                        max_value = round(max_value, 1)
                    # get the text string to display in the results table, which will be min to the max
                    text = str(min_value) + " - " + str(max_value)
                    # update the button text in the list of button text values
                    attribute_button_text[attribute_counter] = text
                    # increment the counter to move onto the next attribute
                    attribute_counter += 1

            # get the output result directory
            relative_result_directory = self.Result_Directory[len(self.series.root_directory):]

            # generate a list of row values (it will be at least series id and out directory)
            # then a list of min/max mean values for the attributes to display in GUI
            result_row_values = [(str(self.series.id)), (relative_result_directory)]
            # iterate through the list of attributes to plot
            for i in range(len(self.attributes_to_plot)):
                # append the string to the list of values to show in table row
                result_row_values.append(attribute_button_text[i])
            # update the GUI accordingly, as we are not in the main thread, push functionality to update GUI to GUI queue
            GuiQueue.gui_queue.put(partial(self.app.results_table.insert, "", END, values=result_row_values))
            # analysis complete to update status column in series processing table
            self.update_tree("Complete")
            # ----------------------

            self.print_to_log("\nFinish the analysis - see results in: ", self.Result_Directory)
            self.print_to_log("Processing Summary:")
            self.print_to_log("\tThere were %d images successfully processed, with %d failures" % (pass_count, fail_count))

        except Exception as e:
            self.print_to_log("There was an error processing the result. Analysis terminated.")
            self.log.print_to_file_log("Exception:", e)
            self.update_tree("Error: Analysis Terminated")
        finally:
            # detach the log (this means the the log associated with this thread/process is flagged as being available for output information for another process)
            self.log.detach()

    # this is the code to run when processing arabidopsis data
    def perform_arabidopsis_analysis(self):
        # use the log manager to retrieve an unassigned log to print messages to GUI
        self.log = self.logger_manager.get_first_unassigned_log()
        # set the progress status to 0, as this is the beginning of the analysis
        self.set_progress(0)
        try:
            self.log.print_to_file_log("Started the analysis")
            Platform_Delimiter = Analysis.PlatformDelimiter()

            # we need to get a list of files to process for this series (use date/file dictionary)
            # get all the image dates
            dates = list(self.series.date_file_dict.keys())
            # sort the dates (strings but in YYYY-MM-DD format)
            dates.sort()
            # get the total number of files to process (used for progress display)
            total_num_files = 0
            # iterate through the dates
            for date in dates:
                # get the number of images in the series for the current date (add this to the total number of images to process)
                total_num_files += len(self.series.date_file_dict[date])
            # get the GUI inputs
            # get the reference marker radius
            sticker_radius = int(self.app.in_ref_radius.get())
            # get the number of input rows
            input_rows = int(self.app.in_rows_text.get())
            # get the number of input columns
            input_cols = int(self.app.in_cols_text.get())
            # get the number of pixel cluster groups
            k_value = int(self.app.pixel_groups_value.get())
            # output the retrieved input parameters to the log file (useful when debugging)
            self.log.print_to_file_log("Root directory:", self.series.root_directory)
            self.log.print_to_file_log("Experiment reference:", self.series.experiment_ref)
            self.log.print_to_file_log("Tray number:", self.series.tray_number)
            self.log.print_to_file_log("Number of images:", total_num_files)
            self.log.print_to_file_log("Reference marker radius:", sticker_radius)
            self.log.print_to_file_log("Number of rows:", input_rows)
            self.log.print_to_file_log("Number of columns:", input_cols)
            self.log.print_to_file_log("Number of pixel groups:", k_value)
            self.log.print_to_file_log("Plant species:", self.app.plant_species_value.get())
            self.log.print_to_file_log("Meta source:", self.app.experimental_data_value.get())



            gc.enable()
            ##STEP 2.4: Set up a result folder to contain the processing results


            # Get the pre-processing date
            timeStamp = Analysis.LocalTimeStamp()
            curYear = timeStamp[0]
            curMonth = timeStamp[1]
            curDay = timeStamp[2]

            # Assemble the result folder
            Result_Folder = '_Processed_%d' % curDay + '-%d' % curMonth + '-%d' % curYear
            Result_Folder += "_" + str(self.series.experiment_ref) + "_Tray_" + str(self.series.tray_number)
            self.Result_Directory = self.workingRootFolder + Result_Folder
            self.print_to_log("Created output result directory:", self.Result_Directory)

            # If repeating the selection again with different criteria
            if not os.path.exists(self.Result_Directory):
                os.makedirs(self.Result_Directory)
            # Folder for processed results

            ImgDirectory = self.workingRootFolder

            # Loop based on the index of image files
            ##STEP 4.2: Create an empty csv file to save analysis results
            # chris changed ----------------
            output_csv_name = time.strftime("%Y-%m-%d") + "_LeafMeasure.csv"
            output_csv_path = os.path.join(self.Result_Directory, output_csv_name)
            # ------------------------------
            Leaf_file = open(output_csv_path, 'wb')

            result_List = [Trait.IMAGE_NAME, Trait.EXP_DATE, Trait.EXPERIMENT_REF, Trait.TRAY_NO, Trait.POT_ID, Trait.POT_X, Trait.POT_Y,
                           Trait.PROJECTED_LEAF_AREA, Trait.LEAF_PERIMETER, Trait.CANOPY_LENGTH, Trait.CANOPY_WIDTH,
                           Trait.STOCKINESS, Trait.LEAF_CANOPY_SIZE, Trait.LEAF_COMPACTNESS,
                           Trait.LARGE_LEAF_NO, Trait.LEAF_TOTAL_NO, Trait.GREENNESS, Trait.PIX2MM_RATIO]
            # Close the file object as the format of the csv is finished
            write_line = csv.writer(Leaf_file, delimiter=',', quoting=csv.QUOTE_NONE)
            write_line.writerow(result_List)
            Leaf_file.close()
            self.print_to_log("A CSV file called", output_csv_name, "is created in:", self.Result_Directory)

            ##STEP 4.3: Batch process the image series
            # Set up counters for processing stats
            pass_count, fail_count = (0, 0)

            # we need to iterate through all of the image files in the image series
            # so iterate through the list of sorted dates
            for date in dates:
                # for each date, iterate through the list of image files
                for file_to_process in self.series.date_file_dict[date]:
                    # this line is used to bridge the GUI code with the core algorithm code (used different variable names)
                    tmp_IMG_File = file_to_process
                    # calculate the progress (how many images processed vs total) and update GUI
                    self.set_progress((pass_count + fail_count)/float(total_num_files) * 100)
                    # An image used for analysis
                    # pattern match jpg or png files, make sure images are .jpg, .jpeg or .png
                    Image_FullName = tmp_IMG_File.split(Platform_Delimiter)[-1]
                    # Other parts of an image name
                    ImageName_Length = len(Image_FullName)
                    ImageDirectory = tmp_IMG_File[:(ImageName_Length * -1)]
                    ImageName = Image_FullName[:-4]
                    self.print_to_log("Start reading image: ", ImageName)

                    # ***Step_4.3.1***#
                    # Buffer the image file to the memory
                    img_tmp = io.imread(tmp_IMG_File)
                    Resize_Ratio = 1.0 / (img_tmp.shape[0] / 1024.0)  # dynamically transfer the original resolution
                    image_resized = img_as_ubyte(rescale(img_tmp.copy(), Resize_Ratio))
                    # The length (y-axis) of the image has been transferred to 1024 pixels
                    # Standardise the image size to improve processing efficiency and accuracy
                    # Set up a blank image for carrying following image objects in the process
                    Blank_Img = np.zeros((image_resized.shape[0], image_resized.shape[1]), dtype=np.uint8)
                    # Buffer only one blak image in the memory using numpy

                    # Start to process the image file
                    try:  # try...except is used to handle excpetional cases
                        # ***Step_4.3.2***# Detect Red reference points on the image
                        # The reference points mask is only generated for presentation purpose
                        Ref_Point_Array_Ref, Ref_Point_Areas, RefPoint_image = Analysis.RefPoints(image_resized, Blank_Img)

                        # ***Step_4.3.3***# Find the ratio of pixels in the detected red circular reference points
                        ## radius of the red sticker used in the experiment is 4mm
                        # ..! This value shall be input from the GUI
                        pix2mm_ratio = Analysis.convert_pixels_to_mm(Ref_Point_Areas, sticker_radius)

                        # ***Step_4.3.4***# Perspective Transformation based on reference points
                        Transformed_img, New_Ref_Points_Array = Analysis.PerspectiveTrans_2D(image_resized, Ref_Point_Array_Ref)

                        # ***Step_4.3.5***# Generate segmented pot sections, for a rows x cols tray
                        # ..! Hard code in iPython notebook, GUI can capture these parameters
                        #######################################
                        row_no = input_rows;
                        column_no = input_cols  # or 5 and 8
                        #######################################
                        Pot_Image, Pot_Segment_Refine = Analysis.PotSegmentation(Transformed_img, row_no, column_no,
                                                                        New_Ref_Points_Array)

                        # ***Step_4.3.6***# Generate segmented pot sections, for a rows x cols tray
                        # Denoising using OpenCV to smooth the leaf surface so that shadows can be presented
                        # in a linear manner. Skimage functions are too slow for this process
                        Transformed_img_Denoised = cv2.fastNlMeansDenoisingColored(Transformed_img, None, 10, 10, 5,
                                                                                   searchWindowSize=15)
                        dilated_lab_Img, leaf_LAB_img = Analysis.LAB_Img_Segmentation(Transformed_img_Denoised, pix2mm_ratio)
                        # Rescale lab image so that it can be used for leaf intensity check
                        # LAB colour spacing deals with green-related lights in uniform chromaticity diagram
                        # This function is designed for select features from the mature leaves
                        p0, p100 = np.percentile(leaf_LAB_img[:, :, 1], (0, 100))
                        leaf_LAB_img_rescale = exposure.rescale_intensity(leaf_LAB_img[:, :, 1], in_range=(p0, p100))

                        # ***Step_4.3.7***# Use Excessive greenness and kmeans to define pixel clustering
                        # Four groups were set: green leaves, soil/compost, reference points and others such as reflection
                        img_EGreen = Analysis.compute_greenness_img(Transformed_img.copy())
                        # Reduce the RGB to excessive greenness image for ML performance
                        img_ref = img_EGreen.copy()
                        # Use kmeans to cluster pixel groups
                        kmeans_img_final = Analysis.kmeans_cluster(img_ref, k_value)  # the value of k can be changed, maximum k = 10
                        kmeans_mask = kmeans_img_final > np.mean(kmeans_img_final) * 1.0125  # ONLY Five Pixel Groups, 1.025

                        # ***Step_4.3.8***# Finish finding the leaf mask
                        dilated_kmeans_Img = dilation(kmeans_mask, disk(3))
                        leaf_mask_final = np.logical_and(dilated_lab_Img, dilated_kmeans_Img)  # Leaf ROI
                        leaf_mask_ref_1 = np.logical_and(leaf_mask_final, dilation(kmeans_mask, disk(1)))
                        leaf_mask_ref = remove_small_holes(leaf_mask_ref_1, 125)  # remove small holes in the mask
                        Leaf_size_overall = int(np.sum(leaf_mask_ref) * 0.01)  # Calculate 1% of the total leaf area

                    except IOError:
                        self.print_to_log("Error: can\'t find file or read data")
                        self.log.print_to_file_log("Exception:", e)
                        fail_count += 1
                    except ValueError:
                        self.print_to_log('Non-numeric data found in the file.')
                        self.log.print_to_file_log("Exception:", e)
                        fail_count += 1
                    except ImportError:
                        self.print_to_log("Import module(s) cannot be found")
                        self.log.print_to_file_log("Exception:", e)
                        fail_count += 1
                    # Finish the section for running functions
                    except Exception as e:
                        self.print_to_log("Process failed due to analysis issues, see the log file for more information...")
                        self.print_to_log("\t\t Continuing with the next image!")
                        self.log.print_to_file_log("Exception:", e)
                        fail_count += 1
                        gc.collect()
                        continue


                        ##STEP 4.4: Quantify traits from the images
                    try:
                        # use the locking mechanism here as we are using matplotlib
                        with Analysis.lock:
                            # Start to extract the quantification
                            if leaf_mask_ref.mean() > 0:  # The leaf mask contains information
                                # ***Step_4.4.1***# Prepare leaf measurements
                                ## Step_4.4.1.1 Prepare lists for carrying measurements
                                pot_array = []  # collect the pot location
                                label_array = []  # Collect labels for objects that have been selected
                                no_Count = 0  # used for counting the iteration

                                ## Step_4.4.1.2 Prepare images for carrying processed images
                                Final_Leaf_img = Blank_Img.copy()
                                Final_Leaf_Skeleton = Blank_Img.copy()
                                Final_Tip_img = Blank_Img.copy()
                                Leaf_hull_Final = Blank_Img.copy()
                                Leaf_hull_Outline_Final = Blank_Img.copy()
                                (Leaf_Region_Width, Leaf_Region_Length) = (0, 0)

                                # Prepare images for carry leaf tip detection
                                blank_h, blank_w = Blank_Img.shape[:2]
                                Final_Leaf_Sweep_img = np.zeros((blank_h, blank_w, 3), np.uint8)
                                hull_region = None

                                ## Step_4.4.1.3 For segmenting pots, use ndimage to extract pot level features
                                Labelled_Pot_Img, num_features = ndimage.measurements.label(Pot_Segment_Refine)
                                ## Step_4.4.1.4 Prepare the processed image
                                fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(25, 15))
                                # The background of the processed image
                                ax1.imshow(Transformed_img)
                                ax2.imshow(Transformed_img)

                                # ***Step_4.4.2***# Start to generate leaf measurements
                                # Go through every recognised pot
                                for pot_region in regionprops(Labelled_Pot_Img, cache=True):

                                    ## Step_4.4.2.1 skip too small or too large objects
                                    if (pot_region.area > (0.925 * Transformed_img.size / (row_no * column_no))) or (
                                        pot_region.area <
                                        (0.075 * Transformed_img.size / (row_no * column_no))):
                                        continue  # Ignore the pot as to big/small***")

                                    ## Step_4.4.2.2 Prepare temporary pot image base
                                    pot_tmp_img = Blank_Img.copy()
                                    pot_coord = pot_region.coords.astype(int)
                                    # coordinates based on labelled outlines
                                    pot_tmp_img[pot_coord[:, 0], pot_coord[:, 1]] = 1
                                    # expand the pot region to include leaves growing outside the pot
                                    kernel = np.ones((18, 18), np.uint8)  # Use OpenCV to increase the dilation performance
                                    pot_tmp_img_Ref = cv2.dilate(pot_tmp_img, kernel, iterations=1)

                                    ## Step_4.4.2.3 Generate temporary leaf image base
                                    Overlap_Leaf_tmp = Blank_Img.copy()
                                    # Use leaf_mask_final to leaf level detection, use leaf_mask_ref for leaf area measurement
                                    Overlap_Leaf_tmp = np.logical_and(leaf_mask_ref, pot_tmp_img_Ref)
                                    Overlap_Leaf_tmp_Ref = Blank_Img.copy()
                                    Leaf_Area_tmp = 0.0

                                    ## Step_4.4.2.4 Start to trim leaves in every pot
                                    # Normalised LAB is used as the intensity template for feature seleciton
                                    Labelled_leaf_Img = measure.label(Overlap_Leaf_tmp)
                                    for leaf_region in regionprops(Labelled_leaf_Img, intensity_image=leaf_LAB_img_rescale,
                                                                   cache=True):
                                        # Remove stones and other non-green small objects
                                        if leaf_region.area <= 25:  # the resolution is fixed
                                            continue
                                        Intensity_Diff = leaf_region.max_intensity - leaf_region.mean_intensity
                                        if Intensity_Diff < 0.15 and leaf_region.area < 225:  # Deviation from the scaled difference,
                                            continue
                                            # After rescaling, the most green part has a value -1, 1SD round to 65% -> 35%
                                        if leaf_region.mean_intensity > -0.3 or leaf_region.min_intensity > -0.6:
                                            # the LAB has been rescaled between -1.0 and 0 for presenting greenness
                                            continue
                                        # The position of the leaf object
                                        if leaf_region.centroid[0] < pot_region.bbox[0] * 1.025 or leaf_region.centroid[0] > \
                                                        pot_region.bbox[2] * 0.975:
                                            continue  # Column, y-axis
                                        if leaf_region.centroid[1] < pot_region.bbox[1] * 1.025 or leaf_region.centroid[1] > \
                                                        pot_region.bbox[3] * 0.975:
                                            continue  # Row, x-axis
                                        leaf_contrast = Intensity_Diff * 1.0 / (
                                        leaf_region.max_intensity + leaf_region.min_intensity)
                                        # Remove based on contrast levels and intensity values
                                        if (
                                                leaf_region.mean_intensity > -0.525 or leaf_contrast > -0.185) and leaf_region.area < pix2mm_ratio * pix2mm_ratio * 10:
                                            # very small leaves
                                            continue
                                        if leaf_contrast > -0.125 and leaf_region.area < pix2mm_ratio * pix2mm_ratio * 30:  # 30 mm^2
                                            # middle size leaves
                                            continue
                                        if leaf_contrast > -0.195 and leaf_region.mean_intensity > -0.525 and leaf_region.area >= pix2mm_ratio * pix2mm_ratio * 30:
                                            # large leaves
                                            continue
                                            # print (no_Count+1), ': ', leaf_contrast, " - ", Intensity_Diff, " - ", leaf_region.area
                                        # print leaf_region.mean_intensity, "_", Leaf_size_overall*0.15, "_"
                                        leaf_coord = leaf_region.coords.astype(int)
                                        Overlap_Leaf_tmp_Ref[leaf_coord[:, 0], leaf_coord[:, 1]] = 1
                                        # Final refinement
                                    Final_Leaf_Clean = remove_small_objects(dilation(leaf_mask_ref, disk(3)),
                                                                            Leaf_size_overall * 0.15)
                                    Overlap_Leaf_tmp_Ref = np.logical_and(Final_Leaf_Clean, Overlap_Leaf_tmp_Ref)
                                    Overlap_Leaf_tmp_Ref = remove_small_objects(Overlap_Leaf_tmp_Ref, Leaf_size_overall * 0.32)
                                    Final_Leaf_img = np.logical_or(Overlap_Leaf_tmp_Ref, Final_Leaf_img)
                                    Leaf_Area_tmp = np.sum(Overlap_Leaf_tmp_Ref) * 1.0 / (pix2mm_ratio * pix2mm_ratio)
                                    # print "Finish processing leaf surface in pot %d!" %(no_Count + 1)
                                    ## Finish trimming leaves

                                    ## Step_4.4.2.5 Label pots
                                    no_Count = no_Count + 1
                                    ax1.text(pot_region.bbox[1] + 25, pot_region.bbox[0] + 25, '%d' % (no_Count), ha='center',
                                             va='bottom', color='lime', size=16)  # altered from +25 and +25
                                    ax2.text(pot_region.bbox[1] + 25, pot_region.bbox[0] + 25, '%d' % (no_Count), ha='center',
                                             va='bottom', color='lime', size=16)  # altered from +25 and +25

                                    ## Step_4.4.2.6 Extract Convex Hull for leaf objects
                                    # Add convex hull to the leaf objects
                                    Leaf_hull_tmp = Blank_Img.copy()
                                    Leaf_hull_Outline = Blank_Img.copy()
                                    if np.sum(Overlap_Leaf_tmp_Ref) > 0:  # Plants found in a pot
                                        Leaf_hull_tmp = convex_hull_image(Overlap_Leaf_tmp_Ref)
                                        Leaf_hull_Outline = np.logical_xor(Leaf_hull_tmp, erosion(Leaf_hull_tmp, disk(2)))
                                        Leaf_hull_Outline_Final = np.logical_or(Leaf_hull_Outline_Final, Leaf_hull_Outline)
                                        Leaf_hull_Final = np.logical_or(Leaf_hull_Final, Leaf_hull_tmp)
                                    # Leaf canopy
                                    Hull_area = np.sum(Leaf_hull_tmp)
                                    # generate a circle to present the centroid of convex hull
                                    Leaf_Centroid_TMP = Blank_Img.copy()
                                    if (np.sum(Leaf_hull_tmp) > 0):  # Plant canopy found in a pot
                                        # Measure the region properties of the leaf canopy
                                        label_hull_img = label(Leaf_hull_tmp, connectivity=Leaf_hull_tmp.ndim)
                                        hull_region = measure.regionprops(label_hull_img, intensity_image=leaf_LAB_img_rescale,
                                                                          cache=True)
                                        # Measure the full length and width of the canopy
                                        Leaf_Region_Width = round(hull_region[0].minor_axis_length, 0)
                                        Leaf_Region_Length = round(hull_region[0].major_axis_length, 0)
                                        Leaf_Region_Radius = (Leaf_Region_Length * 1.0) / 2
                                        # to remove early small leaves, as the calculation is dynamic
                                        Leaf_Region_Hull = (2.75 * pix2mm_ratio) * sticker_radius  # diameter is 4 mm
                                        rr, cc = circle(hull_region[0].centroid[0], hull_region[0].centroid[1],
                                                        Leaf_Region_Hull)
                                        Leaf_Centroid_TMP[rr, cc] = 1  # A filled circle is generated
                                    else:  # an empty canopy region
                                        Leaf_Region_Width = 0
                                        Leaf_Region_Length = 0
                                        Leaf_Region_Radius = 0
                                    # Finish detecting leaf canopy
                                    # Overlap_Leaf_tmp_Ref
                                    # print "Finish processing leaf canopy in pot %d!" %(no_Count)

                                    ## Step_4.4.2.7 Extract skeletons based on the detected leaf regions
                                    Sk_tmp_img = Blank_Img.copy()
                                    Large_Leaf_tmp = Blank_Img.copy()
                                    # Process the image before skeletonisation
                                    total_size_leaf = np.sum(Overlap_Leaf_tmp_Ref)
                                    # Remove 5% small objects, if the trim function still kept some shining soil objects
                                    Overlap_Leaf_Sk = remove_small_objects(Overlap_Leaf_tmp_Ref, int(total_size_leaf * 0.05))
                                    # Improve the skeleton detection by filling holes and dilating the leaf masks
                                    Overlap_Leaf_Sk_Ref = ndimage.binary_fill_holes(dilation(Overlap_Leaf_Sk, disk(3))).astype(
                                        int)
                                    # Extract the skeleton
                                    Sk_tmp_img = skeletonize(Overlap_Leaf_Sk_Ref)
                                    # Find the end points of the skeleton
                                    Leaf_tip_array = Analysis.find_end_points(dilation(Sk_tmp_img, disk(2)))
                                    Leaf_tmp_tip = Blank_Img.copy()
                                    Leaf_tmp_tip[Leaf_tip_array[:, 1], Leaf_tip_array[:, 0]] = 1
                                    # Overlap with the centroid region to remove small leaves
                                    Large_Leaf_tmp_tip = np.logical_xor(np.logical_and(Leaf_tmp_tip, Leaf_Centroid_TMP),
                                                                        Leaf_tmp_tip)
                                    # Prepare for the measurement and output processed picture
                                    Sk_tmp_img = dilation(Sk_tmp_img, disk(1))
                                    Large_Leaf_tmp_tip = dilation(Large_Leaf_tmp_tip, disk(5))
                                    # Start to trim leaves in every pot
                                    Labelled_leaf_Img = measure.label(Large_Leaf_tmp_tip)
                                    tip_Count = 1
                                    for leaf_tip_counter in regionprops(Labelled_leaf_Img, cache=True):
                                        # add text onto the leaf tips
                                        ax2.text(leaf_tip_counter.bbox[1], leaf_tip_counter.bbox[0], '%d' % (tip_Count),
                                                 ha='center',
                                                 va='center', color='yellow', fontweight='bold', size=12)
                                        tip_Count = tip_Count + 1
                                    # Finalise the skeleton graph
                                    Final_Leaf_Skeleton = np.logical_or(Sk_tmp_img, Final_Leaf_Skeleton)
                                    Final_Tip_img = np.logical_or(Large_Leaf_tmp_tip, Final_Tip_img)
                                    # print "Finish processing leaf skeleton in pot %d!" %(no_Count)
                                    ## Finish generating skeleton for leaf measurements

                                    ## Step_4.4.2.8 Leaf detection based on waveform sweeping method
                                    if hull_region is not None:
                                        Final_Leaf_Sweep_img, total_leaf_number = find_leaves(Overlap_Leaf_tmp_Ref,
                                                                                              Final_Leaf_Sweep_img,
                                                                                              hull_region[0].centroid[1],
                                                                                              hull_region[0].centroid[0],
                                                                                              pot_region)
                                    else:
                                        total_leaf_number = 0  # If not leaf has found in the pot
                                    ## Finish generating leaf tip measurements

                                    # ***Step_4.4.3***# Append the quantification to the csv file
                                    # chris changed --------------
                                    Leaf_file = open(output_csv_path, 'a')

                                    # get the series information from the series instance
                                    Imaging_Date = date
                                    Genotype_Treatment_tmp = self.series.experiment_ref
                                    Tray_No_tmp = self.series.tray_number

                                    #Leaf_file = open(Result_Directory + "/" + time.strftime("%Y-%m-%d") + "_LeafMeasure.csv",
                                    #                 'a')
                                    #if ImageName.count('_') < 2:  # At least two underscores need to be found
                                    #    # wrong format of the image name
                                    #    Imaging_Date = ''
                                    #    Genotype_Treatment_tmp = ''
                                    #    Tray_No_tmp = ''
                                    #elif ImageName.count('_') >= 4:
                                    #    # Proper annotation
                                    #    Imaging_Date = ImageName.split("_")[0]
                                    #    Genotype_Treatment_tmp = ImageName.split("_")[1] + "_" + ImageName.split("_")[2]
                                    #    Tray_No_tmp = ImageName.split("_")[4]
                                    #else:
                                    #    # Format with two-three '_' in the image name
                                    #    Imaging_Date = ImageName.split("_")[0]
                                    #    Genotype_Treatment_tmp = ''
                                    #    for elem in ImageName.split("_")[1:-2]:
                                    #        Genotype_Treatment_tmp = Genotype_Treatment_tmp + elem
                                    #    Tray_No_tmp = ImageName.split("_")[-1]
                                    #--------------------------------------------------------

                                    if Leaf_Area_tmp > 100 and np.sum(Leaf_tmp_tip) > total_leaf_number:
                                        # When projected leaf areas are small, the skeleton method will over count leaf numbers
                                        # 100 means 100 mm^2
                                        total_leaf_number = np.sum(Leaf_tmp_tip)
                                        # Choose the larger value calculated from the skeleton method and th tip detection method

                                    if np.sum(Overlap_Leaf_tmp_Ref) < Leaf_size_overall * 0.2 and Leaf_Area_tmp > 1000:
                                        # only applying when the leaf area is over 1000 mm^2
                                        Leaf_Region_Radius = 0  # not counted for trait analysis

                                    # The string to be appended
                                    if Leaf_Region_Radius > 0:
                                        Leaf_Stockiness = 4.0 * math.pi * (np.sum(Overlap_Leaf_tmp_Ref)) / math.pow(
                                            (2 * math.pi * Leaf_Region_Radius), 2)
                                        result_List = [ImageName, Imaging_Date, Genotype_Treatment_tmp, Tray_No_tmp, no_Count,
                                                       int(pot_region.centroid[1]), int(pot_region.centroid[0]),
                                                       round(np.sum(Overlap_Leaf_tmp_Ref) * 1.0 / (pix2mm_ratio * pix2mm_ratio),
                                                             1),
                                                       round(measure.perimeter(Overlap_Leaf_tmp_Ref,
                                                                               neighbourhood=4) / pix2mm_ratio, 1),
                                                       round(Leaf_Region_Length * 1.0 / pix2mm_ratio, 1),
                                                       round(Leaf_Region_Width * 1.0 / pix2mm_ratio, 1),
                                                       round(Leaf_Stockiness * 100, 1),
                                                       round(Hull_area * 1.0 / (pix2mm_ratio * pix2mm_ratio), 1),
                                                       round(np.sum(Overlap_Leaf_tmp_Ref) * 100.0 / Hull_area, 1),
                                                       (tip_Count - 1), total_leaf_number,
                                                       round(abs(hull_region[0].mean_intensity * 1.125) * 255.0, 1),
                                                       round(pix2mm_ratio, 1)
                                                       ]
                                    else:  # No leaf found in the pot
                                        result_List = [ImageName, Imaging_Date, Genotype_Treatment_tmp, Tray_No_tmp, no_Count,
                                                       int(pot_region.centroid[1]), int(pot_region.centroid[0]),
                                                       0.0, 0.0, 0.0, 0.0, 0.000, 0.0, 0.0, 0, 0,
                                                       0.0, round(pix2mm_ratio, 1)
                                                       ]

                                    write_line = csv.writer(Leaf_file, delimiter=',', quoting=csv.QUOTE_NONE)
                                    write_line.writerow(result_List)
                                    Leaf_file.close()
                                    self.print_to_log("\tFinish exporting leaf measurements in pot", str(no_Count))
                                    # Finish exporting to the csv file for leaf measurements



                                    ##STEP 5: Export processed images
                                ## Step_5.1 Generate contour images for leaf outlines and pot ID
                                contours_image = np.logical_and(Final_Leaf_Clean, dilation(leaf_mask_ref, disk(0)))
                                tip_contours = find_contours(Final_Tip_img, 0.9)
                                for n3, contour3 in enumerate(tip_contours):
                                    ax2.plot(contour3[:, 1], contour3[:, 0], linewidth=9, color='red', fillstyle='full')
                                contours2 = find_contours(Leaf_hull_Outline_Final, 0.9)
                                for n2, contour2 in enumerate(contours2):
                                    ax1.plot(contour2[:, 1], contour2[:, 0], linewidth=2, color='yellow')

                                ax2.set_title('Large Leaf Detection', fontsize=12)
                                ax2.axis([-25, Final_Leaf_img.shape[1] + 25, Final_Leaf_img.shape[0] + 25, -25])
                                # Prepare the processed figure
                                ax1.set_title('Leaf Canopy', fontsize=12)
                                ax1.axis([-25, Final_Leaf_img.shape[1] + 25, Final_Leaf_img.shape[0] + 25, -25])
                                plt.tight_layout()
                                # The following can handle different platforms
                                fig.savefig(os.path.join(self.Result_Directory + Platform_Delimiter,
                                                         (ImageName + '_05_Leaf_Analysis.jpg')), bbox_inches='tight')
                                plt.close(fig)  # Close the figure

                                ## Step_5.2 Output processed images - calibration
                                # Produce processed images
                                fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(24, 12))
                                ax1.set_title('Original Image', fontsize=12)
                                ax1.axis([-25, Final_Leaf_img.shape[1] + 25, Final_Leaf_img.shape[0] + 25, -25])
                                ax1.imshow(image_resized)  # The background of the processed image
                                ax2.set_title('Transformed Image', fontsize=12)
                                ax2.imshow(Transformed_img_Denoised)  # Transformed image
                                # Prepare the processed figure
                                plt.axis([-25, Final_Leaf_img.shape[1] + 25, Final_Leaf_img.shape[0] + 25, -25])
                                plt.tight_layout()
                                # The following can handle different platforms
                                fig.savefig(os.path.join(self.Result_Directory + Platform_Delimiter,
                                                         (ImageName + '_01_Image_Calibration.jpg')), bbox_inches='tight')
                                plt.close(fig)  # Close the figure

                                ## Step_5.3 Output processed images - ML clustering
                                # Produce processed images
                                fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(24, 12))
                                ax1.set_title('LAB Colour Space with Green-Red Colour Opponents', fontsize=12)
                                ax1.imshow(leaf_LAB_img[:, :, 2])  # The non-linear LAB image
                                ax2.set_title('Colour Clustering Groups Using Kmeans', fontsize=12)
                                ax2.imshow(kmeans_mask, cmap='rainbow')
                                # Prepare the processed figure
                                ax1.axis([-25, Final_Leaf_img.shape[1] + 25, Final_Leaf_img.shape[0] + 25, -25])
                                ax2.axis([-25, Final_Leaf_img.shape[1] + 25, Final_Leaf_img.shape[0] + 25, -25])
                                plt.tight_layout()
                                # The following can handle different platforms
                                fig.savefig(os.path.join(self.Result_Directory + Platform_Delimiter,
                                                         (ImageName + '_02_Colour_Clustering.jpg')), bbox_inches='tight')
                                plt.close(fig)  # Close the figure

                                ## Step_5.4 Output leaf regions
                                # Produce processed images
                                fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(24, 12))
                                ax1.imshow(np.logical_and(np.invert(Final_Leaf_img), Pot_Segment_Refine), cmap='gray')
                                # Prepare the processed figure
                                ax1.set_title('Refined Projected Leaf Regions in pots', fontsize=12)
                                ax1.axis([-25, Final_Leaf_img.shape[1] + 25, Final_Leaf_img.shape[0] + 25, -25])
                                ax2.imshow(np.logical_and(Final_Leaf_img, leaf_mask_ref_1), cmap='gray')
                                ax2.set_title('Selected Leaves for Trait Analysis', fontsize=12)
                                ax2.axis([-25, Final_Leaf_img.shape[1] + 25, Final_Leaf_img.shape[0] + 25, -25])
                                plt.tight_layout()
                                # The following can handle different platforms
                                fig.savefig(os.path.join(self.Result_Directory + Platform_Delimiter,
                                                         (ImageName + '_04_Refined_Leaf_ROI.jpg')), bbox_inches='tight')
                                plt.close(fig)  # Close the figure

                                ## Step_5.4 Output processed images - Leaf extraction
                                # Produce processed images
                                fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(24, 12))
                                # The background of the processed image
                                ax2.imshow(Transformed_img_Denoised)  # The denoised image

                                contours1 = find_contours(erosion(remove_small_holes(Final_Leaf_Clean, min_size=525), disk(3)),
                                                          0.9)
                                for n1, contour1 in enumerate(contours1):
                                    ax2.plot(contour1[:, 1], contour1[:, 0], linewidth=2, color='yellow')
                                ax2.set_title('Outline of Leaves of Interest', fontsize=12)
                                # Crop leaf for presentation
                                Leaf_Crop = Transformed_img_Denoised.copy()
                                for layer in range(Leaf_Crop.shape[-1]):
                                    Leaf_Crop[np.where(np.invert(erosion(Final_Leaf_Clean, disk(3))))] = 1
                                ax1.set_title('Leaf Extraction using Lab and Kmeans', fontsize=12)
                                ax1.imshow(Leaf_Crop)
                                # Prepare the processed figure
                                ax1.axis([-25, Final_Leaf_img.shape[1] + 25, Final_Leaf_img.shape[0] + 25, -25])
                                ax2.axis([-25, Final_Leaf_img.shape[1] + 25, Final_Leaf_img.shape[0] + 25, -25])
                                plt.tight_layout()
                                # The following can handle different platforms
                                fig.savefig(os.path.join(self.Result_Directory + Platform_Delimiter,
                                                         (ImageName + '_03_Leaf_Extraction.jpg')), bbox_inches='tight')
                                plt.close(fig)  # Close the figure

                                ## Step_5.5 Output leaf skeletons
                                # Produce processed images
                                fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(24, 12))
                                Skel_img = Final_Leaf_Skeleton.copy()
                                # Add end points to the leaf
                                Tip_img = Final_Tip_img.copy()
                                Skel_Overall = np.logical_or(Skel_img, Tip_img)
                                Skel_Overall = np.ma.masked_where(Skel_Overall == 1, Transformed_img[:, :, 1])
                                # Prepare the processed figure
                                ax1.imshow(np.logical_xor(Pot_Segment_Refine, np.invert(Skel_Overall)), cmap='gray')
                                ax1.set_title('Leaf Skeleton', fontsize=12)
                                ax1.axis([-25, Final_Leaf_img.shape[1] + 25, Final_Leaf_img.shape[0] + 25, -25])
                                ax2.imshow(Final_Leaf_Sweep_img)
                                ax2.set_title('Leaf Tip Detection', fontsize=12)
                                ax2.axis([-25, Final_Leaf_img.shape[1] + 25, Final_Leaf_img.shape[0] + 25, -25])
                                plt.tight_layout()
                                # The following can handle different platforms
                                fig.savefig(os.path.join(self.Result_Directory + Platform_Delimiter,
                                                         (ImageName + '_06_Leaf_Detection.jpg')), bbox_inches='tight')
                                plt.close(fig)  # Close the figure

                                ##STEP 6: Processing stats and garbage collection
                                pass_count += 1
                                gc.collect()
                                self.print_to_log("Successfully processed: ", ImageName)

                            # If an image does not contain any leaf information
                            else:
                                self.print_to_log("Error: can\'t find any leaf information!")
                                fail_count += 1
                    except Exception as e:
                        # In case we failed in the process
                        self.print_to_log("Process failed due to image analysis issues... Continuing with the next image!")
                        self.log.print_to_file_log("Exception:", e)
                        fail_count += 1
                        gc.collect()
                        # Clear memory
                        del img_tmp, image_resized, Transformed_img, Transformed_img_Denoised
                        del leaf_LAB_img, leaf_LAB_img_rescale, img_EGreen
                        del img_ref, kmeans_img_final, kmeans_mask
                        continue

                        # One image has been processed
                    gc.collect()
                    # Clear memory for the batch processing
                    del img_tmp, image_resized, Transformed_img, Transformed_img_Denoised
                    del leaf_LAB_img, leaf_LAB_img_rescale, img_EGreen
                    del img_ref, kmeans_img_final, kmeans_mask
                    self.log.print_to_file_log("garbage collected")
                    # Plot results of the processed image

            # chris added -----------
            attribute_button_text = ["-"] * len(self.attributes_to_plot)

            # if this is truly an image series (i.e. there is date information)
            if date is not "-":
                # we want to produce plots for each plot attribute
                # use the locking mechanism to ensure that no other thread is using matplotlib functionality whilse we are using it here
                with Analysis.lock:
                    num_pots = input_rows * input_cols
                    # iterate through the attributes to plot (these are the plots we will show in the GUI in the results table)
                    for attribute in self.attributes_to_plot:
                        # some of the attribute names include characters that we shouldn't or can't use in filename
                        # therefore, we replace these characters with underscores
                        attribute_without_spaces = attribute.replace(" ", "_").replace("^", "_")
                        # create suitable plot output filename
                        output_plot_filename = os.path.join(self.Result_Directory,
                                                            str(attribute_without_spaces + ".png"))
                        # generate the plot for this attribute for this image series
                        img = plot_generator.generate_plot(output_csv_path,
                                                           self.series.experiment_ref, self.series.tray_number,
                                                           attribute)
                        # before we save, we need to convert from RGB to BGR format when using OpenCV functionality
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        # save the plot to file
                        cv2.imwrite(output_plot_filename, img)

                # in the GUI we also display the minimum and maximum mean values for each plotted attribute
                # this is the min and max of the mean value from each image analysed

                # counter storing the number of attributes we have got min and max mean values for
                attribute_counter = 0
                # iterate through the list of attributes to plot
                for attribute in self.attributes_to_plot:
                    # get the min and max mean value
                    min_value, max_value = Analysis.get_trait_min_max_means(output_csv_path, attribute)
                    # the total leaf number should be represented as integer, so round to int if we are processing this attribute
                    if attribute == Trait.LEAF_TOTAL_NO:
                        # round the min
                        min_value = round(min_value)
                        # round the max
                        max_value = round(max_value)
                    # otherwise, just round to 1 decimal place (for display purposes)
                    else:
                        # round the min
                        min_value = round(min_value, 1)
                        # round the max
                        max_value = round(max_value, 1)
                    # get the text string to display in the results table, which will be min to the max
                    text = str(min_value) + " - " + str(max_value)
                    # update the button text in the list of button text values
                    attribute_button_text[attribute_counter] = text
                    # increment the counter to move onto the next attribute
                    attribute_counter += 1

            # get the output result directory
            relative_result_directory = self.Result_Directory[len(self.series.root_directory):]

            # generate a list of row values (it will be at least series id and out directory)
            # then a list of min/max mean values for the attributes to display in GUI
            result_row_values = [(str(self.series.id)), (relative_result_directory)]
            # iterate through the list of attributes to plot
            for i in range(len(self.attributes_to_plot)):
                # append the string to the list of values to show in table row
                result_row_values.append(attribute_button_text[i])
            # update the GUI accordingly, as we are not in the main thread, push functionality to update GUI to GUI queue
            GuiQueue.gui_queue.put(partial(self.app.results_table.insert, "", END, values=result_row_values))
            # analysis complete to update status column in series processing table
            self.update_tree("Complete")
            # ----------------------

            self.print_to_log("\nFinish the analysis - see results in: ", self.Result_Directory)
            self.print_to_log("Processing Summary:")
            self.print_to_log("\tThere were %d images successfully processed, with %d failures" % (pass_count, fail_count))

        except Exception as e:
            self.print_to_log("There was an error processing the result. Analysis terminated.")
            self.log.print_to_file_log("Exception:", e)
            self.update_tree("Error: Analysis Terminated")
        finally:
            # detach the log (this means the the log associated with this thread/process is flagged as being available for output information for another process)
            self.log.detach()


    # this function returns the minimum and maximum mean value for the specified attribute for an image in the image series
    # NOTE: the trait must be able to be cast to floating point representation
    # @param csv_filename: the csv filename to read the data from
    # @param trait_id: the name of the column in the csv file for the attribute
    @staticmethod
    def get_trait_min_max_means(csv_filename, trait_id):
        # load the csv file as a list of dictionaries for each row
        row_data = plot_generator.load_csv(csv_filename)
        # store a dictionary where key is image name and value is the is a list of trait values
        filename_trait_dict = dict()
        # iterate through the rows in the loaded csv file data structure
        for row in row_data:
            # get the key, which is the image name
            key = row[Trait.IMAGE_NAME]
            # get the trait value (will convert to floating point representation)
            trait_value = float(row[trait_id])
            # is the key already in the dictionary
            if key in filename_trait_dict.keys():
                # if so, then just add the trait value to the list of trait values for this image
                filename_trait_dict[key].append(trait_value)
            # otherwise add new dictionary entry with the key and create a new list of trait values, with this trait value
            else:
                filename_trait_dict[key] = [trait_value]

        # now we have a dictionary where each key represents a different file and its value is a list of all trait values for that image
        # we want to return the min and max mean for any image
        # store the global min and max value
        global_min_trait = None
        global_max_trait = None
        # iterate through the dictionary (images)
        for key in filename_trait_dict.keys():
            # get the mean of the trait values in the image
            mean_trait = np.mean(filename_trait_dict[key])
            # is this mean the new global minimum?
            if global_min_trait is None or mean_trait < global_min_trait:
                # if so, update the global minimum
                global_min_trait = mean_trait
            # is this mean the new global maximum?
            if global_max_trait is None or mean_trait > global_max_trait:
                # if so, update the global maximum
                global_max_trait = mean_trait
        # return the min and max mean trait values
        return global_min_trait, global_max_trait


    # Function_2
    # Locate the red reference points
    @staticmethod
    def RefPoints(ref_image, blank_image):
        """Locate red reference points"""
        # STEP 1: Locate red pixels
        # R >= 150, B <= 100, and G <= 100, due to colour distortion
        Ref_Points_Red = ref_image[:, :, 0] > 125
        # Use blue channel to remove the white pixels
        Ref_Points_Blue = ref_image[:, :, 2] < 225
        Ref_Points_Color_Selection_1 = np.logical_and(Ref_Points_Red, Ref_Points_Blue)
        # Add second approach based on the difference between red and green channles
        Ref_Points_Color_Selection_2 = np.array(ref_image[:, :, 0], dtype='int') - np.array(ref_image[:, :, 1],
                                                                                            dtype="int") > 50

        # STEP 2: Extract Red ref points from the previous mask
        Ref_Points_Refined = np.logical_and(Ref_Points_Color_Selection_1, Ref_Points_Color_Selection_2)
        Ref_Point = remove_small_objects(Ref_Points_Refined,
                                         125)  # get rid of small pixels, as the resolution is fixed
        Fill_Ref_Points = ndimage.binary_fill_holes(Ref_Point)

        ## Create a list for the areas of the detected red circular reference points
        Ref_Point_Areas = []
        Ref_Point_Array = []
        # Start to Label the Red reference points
        Labelled_Ref_Point = label(Fill_Ref_Points, connectivity=1)
        processed_Ref_Img = blank_image.copy()
        Obj_counter = 0
        maxLength = Ref_Points_Refined.shape[1] / 25
        minLength = Ref_Points_Refined.shape[1] / 55
        # Go through every red reference point objects
        for region in regionprops(Labelled_Ref_Point):
            Ref_Tmp_Img = blank_image.copy()
            # Purely based on morphology features
            if region.area < math.pi * (minLength / 2) ** 2 or region.area > math.pi * (maxLength / 2) ** 2:
                # set area lower and upper bounds
                continue
            if region.solidity < 0.75:  # Compare ref points with their convex areas.
                continue
            if region.eccentricity > 0.75:  # These ref points are not totally round.
                continue
            # get 2D coordinates of the ref points
            Obj_counter = Obj_counter + 1
            # print "ID: ", str(Obj_counter), 'Centroid: ', int(region.centroid[0]), int(region.centroid[1])
            pot_row_max = [region.bbox[2], ref_image.shape[0]]
            # column first (y) and then row (x)
            Ref_Point_Array.append([int(region.centroid[1]), int(region.centroid[0])])
            Ref_Point_Areas.append(region.area)
            Ref_coord = region.coords.astype(int)
            Ref_Tmp_Img[Ref_coord[:, 0], Ref_coord[:, 1]] = 1
            # Generate the final mask
            processed_Ref_Img = np.logical_or(processed_Ref_Img, Ref_Tmp_Img)

        # Set up empty arraies for boundary
        Upper_Left_Point = []
        Lower_Left_Point = []
        Upper_Right_Point = []
        Lower_Right_Point = []

        # Sort the position of red reference points on the image
        for x_coor, y_coor in Ref_Point_Array:
            if x_coor < ref_image.shape[1] * 0.75 and y_coor < ref_image.shape[0] * 0.75:
                Upper_Left_Point.append([x_coor, y_coor])
            if x_coor < ref_image.shape[1] * 0.75 and y_coor > ref_image.shape[0] * 0.75:
                Lower_Left_Point.append([x_coor, y_coor])
            if x_coor > ref_image.shape[1] * 0.75 and y_coor < ref_image.shape[0] * 0.75:
                Upper_Right_Point.append([x_coor, y_coor])
            if x_coor > ref_image.shape[1] * 0.75 and y_coor > ref_image.shape[0] * 0.75:
                Lower_Right_Point.append([x_coor, y_coor])

        Ref_Point_Array_Ref = []
        Ref_Point_Array_Ref.append(Upper_Left_Point[0])
        Ref_Point_Array_Ref.append(Upper_Right_Point[0])
        Ref_Point_Array_Ref.append(Lower_Left_Point[0])
        Ref_Point_Array_Ref.append(Lower_Right_Point[0])

        # Return the Red reference points' coordindates
        return Ref_Point_Array_Ref, Ref_Point_Areas, processed_Ref_Img

    @staticmethod
    def RefPoints_Wheat(ref_image, blank_image):
        """Locate red reference points"""
        # STEP 1: Locate red pixels
        # R >= 150, B <= 100, and G <= 100, due to colour distortion
        Ref_Points_Red = ref_image[:, :, 0] > 125
        # Use blue channel to remove the white pixels
        Ref_Points_Blue = ref_image[:, :, 2] < 225
        Ref_Points_Color_Selection_1 = np.logical_and(Ref_Points_Red, Ref_Points_Blue)
        # Add second approach based on the difference between red and green channles
        Ref_Points_Color_Selection_2 = np.array(ref_image[:, :, 0], dtype='int') - np.array(ref_image[:, :, 1],
                                                                                            dtype="int") > 50

        # STEP 2: Extract Red ref points from the previous mask
        Ref_Points_Refined = np.logical_and(Ref_Points_Color_Selection_1, Ref_Points_Color_Selection_2)
        Ref_Point = remove_small_objects(Ref_Points_Refined, 25)  # get rid of small pixels, as the resolution is fixed
        Fill_Ref_Points = ndimage.binary_fill_holes(Ref_Point)

        ## Create a list for the areas of the detected red circular reference points
        Ref_Point_Areas = []
        Ref_Point_Array = []
        # Start to Label the Red reference points
        Labelled_Ref_Point = label(Fill_Ref_Points, connectivity=1)
        processed_Ref_Img = blank_image.copy()
        Obj_counter = 0
        maxLength = Ref_Points_Refined.shape[1] / 25
        minLength = Ref_Points_Refined.shape[1] / 55
        # Go through every red reference point objects
        for region in regionprops(Labelled_Ref_Point):
            Ref_Tmp_Img = blank_image.copy()
            if region.area < 125:
                continue
            # Purely based on morphology features
            if region.solidity < 0.75:  # Compare ref points with their convex areas.
                continue
            if region.eccentricity > 0.75:  # These ref points are not totally round.
                continue
            # get 2D coordinates of the ref points
            Obj_counter = Obj_counter + 1
            # print "ID: ", str(Obj_counter), 'Centroid: ', int(region.centroid[0]), int(region.centroid[1])
            pot_row_max = [region.bbox[2], ref_image.shape[0]]
            # column first (y) and then row (x)
            Ref_Point_Array.append([int(region.centroid[1]), int(region.centroid[0])])
            Ref_Point_Areas.append(region.area)
            Ref_coord = region.coords.astype(int)
            Ref_Tmp_Img[Ref_coord[:, 0], Ref_coord[:, 1]] = 1
            # Generate the final mask
            processed_Ref_Img = np.logical_or(processed_Ref_Img, Ref_Tmp_Img)

        # Set up empty arraies for boundary
        Upper_Left_Point = []
        Lower_Left_Point = []
        Upper_Right_Point = []
        Lower_Right_Point = []

        centre_x, centre_y = ([], [])
        centre_x = np.sum([i[0] for i in Ref_Point_Array]) / 4
        centre_y = np.sum([i[1] for i in Ref_Point_Array]) / 4
        # Sort the position of red reference points on the image
        for x_coor, y_coor in Ref_Point_Array:
            if x_coor < centre_x and y_coor < centre_y:
                Upper_Left_Point.append([x_coor, y_coor])
            if x_coor < centre_x and y_coor > centre_y:
                Lower_Left_Point.append([x_coor, y_coor])
            if x_coor > centre_x and y_coor < centre_y:
                Upper_Right_Point.append([x_coor, y_coor])
            if x_coor > centre_x and y_coor > centre_y:
                Lower_Right_Point.append([x_coor, y_coor])

        Ref_Point_Array_Ref = []
        Ref_Point_Array_Ref.append(Upper_Left_Point[0])
        Ref_Point_Array_Ref.append(Upper_Right_Point[0])
        Ref_Point_Array_Ref.append(Lower_Left_Point[0])
        Ref_Point_Array_Ref.append(Lower_Right_Point[0])

        # Return the Red reference points' coordindates
        return Ref_Point_Array_Ref, Ref_Point_Areas, processed_Ref_Img






    # Function_4
    # Flatten an image for pixel rescaling
    @staticmethod
    def flatten_img(img):
        """Convert an image with size (M, N, 3) to (M * N, 3).
        Flatten pixels into a 1D array where each row is a pixel and the columns are RGB values.
        """
        # The image needs to contain 3 channels...
        result_image = img.reshape((np.multiply(*img.shape[:2]), 3))
        return result_image

    # Function_3
    # Calcuale the pixel to mm conversion
    @staticmethod
    def convert_pixels_to_mm(pixels_in_circ, circle_width):
        """pixels_in_circ is the array of `region.area`s"""
        averagePixelArea = np.mean(pixels_in_circ)
        # Calculate the pixel to mm conversion rate
        pixels_per_mm = round(math.sqrt(averagePixelArea / math.pi) / circle_width, 2)
        return (pixels_per_mm)


    # Function_5
    # Perform perspective transformation, OpenCV is used to improve the performance
    @staticmethod
    def PerspectiveTrans_2D(ref_image, Ref_Point_Array):
        """Perform perspective transformation in 2D"""
        # Read the original image for 2D resolution
        img = ref_image.copy()
        rows, cols, ch = img.shape
        # rows - y axis; cols - x axis
        Reference_Distance = math.sqrt((Ref_Point_Array[0][0] - Ref_Point_Array[1][0]) ** 2 +
                                       (Ref_Point_Array[0][1] - Ref_Point_Array[1][1]) ** 2)
        Tray_Cols = 0
        if Reference_Distance > cols * 0.75:
            # This is a full tray 5*8 or 4*6
            columns = cols
        else:
            # This is a half tray 5x4 or 4*4
            columns = cols / 2

        # Red points positions on the original image
        pts1 = np.float32([Ref_Point_Array[0], Ref_Point_Array[1], Ref_Point_Array[2], Ref_Point_Array[3]])
        # Positions on the transformed image
        pts2 = np.float32([[0, 0], [columns, 0], [0, rows], [columns, rows]])
        # Using 2D list, not an array
        # This will be returned for following process
        New_Ref_Points_Array = [[0, 0], [columns, 0], [0, rows], [columns, rows]]
        # Use OpenCV to perform the perspective transformation
        Transform_Array = cv2.getPerspectiveTransform(pts1, pts2)
        Transformed_img = cv2.warpPerspective(img, Transform_Array, (columns, rows))  # x, y

        # Return the transformed image and reference points array
        return Transformed_img, New_Ref_Points_Array

    @staticmethod
    def PerspectiveTrans_2D_Wheat(ref_image, Ref_Point_Array):
        """Perform perspective transformation in 2D"""
        # Read the original image for 2D resolution
        img = ref_image.copy()
        rows, cols, ch = img.shape
        # rows - y axis; cols - x axis
        Reference_Distance_x = math.sqrt((Ref_Point_Array[0][0] - Ref_Point_Array[1][0]) ** 2 +
                                         (Ref_Point_Array[0][1] - Ref_Point_Array[1][1]) ** 2)
        Reference_Distance_y = math.sqrt((Ref_Point_Array[0][0] - Ref_Point_Array[2][0]) ** 2 +
                                         (Ref_Point_Array[0][1] - Ref_Point_Array[2][1]) ** 2)
        columns = int(Reference_Distance_x)
        rows = int(Reference_Distance_y)

        # Red points positions on the original image
        pts1 = np.float32([Ref_Point_Array[0], Ref_Point_Array[1], Ref_Point_Array[2], Ref_Point_Array[3]])
        # Positions on the transformed image
        pts2 = np.float32([[0, 0], [columns, 0], [0, rows], [columns, rows]])
        # Using 2D list, not an array
        # This will be returned for following process
        New_Ref_Points_Array = [[0, 0], [columns, 0], [0, rows], [columns, rows]]
        # Use OpenCV to perform the perspective transformation
        Transform_Array = cv2.getPerspectiveTransform(pts1, pts2)
        Transformed_img = cv2.warpPerspective(img, Transform_Array, (columns, rows))  # x, y
        # Return the transformed image and reference points array
        return Transformed_img, New_Ref_Points_Array

    # Function_6
    # Generate segmented pots, 1x1 is accepted
    @staticmethod
    def PotSegmentation(img, row_no, column_no, Ref_Points_Array):
        """Generate an image to contain segmented pots"""
        # Generate a blank image to contain segmented pots
        Blank_Img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        Pot_Image = Blank_Img.copy()
        Seg_Img_tmp = Blank_Img.copy()
        tmp_line_img = Blank_Img.copy()

        # Set up empty arrays for boundary
        Upper_Left_Point = []
        Lower_Left_Point = []
        Upper_Right_Point = []
        Lower_Right_Point = []

        # Read x,y coordinates from the reference points array
        for x_coor, y_coor in Ref_Points_Array:
            if x_coor < img.shape[0] * 0.5 and y_coor < img.shape[1] * 0.5:
                Upper_Left_Point.append([x_coor, y_coor])
            if x_coor < img.shape[0] * 0.5 and y_coor > img.shape[1] * 0.5:
                Lower_Left_Point.append([x_coor, y_coor])
            if x_coor > img.shape[0] * 0.5 and y_coor < img.shape[1] * 0.5:
                Upper_Right_Point.append([x_coor, y_coor])
            if x_coor > img.shape[0] * 0.5 and y_coor > img.shape[1] * 0.5:
                Lower_Right_Point.append([x_coor, y_coor])
        # print Upper_Left_Point, Lower_Left_Point, Upper_Right_Point, Lower_Right_Point

        # Start to segment pots - rows
        # Add the bottom line!!!
        for i in range(row_no):  # Draw the top/bottom lines
            # row_no + 1 lines will be produced
            range_row = int((i * (Upper_Left_Point[0][1] + Lower_Left_Point[0][1]) / row_no))
            # In case the calculation is within the image boundary
            if (Upper_Left_Point[0][1] + range_row) < (img.shape[0] - 5):
                # Draw lines in rows, make it a little bit skew to correct labelling issue
                rr, cc, val = line_aa(Upper_Left_Point[0][1] + range_row, Upper_Left_Point[0][0],
                                      Upper_Right_Point[0][1] + range_row + 1, Upper_Right_Point[0][0] - 1)
            else:  # out of the image boundary
                rr, cc, val = line_aa(img.shape[0] - 5, Upper_Left_Point[0][0],
                                      img.shape[0] - 5, Upper_Right_Point[0][0] - 1)
            # At y-axis will be 0..shape[1]-1
            tmp_line_img[rr, cc] = val * 255
            Pot_Image = np.logical_or(Pot_Image, tmp_line_img)

            # Start to segment pots - columns
        tmp_line_img = Blank_Img.copy()  # reset the temp line image
        for j in range(column_no):
            # column_no + 1 lines will be produced
            range_col = int((j * (Upper_Left_Point[0][0] + Upper_Right_Point[0][0]) / column_no))
            # In case the calculation is within the image boundary
            if (Upper_Left_Point[0][0] + range_row) < (img.shape[1] - 5):
                # Draw lines in columns
                rr, cc, val = line_aa(Upper_Left_Point[0][1], Upper_Left_Point[0][0] + range_col,
                                      Lower_Left_Point[0][1] - 1, Lower_Left_Point[0][0] + range_col)
            else:  # out of the image boundary
                rr, cc, val = line_aa(Upper_Left_Point[0][1], img.shape[1] - 5,
                                      Lower_Left_Point[0][1] - 1, img.shape[1] - 5)
            # At x-axis will be 0..shape[0]-1
            tmp_line_img[rr, cc] = val * 255
            Pot_Image = np.logical_or(Pot_Image, tmp_line_img)

            # Dilate the pot image
        selem = disk(3)
        Pot_Image = dilation(Pot_Image, selem)
        # Find the segment region
        Pot_Segment = np.logical_not(Pot_Image)
        Pot_Segment_Refine = erosion(Pot_Segment, disk(4))

        # Return pot and segment border images
        return Pot_Image, Pot_Segment_Refine

    @staticmethod
    def PotSegmentation_Wheat(img, row_no, column_no, Ref_Points_Array):
        """Generate an image to contain segmented pots"""
        # Generate a blank image to contain segmented pots
        Blank_Img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        Pot_Image = Blank_Img.copy()
        Seg_Img_tmp = Blank_Img.copy()
        tmp_line_img = Blank_Img.copy()

        # Set up empty arrays for boundary
        Upper_Left_Point = []
        Lower_Left_Point = []
        Upper_Right_Point = []
        Lower_Right_Point = []

        # Read x,y coordinates from the reference points array
        for x_coor, y_coor in Ref_Points_Array:
            if x_coor < img.shape[0] * 0.5 and y_coor < img.shape[1] * 0.5:
                Upper_Left_Point.append([x_coor, y_coor])
            if x_coor < img.shape[0] * 0.5 and y_coor > img.shape[1] * 0.5:
                Lower_Left_Point.append([x_coor, y_coor])
            if x_coor > img.shape[0] * 0.5 and y_coor < img.shape[1] * 0.5:
                Upper_Right_Point.append([x_coor, y_coor])
            if x_coor > img.shape[0] * 0.5 and y_coor > img.shape[1] * 0.5:
                Lower_Right_Point.append([x_coor, y_coor])
        # print Upper_Left_Point, Lower_Left_Point, Upper_Right_Point, Lower_Right_Point

        # Start to segment pots - rows
        # Add the bottom line!!!
        for i in range(row_no):  # Draw the top/bottom lines
            # row_no + 1 lines will be produced
            range_row = int((i * (Upper_Left_Point[0][1] + Lower_Left_Point[0][1]) / row_no))
            # In case the calculation is within the image boundary
            if (Upper_Left_Point[0][1] + range_row) < (img.shape[0] - 5):
                # Draw lines in rows, make it a little bit skew to correct labelling issue
                rr, cc, val = line_aa(Upper_Left_Point[0][1] + range_row, Upper_Left_Point[0][0],
                                      Upper_Right_Point[0][1] + range_row + 1, Upper_Right_Point[0][0] - 1)
            else:  # out of the image boundary
                rr, cc, val = line_aa(img.shape[0] - 5, Upper_Left_Point[0][0],
                                      img.shape[0] - 5, Upper_Right_Point[0][0] - 1)
            # At y-axis will be 0..shape[1]-1
            tmp_line_img[rr, cc] = val * 255
            Pot_Image = np.logical_or(Pot_Image, tmp_line_img)

            # Start to segment pots - columns
        tmp_line_img = Blank_Img.copy()  # reset the temp line image
        for j in range(column_no):
            # column_no + 1 lines will be produced
            range_col = int((j * (Upper_Left_Point[0][0] + Upper_Right_Point[0][0]) / column_no))
            # In case the calculation is within the image boundary
            if (Upper_Left_Point[0][0] + range_row) < (img.shape[1] - 5):
                # Draw lines in columns
                rr, cc, val = line_aa(Upper_Left_Point[0][1], Upper_Left_Point[0][0] + range_col,
                                      Lower_Left_Point[0][1] - 1, Lower_Left_Point[0][0] + range_col)
            else:  # out of the image boundary
                rr, cc, val = line_aa(Upper_Left_Point[0][1], img.shape[1] - 5,
                                      Lower_Left_Point[0][1] - 1, img.shape[1] - 5)
            # At x-axis will be 0..shape[0]-1
            tmp_line_img[rr, cc] = val * 255
            Pot_Image = np.logical_or(Pot_Image, tmp_line_img)

            # Dilate the pot image
        selem = disk(3)
        Pot_Image = dilation(Pot_Image, selem)
        # Find the segment region
        Pot_Segment = np.logical_not(Pot_Image)
        Pot_Segment_Refine = erosion(Pot_Segment, disk(4))

        # Return pot and segment border images
        return Pot_Image, Pot_Segment_Refine



    # Function_7
    # Non-linear LAB based green region extraction
    @staticmethod
    def LAB_Img_Segmentation(img, pix2mm):
        """Extract green regions from the image"""
        # Colour space with dimension L for lightness
        # a and b for the colour-opponent dimensions, based on nonlinearly compressed

        # Colour space with dimension L for lightness
        # a and b for the colour-opponent dimensions, nonlinearly compressed CIELAB
        leaf_img_LAB = color.rgb2lab(img)
        # color-opponent - negative values indicate green and hence are retained
        a_2D_Color_ND = leaf_img_LAB[:, :, 1]
        # color-opponent - positive values indicate yellow, which means bright objects will be retained
        b_2D_Color_ND = leaf_img_LAB[:, :, 2]
        LAB_Image_Ref = (b_2D_Color_ND - a_2D_Color_ND)
        # Use global adaptive thresholding to seek the threshold, as the image has been denoised
        global_thresh_Ref_Points_Value = filters.threshold_otsu(LAB_Image_Ref)
        # Use one standard deviation to segment intensity distribution, if the otus value is too low
        if global_thresh_Ref_Points_Value > 0 and global_thresh_Ref_Points_Value >= LAB_Image_Ref.max() * 0.425:
            binary_global_LAB = LAB_Image_Ref > LAB_Image_Ref.max() * 0.75  # close to 32% - tails of 1SD
        elif global_thresh_Ref_Points_Value > 0 and global_thresh_Ref_Points_Value < LAB_Image_Ref.max() * 0.425:
            binary_global_LAB = LAB_Image_Ref > LAB_Image_Ref.max() * 0.35  # close to 68% - 1SD
            # Close to 2 standard deviations, Was using 0.25 for 1SD
        else:  # global_thresh_Ref_Points_Value < 0, in very rare cases
            binary_global_LAB = LAB_Image_Ref > 0

        # 1-pixel regions, expanded 1 pixel, which needs to be rescaled during quantification
        Img_cleaned = remove_small_objects(binary_global_LAB, pix2mm * 2.5)
        # use the pixel and metrics ratio, as small leaves are around 10 mm^2
        # Do not fill images, as the first two true leaves are not touched
        selem = disk(3)
        Img_cleaned_Dilated = dilation(Img_cleaned, selem)
        Img_cleaned_Ref = remove_small_holes(Img_cleaned_Dilated, min_size=int(pix2mm * 3.25) ** 3)
        # In total 1 pixels have been expanded around the outline of every object
        erode_binary_Img = erosion(Img_cleaned_Ref, disk(2))
        erode_binary_Img = remove_small_objects(erode_binary_Img, pix2mm * 5.25)
        # use the pixel and metrics ratio, as dilated small leaves are around 25 mm^2

        # Leaf segmented image based on LAB colour spacing
        return erode_binary_Img, leaf_img_LAB

    @staticmethod
    def LAB_Img_Segmentation_Wheat(img, pix2mm):
        """Extract green regions from the image"""
        # Colour space with dimension L for lightness
        # a and b for the colour-opponent dimensions, based on nonlinearly compressed

        # Colour space with dimension L for lightness
        # a and b for the colour-opponent dimensions, nonlinearly compressed CIELAB
        leaf_img_LAB = color.rgb2lab(img)
        # color-opponent - negative values indicate green and hence are retained
        a_2D_Color_ND = leaf_img_LAB[:, :, 1]
        # color-opponent - positive values indicate yellow, which means bright objects will be retained
        b_2D_Color_ND = leaf_img_LAB[:, :, 2]
        LAB_Image_Ref = (b_2D_Color_ND - a_2D_Color_ND)
        # Use global adaptive thresholding to seek the threshold, as the image has been denoised
        global_thresh_Ref_Points_Value = filters.threshold_otsu(LAB_Image_Ref)
        # Use one standard deviation to segment intensity distribution, if the otus value is too low
        if global_thresh_Ref_Points_Value > 0 and global_thresh_Ref_Points_Value >= LAB_Image_Ref.max() * 0.425:
            binary_global_LAB = LAB_Image_Ref > LAB_Image_Ref.max() * 0.625
        elif global_thresh_Ref_Points_Value > 0 and global_thresh_Ref_Points_Value < LAB_Image_Ref.max() * 0.425:
            binary_global_LAB = LAB_Image_Ref > LAB_Image_Ref.max() * 0.125
            # Close to 2 standard deviations, Was using 0.25 for 1SD
        else:  # global_thresh_Ref_Points_Value < 0, in very rare cases
            binary_global_LAB = LAB_Image_Ref > 0

        # 1-pixel regions, expanded 1 pixel, which needs to be rescaled during quantification
        Img_cleaned = remove_small_objects(binary_global_LAB, pix2mm * 2.5)
        # use the pixel and metrics ratio, as small leaves are around 10 mm^2
        # Do not fill images, as the first two true leaves are not touched
        selem = disk(1)
        Img_cleaned_Dilated = dilation(Img_cleaned, selem)
        Img_cleaned_Ref = remove_small_holes(Img_cleaned_Dilated, min_size=int(pix2mm * 3.25) ** 3)
        # In total 1 pixels have been expanded around the outline of every object
        erode_binary_Img = erosion(Img_cleaned_Ref, disk(1))
        erode_binary_Img = remove_small_objects(erode_binary_Img, pix2mm * 5.25)
        # use the pixel and metrics ratio, as dilated small leaves are around 25 mm^2

        # Leaf segmented image based on LAB colour spacing
        return erode_binary_Img, leaf_img_LAB


    # Function_8
    # Generate an image that is represented by excessive greenness and excessive red
    # normalise the specified numpy matrix so that it is in range [0,1]
    # @param mat: the numpy matrix to normalise
    @staticmethod
    def norm_range(mat, min_val, max_val):
        """Normalise a specified numpy matrix"""
        # get the range
        range = max_val - min_val
        # as long as the range is not 0 then scale so that range is 1
        if range > 0:
            # subtract offset so min value is 0
            mat -= min_val
            # normalise so values are in range 0
            mat /= float(range)


    # return excessive green representation of a provided image
    # @param img: RGB image needs to be converted (np.uint8)
    @staticmethod
    def compute_greenness_img(img):
        """Transfer a given image to excessive greenness and excessive red"""
        # convert to floating point representation [0, 1]
        img = img.astype(np.float64) / 255.0
        # split image into its r, g, and b channels
        r, g, b = cv2.split(img)
        # create 2D sum matrix (element-wise addition of r, g, and b values
        sum = r + g + b
        # divide each colour channel by the sum (element-wise)
        r = np.divide(r, sum)
        g = np.divide(g, sum)
        b = np.divide(b, sum)
        # compute excessive green image
        ex_g = 2.0 * g - r - b
        # compute excessive red image
        ex_r = 1.4 * r - b
        # compute vegetative image (excessive green - excessive red)
        veg = ex_g - ex_r
        # noramlsie the image
        Analysis.norm_range(veg, -2.4, 2.0)  # -2.4 is the minimum veg value (1, 0, 0) and 2.0 is maximum veg value (0, 1, 0)
        # convert back to 8-bit unsigned int representation [0, 255]
        veg = veg * 255
        veg = veg.astype(np.uint8)
        # return the vegetative image
        return veg


    # Function_9
    # Use kmeans to precisely segment pixel groups
    @staticmethod
    def kmeans_cluster(img, k_value):
        """Cluster pixels"""
        # As requested by OpenCV, transfer to float32
        kmeans_img = np.float32(img.reshape((-1, 1)))
        # Define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        # cv2.TERM_CRITERIA_EPS stops the algorithm iteration if specified accuracy is reached.
        # cv2.TERM_CRITERIA_MAX_ITER stops the algorithm after the specified number of iterations is reached
        if k_value > 10:
            k_value = 10  # k_value should be less than 10, otherwise the classificaiton will be too slow (this should be sorted by GUI, so this should never be an issue)
        # Four groups were determined based on distances NearestNeighbors
        # 1) leaf, 2) soil/compost, 3) reference points, 4) others such as reflection
        # ..! OpenCV 2.4.11 version below
        ret, label, center = cv2.kmeans(kmeans_img, k_value, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        # cv2.KMEANS_RANDOM_CENTERS - for cluster centers, not required in the analysis
        # Convert the image back to uint8
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((img.shape))
        # return the kmeans image
        return res2







    @staticmethod
    def LUV_Img_Segmentation(img, labImg, pix2mm):
        """Use LUV to precisely detect leaves"""
        # Use LUV colour spacing - International Commission on Illumination
        # LUV color spacing deals with colored lights in CIELUV's uniform chromaticity diagram
        # Use green colour in this case
        leaf_img_luv = color.rgb2luv(img)
        LUV_Ref = leaf_img_luv[:, :, 1] - leaf_img_luv[:, :, 0]
        global_thresh_luv = filters.threshold_otsu(LUV_Ref)
        # Get the adaptive threshold based on LUV color space

        if global_thresh_luv < 0:
            binary_global_luv = LUV_Ref < global_thresh_luv * 0.725  # close to 2SD
        else:
            binary_global_luv = LUV_Ref < global_thresh_luv * 0.525  # close to 1SD
        # Refine the result with lab-based detection
        binary_global_luv_Ref = np.logical_and(binary_global_luv, labImg)
        # 5-pixel regions, expanded 1 pixel, which needs to be rescaled during quantification
        Img_cleaned = remove_small_objects(binary_global_luv_Ref, pix2mm * 2.5)
        # Do not fill images, as the first two true leaves are not touched
        selem = disk(5)
        Img_cleaned_Dilated = dilation(Img_cleaned, selem)
        Img_cleaned_Ref = remove_small_holes(Img_cleaned_Dilated, min_size=int(pix2mm * 3.25) ** 3)
        # 5-pixel kernel with 25% continuously increase, see skimage.morphology
        # In total 1 pixel has been expaned around the outline of every object
        erode_binary_Img = erosion(Img_cleaned_Ref, disk(5))
        erode_binary_Img = remove_small_objects(erode_binary_Img, pix2mm * 5.25)
        # Make sure small leaves can be passed

        # Leaf segmented image based on LUV colour spacing
        return erode_binary_Img

    # Function_10
    # Use graph theory to locate end or branching points of a skeleton
    @staticmethod
    def find_end_points(skel):
        """Detect end points of a skeleton"""
        # Four possible matrix representation
        struct1, origin1 = np.array([
            [0, 0, 0],
            [0, 1, 0],
        ]), (0, 0)
        struct2, origin2 = np.array([
            [0, 0],
            [0, 1],
            [0, 0],
        ]), (0, 0)
        struct3, origin3 = np.array([
            [0, 1, 0],
            [0, 0, 0],
        ]), (-1, 0)
        struct4, origin4 = np.array([
            [0, 0],
            [1, 0],
            [0, 0],
        ]), (0, -1)

        # Match end point structures with the skeleton
        ret = None
        for i in range(1, 5):
            struct, origin = locals()['struct%d' % (i)], locals()['origin%d' % (i)]
            if ret is None:
                ret = binary_hit_or_miss(skel, structure1=struct, origin1=origin)
            else:
                ret = np.logical_or(ret, binary_hit_or_miss(skel, structure1=struct, origin1=origin))
        return np.transpose(np.nonzero(ret)[::-1])

    @staticmethod
    def find_branches(skel):
        """Detect branching points of a skeleton"""
        # Four possible matrix representation
        struct1 = np.array([
            [0, 1, 0],
            [0, 1, 0],
            [1, 0, 1]
        ])
        struct2 = np.array([
            [1, 0, 0],
            [0, 1, 1],
            [0, 1, 0]
        ])
        struct3 = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [1, 0, 1]
        ])
        struct4 = np.array([
            [0, 0, 0],
            [1, 1, 1],
            [0, 1, 0]
        ])
        struct5 = np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ])

        # Match branching point structures with the skeleton
        structs = [np.rot90(struct1, i) for i in range(4)]
        structs += [np.rot90(struct2, i) for i in range(4)]
        structs += [np.rot90(struct3, i) for i in range(4)]
        structs += [np.rot90(struct4, i) for i in range(4)]
        structs += [np.rot90(struct5, i) for i in range(4)]

        ret = None
        for i in range(len(structs)):
            if ret is None:
                ret = binary_hit_or_miss(skel, structure1=structs[i])
            else:
                ret = np.logical_xor(ret, binary_hit_or_miss(skel, structure1=structs[i]))
        return np.transpose(np.nonzero(ret)[::-1])

    # this function sets the progress status to the specified value in the GUI series processing table
    # @param progress: progress value (should be between 0 and 100)
    def set_progress(self, progress):
        # update the series processing table to show the specified progress value
        self.update_tree(Analysis.PROCESSING_LABEL + "(" + str(int(progress)) + "%)")

    # this function updates the status column for this image series in the series processing table with the specified text
    # @param text: the text to display in the corresponding row in the series processing table
    # NOTE: this function must be run in the main thread (use update_tree, if not)
    def update_tree_in_gui_thread(self, text):
        # update the series processing table to show the text
        self.app.update_treeview(self.app.series_processing_table, self.series.result_index, Analysis.STATUS_COL_INDEX, text)
        # as the content of the table has been changed, we should ask the table to resize its columns to fit the contents
        self.app.series_processing_table.auto_size()

    # this function updates the status column for this image series in the series processing table with the specified text
    # @param text: the text to display in the corresponding row in the series processing table
    def update_tree(self, text):
        # update the series processing table to show the text (but send this request to the GUI queue so that it is run in the main thread)
        GuiQueue.gui_queue.put(partial(self.update_tree_in_gui_thread,text))

    # this function generates the figure to display in the machine learning settings section of the GUI
    # the function returns the a plot showing the frequency of pixels in each colour cluster group and the number of suggested pixel groups based on the median frequency value
    # @param img: the image to cluster the pixels on
    @staticmethod
    def generate_machine_learning_pixel_clustering_figure(img):
        # downsize the image to reduce processing demand
        Resize_Ratio = 1.0 / (img.shape[0] / 1024.0)
        img = img_as_ubyte(rescale(img, Resize_Ratio))
        # get the width and height of the image
        h, w = img.shape[:2]
        # cut the sides off the image to ensure that the edges of the image are not interfering with main colours
        # we will cut the image by 10% from each edge
        frac = 0.1
        # compute the new coordinates of the image
        x0 = int(w * frac)
        x1 = int(w * (1.0 - frac))
        y0 = int(h * frac)
        y1 = int(h * (1.0 - frac))
        # crop the image
        img_cropped = img[y0:y1, x0:x1, :]
        # get the image aspect ratio
        img_cropped_ratio = img_cropped.shape[1]/float(img_cropped.shape[0])
        # create the clustering plot and make the plot the same aspect ratio as the image (so display purposes)
        kmeans_img, suggested_clusters = clustering_plot.kmeans(img_cropped, img_cropped_ratio)
        # return the plot and the suggested number of pixel groups
        return kmeans_img, suggested_clusters

    # this function prints the provided message to the assigned log in the GUI
    # additional arguments may be provided. These will be included in the message and separated with spaces (like print function in python)
    def print_to_log(self, msg, *args):
        # all messages to the GUI log will also be output the file log
        self.log.print_to_file_log(msg, *args)
        # output message to log, we need this to run in main thread so add to GUI queue.
        GuiQueue.gui_queue.put(partial(self.log.print_to_log, msg, *args))


# this function is used to find the total number of leaves using a circular sweeping technique
# @param img: the image to count the leaves in
# @param cx: the centroid x position of the plant (pivot point of sweeping algorithm)
# @param cy: the centroid y position of the plant (pivot point of sweeping algorithm)
def sweeping(img, cx, cy):
    # get the width and height of the image
    h, w = img.shape[:2]
    # get the longest distance between any 2 pixels in this image (image diagonal using pythagoras theorem).
    # this is used when drawing the sweeping line onto a mask. We want this line span the entire mask
    diagonal = math.sqrt(h * h + w * w)
    # get a list of squared distances from the centroid to the boundary along the sweeping line.
    # store only the furthest pixel away from the centroid along the sweeping line.
    # we use squared distances to reduce computation associated with sqrt.
    distances2 = []
    # get a list of 2D coordinate points (when the sweeping line intersects the boundary).
    # again, as before, we only store the point on the sweeping line that is furthest away from the centroid)
    points = []
    # mask used to draw sweeping line at each degree that will be used to isolate bounadry pixels that intersect sweeping line at each degree
    mask = np.zeros((h, w), np.uint8)
    # sweep in circular motion (360 degrees), at each degree
    for i in range(0, 360):
        # clear the sweeping line mask
        mask[:, :] = 0
        # convert degree into radians
        rad = math.radians(i)
        # determine the x and y coordinate offset of the sweeping line end-point (based on radians)
        # it is likely to be outside of the image bounds as this is based on image diagonal
        x = diagonal * math.cos(rad)
        y = diagonal * math.sin(rad)
        # get the image coordinates of the sweeping line
        # line starts at the centroid
        x0 = int(round(cx))
        y0 = int(round(cy))
        # line end-point needs to be relative to the centroid
        x1 = int(round(x0 + x))
        y1 = int(round(y0 + y))
        # draw the line on the mask, use thick line
        cv2.line(mask, (x0, y0), (x1, y1), (255), 3)
        # bitwise and the input boundary mask and the sweeping line mask to return only those boundary pixels that intersect the sweeping line
        indices = np.nonzero(np.bitwise_and(img, mask))
        # store the max distance squared of any intersection pixel indentified
        max_dist2 = 0
        # store the corresponding point coordinate
        max_dist2_point = (cx, cy)
        # iterate through the list of intersecting boundary points
        for j in range(len(indices[0])):
            # get the intersection coordinates
            iy = indices[0][j]
            ix = indices[1][j]
            # get the distance in each dimension between the centroid and the boundary point
            dx = ix - cx
            dy = iy - cy
            # compute the squared distance
            dist2 = dx * dx + dy * dy
            # is this squared distance the largest so far?
            if dist2 > max_dist2:
                # if so, store this distance and the corresponding intersection point
                max_dist2 = dist2
                max_dist2_point = (ix, iy)
        # now we have iterated through all of the intersecting boundary points of this sweeping line at this rotation,
        # add the maximum distance and point to these respective lists
        distances2.append(max_dist2)
        points.append(max_dist2_point)
        # return the array of squared distances and corresponding points
    return distances2, points

# find and return the peaks in the provided 1D time-series signal
# returns np array of peak indices and np array of peak values
# @param data: 1D array of values representing time-series signal
def find_peaks(data):
    # use third-party algorithm to detect peaks
    [peaks, _] = findpeaks.peakdetect(np.array(data), lookahead=5)
    # now convert the output of this algorithm into 2 separate array, peak indices and peak values
    # list of peak indices (position along input signal)
    indices = []
    # list of peak values (the value of the signal at the peak)
    values = []
    # iterate through the output peak information
    for peak in peaks:
        # add the peak index to the indices list
        indices.append(peak[0])
        # add the peak value to the values list
        values.append(peak[1])
    # return these as arrays
    return np.array(indices), np.array(values)


# this function smooths and returns the provided 1D signal
# @param data: 1D array of values representing time-series signal
# @param half_kernel_size: the half size of the gaussian kernel to use to smooth the signal. Kernel size will be half_size * 2 + 1
# @param sigma: the sigma value used to generate the gaussian kernel
def smooth_series(data, half_kernel_size, sigma=-1):
    # determine the full kernel size
    kernel_size = half_kernel_size * 2 + 1
    # generate the kernel
    kernel = list(cv2.getGaussianKernel(kernel_size, sigma)[:, 0])
    # create an array to represent the smoothed signal
    new_data = [0] * len(data)
    # iterate through all of the data points in the signal
    for i in range(len(data)):
        # get a list of neighbouring values that we be used to determine signal's new value at this index
        values = []
        # iterate through the neighbouring signal values which the kernel overlaps when the kernel centre is placed at current index
        for j in np.arange(-half_kernel_size, half_kernel_size + 1):
            # for each position in the kernel, get the kernel's value and multiply by signal data
            weighted_value = data[(i + j) % len(data)] * kernel[j + half_kernel_size]
            # add this value to the list of weighted kernel values
            values.append(weighted_value)
        # get the mean value of the list of weighted kernel values and set this value as the new data value
        new_data[i] = np.mean(values)
    # return the smoothed signal
    return new_data


# this function attempts to find the total number of leaves of a plant within a pot
# @param img: the tray image to analyse (binary mask with 0s and 1s)
# @param out_img: the output image to update (used to display result)
# @param cx: the centroid x position of the plant (pivot point of sweeping algorithm)
# @param cy: the centroid y position of the plant (pivot point of sweeping algorithm)
# @param pot_region: data structure representing the pot region to analyse (should be bbox variable)
def find_leaves(img, out_img, cx, cy, pot_region):
    # convert the binary mask to unsigned int representation [0, 255]
    img = img.astype(np.uint8) * 255
    # get the coordinates of the pot to analyse
    bb_y0, bb_x0, bb_y1, bb_x1 = pot_region.bbox

    try:
        # add the binary mask to the output image (it's a 3 channel image, so must be added to each channel)
        out_img[bb_y0:bb_y1 + 1, bb_x0:bb_x1 + 1, 0] = img[bb_y0:bb_y1 + 1, bb_x0:bb_x1 + 1]
        out_img[bb_y0:bb_y1 + 1, bb_x0:bb_x1 + 1, 1] = img[bb_y0:bb_y1 + 1, bb_x0:bb_x1 + 1]
        out_img[bb_y0:bb_y1 + 1, bb_x0:bb_x1 + 1, 2] = img[bb_y0:bb_y1 + 1, bb_x0:bb_x1 + 1]
        # get the boolean sub image (representing just the pot to process)
        sub_img = img[bb_y0:bb_y1 + 1, bb_x0:bb_x1 + 1]
        # find the binary mask contour
        contours, _ = cv2.findContours(sub_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # clear the sub image
        sub_img[:, :] = 0
        # replace filled-in mask with identified boundary (contour). Use thick line
        cv2.drawContours(sub_img, contours, -1, (255), 3)
        # run sweeping algorithm to generate signal of distances from centroid to plant boundary
        distances, points = sweeping(sub_img, cx - bb_x0, cy - bb_y0)
        # smooth the resulting signal
        distances_smooth = smooth_series(distances, 5, sigma=1.0)
        # if the sweeping algorithm started near the tip of a leaf, a peak may be split (so that one half is at the beginning of the signal and the other is at the end).
        # This may cause the peak counting algorithm to count this as two peaks instead of one.
        # To resolve this issue, we get the index of the position along the signal that has the lowest value
        # and we shift the signal in a wrapping like fashion, such that this lowest point is now the start of the signal.

        # get the position with the lowest value
        lowest_index = np.argmin(distances_smooth)
        # shift the distance and point arrays by the this offset
        # create 2 new arrays (first, copy from the low position to the end of the signal)
        shifted_distances = distances[lowest_index:]
        shifted_points = points[lowest_index:]
        # then copy from the beginning of the signal to the low position
        shifted_distances.extend(distances[:lowest_index])
        shifted_points.extend(points[:lowest_index])
        # smooth the shifted distances
        new_distances = smooth_series(shifted_distances, 5, sigma=1.0)
        # convert the signal into a data type accepted by the find peaks code
        d = np.zeros((len(new_distances)), np.int64)
        # iterate through the signal values (distances from centroid to boundary)
        for i in range(len(new_distances)):
            d[i] = new_distances[i]
        # find the peaks in the signal
        peaks = find_peaks(d)
        # iterate through the peaks
        for peak_index in peaks[0]:
            # get the coordinate of the peak in the sub image
            (ix, iy) = shifted_points[peak_index]
            # convert this coordinate to a coordinate of the whole image
            ix += bb_x0
            iy += bb_y0
            # draw a line on the output image from the centroid to the peak indetified on the plant boundary
            cv2.line(out_img, (int(cx), int(cy)), (int(ix), int(iy)), (255, 0, 0), 2)
            # draw a circle at the boundary point (yellow circle with red border)
            cv2.circle(out_img, (ix, iy), 10, (255, 255, 0), -1)
            cv2.circle(out_img, (ix, iy), 10, (255, 0, 0), 1)
        # return the output image and the number of peaks identified
        return out_img, len(peaks[0])
    # if there was an exception then return blank image and zero peaks
    except:
        # black out the pot region in the output image
        out_img[bb_y0:bb_y1 + 1, bb_x0:bb_x1 + 1] = (0, 0, 0)
        return out_img, 0
