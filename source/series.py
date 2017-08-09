import datetime


# class to store properties relating to an image series
class Series:

    # constructor
    def __init__(self):
        # series id
        self.id = None
        # tray number
        self.tray_number = None
        # experimental reference
        self.experiment_ref = None
        # dictionary storing dates/filenames of images
        self.date_file_dict = dict() # dates are keys and there is a list of files for each date (handles duplicates)
        # stores the root folder directory as input in the dialog
        self.root_directory = None
        # stores the directory of the image series (this may be different to the root directory when getting meta data from folder)
        self.directory = None

    # get the first (earliest) date in the series
    def get_first_date(self):
        # get a list of dates of images in the series
        dates = list(self.date_file_dict.keys())
        # if there is at least 1 date
        if len(dates) > 0:
            # sort the dates (as strings)
            dates.sort()
            # return the first date (okay as ordered YYYY-MM-DD)
            return dates[0]
        # otherwise return null representative string
        return "-"

    # get the last (most recent) date in the series
    def get_last_date(self):
        # get a list of dates of images in the series
        dates = list(self.date_file_dict.keys())
        # if there is at least 1 date
        if len(dates) > 0:
            # sort the dates (as strings)
            dates.sort()
            # return the last date (okay as ordered YYYY-MM-DD)
            return dates[-1]
        # otherwise return null representative string
        return "-"

    # get duration in days from the first and last image dates in the series
    # if the same day then the duration is 1 day
    def get_time_span(self):
        # get a list of dates in the date/file dictionary
        dates = list(self.date_file_dict.keys())
        try:
            # if there are dates
            if len(dates) > 0:
                # get the earliest date
                s_date = datetime.datetime.strptime(self.get_first_date(), "%Y-%m-%d")
                # get the latest date
                e_date = datetime.datetime.strptime(self.get_last_date(), "%Y-%m-%d")
                # get the number of days between these dates (add 1 so that same day is duration of 1)
                return (e_date - s_date).days + 1
            # if there are no dates then return null representative string
            return "-"
        except:
            # if there is an exception then return null representative string
            return "-"

    # add a filename to the date/file dictionary
    # @param date_str: the date string key to store the filename under
    # @param filename: the filename to store in the dictionary under the specified key
    def add_file_date(self, date_str, filename):
        # if this key is already a key in the dictionary
        if date_str in self.date_file_dict.keys():
            # get the list of filenames for the key
            file_list = self.date_file_dict[date_str]
            # add the filename to the list of filenames
            file_list.append(filename)
        # if the key is not in the dictionary
        else:
            # create a new filename list that includes the filename and add it to the dictionary under the key
            self.date_file_dict[date_str] = [filename]

    # get the number of images in the series
    def get_num_images(self):
        # counter storing the number of images
        num_images = 0
        # iterate through the dates in the date/file dictionary
        for date in self.date_file_dict.keys():
            # add the number of files in the dictionary under this date key to the total number of images
            num_images += len(self.date_file_dict[date])
        # return the total number of images
        return num_images
