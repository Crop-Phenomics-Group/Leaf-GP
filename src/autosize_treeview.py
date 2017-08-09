import sys
from threading import Lock

if sys.version_info[0] == 2:
    from Tkinter import *
    from ttk import *
    import ttk
    from tkFont import Font
else:
    from tkinter import *
    import tkinter.ttk
    from tkinter.font import Font

# This class subclasses the Tkinter treeview class to provided automated column resizing
# NOTE: This class was designed to work when treeview is added to and will need to be developed further to support
# other functionality such as remove, reorder, etc.
class AutoSizeTreeView(ttk.Treeview):

    # pixel padding for cells (used when calculating column widths)
    PADDING = 10
    # font style and size that is used
    TREEVIEW_FONT_NAME = "Helvetica"
    TREEVIEW_FONT_SIZE = -12
    TREEVIEW_FONT = (TREEVIEW_FONT_NAME, TREEVIEW_FONT_SIZE)
    # a lock mechanism that ensures that only 1 thread will execute the code that adjusts the column widths.
    # if multiple threads run this code at the same time, we may get unpredictable results.
    lock = Lock()

    # overrides the treeview insert function
    def insert(self, parent, index, iid=None, **kw):
        # insert into the treeview as normal
        item = ttk.Treeview.insert(self, parent, index, iid=iid, **kw)
        # afterwards automatically adjust column widths
        self.auto_size()
        # return the inserted item as in default behaviour
        return item

    # automatically adjusts the width of the treeview columns based on their content
    def auto_size(self):
        AutoSizeTreeView.auto_size_treeview_columns(self)

    # returns the number of columns in the treeview
    # this is a bit hacky, but couldn't find another way to count header columns
    def get_number_columns(self):
        # counter for the header columns
        header_count = 0
        # essentially loop through columns until an exception is thrown
        while True:
            try:
                # get the header column
                self.heading(header_count)
                # if an exception wasn't thrown here, there was a column so increment the counter
                header_count += 1
            except tkinter.TclError:
                # an exception was thrown when trying to retrieve the header column, so stop iterating
                break
        # return the number of columns
        return header_count

    # automatically adjusts the width of the treeview columns based on their content
    # @param tree: the tkinter treeview to automatically resize
    @staticmethod
    def auto_size_treeview_columns(tree):
        # use the lock to ensure that only 1 thread can do this simultaneously
        with AutoSizeTreeView.lock:
            # store an array of column widths (default 0 width)
            col_widths = [0] * tree.get_number_columns()
            # iterate through the headers of the columns
            for i in range(len(col_widths)):
                # calculate the width of the header content (including padding)
                col_widths[i] = AutoSizeTreeView.calculate_text_width(tree.heading(i)["text"]) + AutoSizeTreeView.PADDING
            # iterate through each row of the treeview
            for i in range(len(tree.get_children())):
                # get the row
                item = tree.get_children()[i]
                # get the cells within the row
                values = tree.item(item).get("values")
                # a counter for the column index
                col_counter = 0
                # iterate through the cells
                for value in values:
                    # calculate the width of the content (including padding)
                    width = AutoSizeTreeView.calculate_text_width(value) + AutoSizeTreeView.PADDING
                    # if this is the largest width in the column then update the column width
                    if width > col_widths[col_counter]:
                        col_widths[col_counter] = width
                    # increment the column counter
                    col_counter += 1
            # now we have established the column widths, iterate through the columns and resize accordingly
            for i in range(len(col_widths)):
                tree.column(tree["columns"][i], minwidth=col_widths[i], width=col_widths[i], stretch=True)

    # computes the width of the specified text
    # @param text: the text to calculate the width of
    # @param family: the font family that the text would be rendered in
    # @param size: the font size the text would be rendered in
    @staticmethod
    def calculate_text_width(text, family=TREEVIEW_FONT_NAME, size=TREEVIEW_FONT_SIZE):
        # create the font
        font = Font(family=family, size=size)
        # measure the width of the text and return
        return font.measure(text)
