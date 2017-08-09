Leaf-GP readme

main.py: Main GUI application and Tkinter window
analysis.py: Core algorithm for processing single image series

autosize_treeview.py: extends Tkinter treeview widget class to provided automated column resizing
checkbox_treeview.py: extends AutoSizeTreeView (autosize_treeview.py) to provide treeview widget with checkboxes per row
clustering_plot.py: functions for producing clustering plots for GUI
findpeaks.py: third-party code for finding peaks and troughs in signal
gui_queue.py: Queue class to store, monitor, and execute function calls from non-main threads to interact with GUI.
hook-analysis.py: this file is used to hook hidden imports when using PyInstaller when packaging the program as an executable.
logger_widget.py: class that represents a log widget (for displaying log windows)
main.spec: this is the spec file that can be provided to PyInstaller to create executable
plot_generator.py: code to produce image series plots (for GUI)
series.py: class to store properties relating to an image series (e.g. id, tray number, experimental reference, etc.)
traid_ids.py: stores mapping between csv column ids as enums and text-friendly descriptions
