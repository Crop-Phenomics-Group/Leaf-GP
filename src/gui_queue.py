import Queue


# This is a queue class used to store and execute function calls related to GUI events.
# Tkinter is not thread safe so all GUI calls must be processed by main thread.
# Other threads are permitted to push GUI function calls to this queue.
# If this class is created in main thread then it will check the queue automatically at pre-determined time interval and
# execute waiting requests.
class GuiQueue:

    # GUI function request queue
    gui_queue = Queue.Queue()
    # queue check interval
    QUEUE_CHECK_TIME_INTERVAL = 50

    # constructor
    def __init__(self, root):
        # store the root
        self.root = root
        # check the queue
        self.check_gui_queue()

    # check the GUI function request queue and execute functions
    def check_gui_queue(self):
        # while there are requests in the queue
        while not GuiQueue.gui_queue.empty():
            # get the function from the queue
            item = GuiQueue.gui_queue.get()
            # run the function
            item()
        # once the queue is empty run the check again after pre-determine time interval
        self.root.after(GuiQueue.QUEUE_CHECK_TIME_INTERVAL, self.check_gui_queue)