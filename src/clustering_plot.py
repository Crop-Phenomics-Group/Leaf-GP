import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
from skimage.transform import rescale
import numpy as np


# returns a numpy image of block colours stacked horizontally (used to show mean colours of buckets after k-means classification)
# @param colours: a list of RGB colours to put in the image (in order)
# @param block_size: the size in pixels of each colour block
def generate_block_colour_image(colours, block_size=10):
    # create a black image that is block_size in height and (block_size * number of colours) in width
    img = np.zeros((block_size, block_size * len(colours), 3), np.uint8)
    # loop through the colours in the colour list
    for i in range(len(colours)):
        # calculate the start and end indices of the image that should be coloured in the current colour
        s_index = int(block_size * i)
        e_index = int(block_size * (i + 1))
        # colour this region of the image in the current colour
        img[:, s_index:e_index] = colours[i]
    # return the generated image
    return img


def flatten_img(img):
    """Convert an image with size (M, N, 3) to (M * N, 3).
    Flatten pixels into a 1D array where each row is a pixel and the columns are RGB values.
    """
    # The image needs to contain 3 channels...
    result_image = img.reshape((np.multiply(*img.shape[:2]), 3))
    return result_image


# generate k-means image for GUI machine learning setting preview box
# @param image: the RGB image to cluster colours in
# @param aspect_ratio: the desired aspect ratio of the output image
# @param n_clusters: the number of k-means clusters to use
def kmeans(image, aspect_ratio, n_clusters=8):
    # Prepare the RGB image, rescale it to reduce the computation
    image_rescale = img_as_ubyte(rescale(image, 0.25))  # 25% of the original image
    # Reshape the image into three R,G,B arrays
    image_rescale_ary = flatten_img(image_rescale)
    # cluster the pixel intensities in RGB
    clt = KMeans(n_clusters=n_clusters)
    clt.fit(image_rescale_ary)  # Fit with RGB arrays
    # get the number of cluster groups (should be same as n_clusters)
    n_groups = len(np.unique(clt.labels_))
    # get the number of bin labels
    numLabels = np.arange(0, n_groups + 1)
    # create histogram of frequencies of each colour cluster in the image
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)
    # normalise the histogram
    hist = hist.astype("float")
    hist /= hist.sum()  # normalisation
    # generate a figure
    fig, (ax1) = plt.subplots(ncols=1, nrows=1, figsize=(5 * aspect_ratio, 5))
    # first plot will show line graph detailing how many pixels belong to each cluster group
    ax1.plot(hist, marker="o", linewidth=5, markersize=20)
    ax1.axis('off') # we don't need axis
    fig.tight_layout() # use tight layout
    fig.canvas.draw() # force rendering so that we can retrieve the render to put in numpy matrix
    # store the figure as a numpy matrix (this will allow us to easily concatenate 2 images together)
    out_img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    out_img = out_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # get the height and width of the new image
    out_img_h, out_img_w = out_img.shape[:2]
    # second plot will shows the corresponding colours for each histogram bin (k-means cluster)
    cluster_colours = clt.cluster_centers_
    # compute the height of the colour block image (we want the blocks to be square)
    cluster_colour_img_h = int(out_img_w/float(len(clt.cluster_centers_)))
    #  generate the colour block image
    cluster_colour_img = generate_block_colour_image(cluster_colours, block_size=cluster_colour_img_h)
    # resize the image so that it is the same width as the figure (required to stack vertically)
    cluster_colour_img = cv2.resize(cluster_colour_img,(out_img_w, cluster_colour_img_h))
    # vertically stack the 2 images to create 1 k-mean image to show in GUI
    out_img = np.vstack((out_img, cluster_colour_img))
    # return the output image and the number of pixel groups >= median histogram value
    return out_img, np.sum(hist >= np.median(hist))