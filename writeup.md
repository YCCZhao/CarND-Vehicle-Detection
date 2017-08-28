**Vehicle Detection Project**

The steps of this project are the following:

* Augment training image set
* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images.
* Apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Train a classifier Linear SVM classifier
* Normalize features and randomize a selection for training and testing.
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run your pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.


## Rubric Points
#### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  
&nbsp;

### Writeup / README
---
#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

#### You're reading it!
&nbsp;

### Histogram of Oriented Gradients (HOG)
---

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

**HOG Features**

The code for this step is contained in `get_hog_features()` function which is in lines 48 through 60 of the file called `functions.py`. It is called by `extract_features()` to extract HOG features when HOG features toggle is switched on.

The function uses `skimage.hog()` function to the generate HOG. It takes an image array (range from 0 to 1 in float format), color space, and different `skimage.hog()`  parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`) as inputs. It outputs a vector of HOG features, and a HOG image if the option is turned on. 

**Color Features**

On the top of HOG features, color spatial features and color histogram features are extracted. `extract_features()` function, in lines 65 through 113 of `functions.py`, is called to extract all selected features. It calls `convert_color()` function (lines 7 through 19) to change color space, `bin_spatial()` function (linse 23 through 30) to compute binned color features, `color_hist()` function (lines 34 through 43) to compute color histogram features, and `get_hog_features()` function to compute HOG features.

**Combined**

Once I had the function coded, I then explored different color spaces and different feature parameters:
color space: `cspace`
color spatial size: `spatial_size`
color histogram bins: `hist_bins`
hog orientations: `orient`
hog pixels per cell: `pix_per_cell`
hog cells per block: `cell_per_block`
I grabbed random images from each of the two classes and displayed them to get a feel for what the output looks like.
*Output will be shown in next section*

#### 2. Explain how you settled on your final choice of HOG parameters.

**HOG Features**

I started with the parameters used in the class, and then tried varying parameter once at a time. Below shows some of the setting I explored.

**INSERT IMAGE HERE**

**Color Features**

For color spatial features, I started with `spatial_size=(32, 32)`, and then tried `spatial_size=(16, 16)`. Output with each of the parameters are shown below. 

**INSERT IMAGE HERE**

For color histogram feature, I started with `hist_bins=32`, and then tried `hist_bins=16`. Output with each of the parameters are shown below. 

**INSERT IMAGE HERE**

**Combined**

`YCrCb` color space was chosen because it yielded a better result compared the other two. 

Increasing `orientations` didn't seem benefit much therefore `orientations=8` is used. Similarly decreasing `cells_per_block` didn't seem to affect performance much. To minimize time and space complexity `cells_per_block=(1, 1)` was used.

Deceasing `spatial_size` and `hist_bins` didn't impact the testing accuracy much. In order to improve the computation efficienty smaller number was used to reduce total features.

Here is an example using the `YCrCb` color space, color parameters of `spatial_size=(`, `hist_bins=32`, and     `hist_range=(0, 256)`,  and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

**INSERT IMAGE HERE**


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using `sklearn.svm.LinearSVC`, in lines 82 through 94. Before fitting a SVM, I shuffled the training set and held 20% of them for testing. It had a test accuracy of XX%. Non-linear SVM was tried, but it took much longer time to train and classify. The small accuracy imporvement didn't outweight the longer computation time, so linear SVM was chosen.
&nbsp;

### Sliding Window Search
---
#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search is implemented in `find_cars()` function, lines 117 through 196 of `funtions.py`

`search_boxes = [(1,400,530), (1.5,400,600), (2,400,700),
(2.75,400,700), (2.5,400,700), (3,400,700)]` 

**INSERT IMAGE HERE**

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

**INSERT IMAGE HERE**
&nbsp;

### Video Implementation
---
#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from example images, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the images:

**Here are six frames and their corresponding heatmaps:**

**Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:**

**Here the resulting bounding boxes are drawn onto the last frame in the series:**
&nbsp;

### Discussion
---
#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

