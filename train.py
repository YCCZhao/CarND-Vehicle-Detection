import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
import glob
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from functions import extract_features


# load images
notcars = glob.glob('../Data/vehicle-detection-basic/non-vehicles/non-vehicles/*.png')
cars = []
cars.extend(glob.glob('../Data/vehicle-detection-basic/vehicles/vehicles/GTI_Far/*.png'))
cars.extend(glob.glob('../Data/vehicle-detection-basic/vehicles/vehicles/GTI_Left/*.png'))
cars.extend(glob.glob('../Data/vehicle-detection-basic/vehicles/vehicles/GTI_MiddleClose/*.png'))
cars.extend(glob.glob('../Data/vehicle-detection-basic/vehicles/vehicles/GTI_RIght/*.png'))
cars.extend(glob.glob('../Data/vehicle-detection-basic/vehicles/vehicles/KITTI_extracted/*.png'))

#Feature Parameters
colorspace = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial = 32
histbin = 16

cars_ = cars[:5]
notcars_ = notcars[:5]

t=time.time()
X = []
for file in cars:
    car_features = extract_features(file, cspace='RGB', spatial_size=(32, 32),
                         hist_bins=16, hist_range=(0, 256),
                         orient=9, pix_per_cell=8, cell_per_block=2,
                         hog_channel="ALL", spatial_feat=True, hist_feat=True,
                         hog_feat=True)
    X.append(car_features)
for file in notcars:
    notcar_features = extract_features(file, cspace='RGB', spatial_size=(32, 32),
                         hist_bins=16, hist_range=(0, 256),
                         orient=9, pix_per_cell=8, cell_per_block=2,
                         hog_channel="ALL", spatial_feat=True, hist_feat=True,
                         hog_feat=True)
    X.append(notcar_features)

X = np.array(X)
print(X.shape)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to extract HOG features...')
                  
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(cars)), np.zeros(len(notcars))))
print(y.shape)

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, 
                                                    test_size=0.2, 
                                                    random_state=rand_state)

print('Using spatial binning of:',spatial,
    'and', histbin,'histogram bins')
if hog_channel is not None:
    print('Using:',orient,'orientations',pix_per_cell,
          'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))

# Use a linear SVC 
svc = LinearSVC()

# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')

# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

# Check the prediction time for a single sample
t=time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

