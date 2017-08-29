import numpy as np
import time
import pickle
from sklearn.svm import LinearSVC
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from functions import extract_features
from get_images import get_images

# load images
cars, notcars = get_images()

#Feature Parameters
colorspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 1
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial = 32
histbin = 32
spatial_feat=True
hist_feat=True
hog_feat=True

#Extra image features
t=time.time()
X = []
for idx, file in enumerate(cars):
    car_features = extract_features(file, cspace=colorspace, spatial_size=(spatial, spatial),
                         hist_bins=histbin, hist_range=(0, 256),
                         orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                         hog_channel=hog_channel, spatial_feat=spatial_feat, hist_feat=hist_feat,
                         hog_feat=hog_feat)
    X.append(car_features)
    if idx%100 == 0:
        print(idx)
for idx, file in enumerate(notcars):
    notcar_features = extract_features(file, cspace=colorspace, spatial_size=(spatial, spatial),
                         hist_bins=histbin, hist_range=(0, 256),
                         orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                         hog_channel=hog_channel, spatial_feat=spatial_feat, hist_feat=hist_feat,
                         hog_feat=hog_feat)
    X.append(notcar_features)
    if idx%100 == 0:
        print(idx)
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
scaled_X, y = shuffle(scaled_X, y)
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, 
                                                    test_size=0.2, 
                                                    random_state=rand_state)

print('Using spatial binning of:',spatial,
    'and', histbin,'histogram bins')
if hog_feat:
    print('Using:',orient,'orientations ,',pix_per_cell,
          'pixels per cell, and ', cell_per_block,' cells per block')
print('Feature vector length:', len(X_train[0]))

# Use a linear SVC 
svc = LinearSVC()

# other classifiers tried
#parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
#svc = SVC(kernel='rbf', C=1)
#svc = GridSearchCV(svr, parameters)
#parameters = {'min_samples_split':[10, 50]}
#svr = DecisionTreeClassifier()
#svc = grid_search.GridSearchCV(svr, parameters)

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

# Save trained SVC
clf = {}
clf['svc'] = svc
clf['scaler'] = X_scaler
clf['colorspace'] = colorspace
clf['hog_channel'] = hog_channel
clf['orient'] = orient
clf['pix_per_cell'] = pix_per_cell
clf['cell_per_block'] = cell_per_block
clf['spatial_size'] = spatial
clf['hist_bins'] = histbin

trained_clf = '../Data/vehicle-detection-basic/trained_clf3.p'
with open(trained_clf, mode='wb') as f:
    pickle.dump(clf, f)
