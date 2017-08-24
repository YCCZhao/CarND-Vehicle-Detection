# Define a single function that can extract features using hog sub-sampling and make predictions
def get_aug_data(img, car_loc, overlap_min, ystart, ystop, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
           
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            features = []
            
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step

            # Pixel location            
            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Find if current window labeled as car, or noncar             
            label = label_window(xleft, ytop, window, car_loc, overlap_min)         
            
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            features.append(hog_features)
            
            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            features.append(spatial_features)
            hist_features = color_hist(subimg, nbins=hist_bins)
            features.append(hist_features)
                     
    return np.concatenate(features), label

def label_window(xleft, ytop, window, car_loc, overlap_min):
    qualified_window = [car for car in car_loc \
                        if (xleft>=car_loc[0]-window and x_left<=car_loc[2]+window)\
                           and (ytop>=car_loc[1]-window and ytop<=car_loc[3]+window)]
    for car in qualified_window:
        x_min = max(x_left, car[0])
        y_min = max(ytop, car[1])
        x_max = min(xleft+window, car[2])
        y_max = min(ytop+window, car[3])

        overlap_percent = (x_max-x_min)*(y_max-y_min)/window**2
        if overlap_percent >= overlap_min:
            return 1
    return 0