# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 17:32:49 2021

@author: cse
    2021-01-08 (feature points outside of overlapping band)
"""

import cv2
import numpy as np

def horizontal_band_features(folder, file0, file1, overlapping_ratio=20, top = 0, bottom=30):
    if(folder == ''):
        folder = 'E:/Project Data/RFP_TEST/'
    if(file0 == '') :
        file0 = 'test_0001_0001_C.tiff'
    if(file1 == '') :
        file1 = 'test_0001_0002_C.tiff'
    
    print('==== 1st file: ', file0, '2nd file: ', file1)
    
    img0 = cv2.imread(folder + file0, cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(folder + file1, cv2.IMREAD_GRAYSCALE)
    
    band_size = int(img0.shape[1] * overlapping_ratio / 100)
    top_size = int(img0.shape[0] * top / 100)
    bottom_size = int(img0.shape[0] * bottom / 100)
    
    img0right = img0[top_size:img0.shape[0]-bottom_size,img0.shape[1]-band_size:img0.shape[1]]
    img1left = img1[top_size:img1.shape[0]-bottom_size,0:band_size]
    img0rest = img0[top_size:img0.shape[0]-bottom_size,0:img0.shape[1]-band_size]
    img1rest = img1[top_size:img1.shape[0]-bottom_size,band_size:img1.shape[1]]
    
    orb = cv2.ORB_create(nfeatures=100)
    
    kp0right, desc0right = orb.detectAndCompute(img0right, None)
    kp1left, desc1left = orb.detectAndCompute(img1left, None)
    kp0rest, desc0rest = orb.detectAndCompute(img0rest, None)
    kp1rest, desc1rest = orb.detectAndCompute(img1rest, None)

    # initial matches    
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, False)
    matches = matcher.match(desc0right, desc1left)
    
    pts0right = np.float32([kp0right[m.queryIdx].pt for m in matches]).reshape(-1,2)
    pts1left = np.float32([kp1left[m.trainIdx].pt for m in matches]).reshape(-1,2)
    
    # RANSAC filtering
    H, mask = cv2.findHomography(pts0right, pts1left, cv2.RANSAC, 3.0)
    
    new_matches = [m for i,m in enumerate(matches) if mask[i]]
    print('# of new matches: ', np.size(new_matches))

    dbg_img = cv2.drawMatches(img0right, kp0right, img1left, kp1left, new_matches, None,
                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('new matches', dbg_img[:,:,[2,1,0]])
#    cv2.waitKey()
#    cv2.destroyAllWindows()

    # Sub-Pixel position
    pts0right = np.float32([kp0right[m.queryIdx].pt for m in new_matches]).reshape(-1,2)
    pts1left = np.float32([kp1left[m.trainIdx].pt for m in new_matches]).reshape(-1,2)
    
    pts0indices = [m.queryIdx for m in new_matches]
    pts1indices = [m.trainIdx for m in new_matches]
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    
    pts0right = cv2.cornerSubPix(img0right, pts0right, (2,2), (-1,-1), criteria)
    pts1left = cv2.cornerSubPix(img1left, pts1left, (2,2), (-1,-1), criteria)
    
    total_error = 0.0
    for i in range(pts0right.shape[0]):
        err = pts0right[i,:] - kp0right[pts0indices[i]].pt
        error = np.sum(np.sqrt(err*err))
        total_error += error
        if error > 0.25:
            print("err > 0.25:", error, i, pts0indices[i], pts0right[i,:], kp0right[pts0indices[i]].pt)
    print("total error @ img0: ", total_error)
    
    total_error = 0.0
    for i in range(pts1left.shape[0]):
        err = pts1left[i,:] - kp1left[pts1indices[i]].pt
        error = np.sum(np.sqrt(err*err))
        total_error += error
        if error > 0.25:
            print("err > 0.25:", error, i, pts1indices[i], pts1left[i,:], kp1left[pts1indices[i]].pt)
    print("total error @ img1: ", total_error)

    # Back to original coordinates
    pts0right += [img0.shape[1] - band_size, top_size]
    pts1left += [0, top_size]

    for i in range(pts0right.shape[0]):
        kp0right[pts0indices[i]].pt = tuple(pts0right[i,:]);
        
    for i in range(pts1left.shape[0]):
        kp1left[pts1indices[i]].pt = tuple(pts1left[i,:]);

    final_matches = sorted(new_matches, key = lambda x:x.distance)
    
    dbg_img = cv2.drawMatches(img0, kp0right, img1, kp1left, final_matches[:10], None,
                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('1st 10 matches w/ original images', dbg_img[:,:,[2,1,0]])
    cv2.waitKey()
    cv2.destroyAllWindows()
      
    return pts0right, pts1left, kp0right, kp1left, img0.shape, img1.shape, final_matches