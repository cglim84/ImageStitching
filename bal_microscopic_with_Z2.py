# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 17:32:49 2021

@author: cse
    2021-01-08 (feature points outside of overlapping band)
"""

import cv2
import numpy as np

def horizontal_band_features(folder, file0, file1, overlapping_ratio=20, top = 0, bottom=30, left=20, right=20):
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
    left_size = int(img0.shape[1] * left / 100)
    right_size = int(img1.shape[1] * right / 100)

    img0right = img0[top_size:img0.shape[0]-bottom_size,img0.shape[1]-band_size:img0.shape[1]]
    img1left = img1[top_size:img1.shape[0]-bottom_size,0:band_size]
    img0rest = img0[top_size:img0.shape[0]-bottom_size,left_size:img0.shape[1]-band_size]
    img1rest = img1[top_size:img1.shape[0]-bottom_size,band_size:img1.shape[1]-right_size]
    
    orb = cv2.ORB_create(nfeatures=100)
    
    kp0right, desc0right = orb.detectAndCompute(img0right, None)
    kp1left, desc1left = orb.detectAndCompute(img1left, None)
    kp0rest, desc0rest = orb.detectAndCompute(img0rest, None)
    kp1rest, desc1rest = orb.detectAndCompute(img1rest, None)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    
    pts0rest = np.float32([k.pt for k in kp0rest]).reshape(-1,2)
    pts1rest = np.float32([k.pt for k in kp1rest]).reshape(-1,2)
    
    pts0rest = cv2.cornerSubPix(img0rest, pts0rest, (2,2), (-1,-1), criteria)
    pts1rest = cv2.cornerSubPix(img1rest, pts1rest, (2,2), (-1,-1), criteria)
    
    # Back to original coordinates
    pts0rest += [left_size, top_size]
    pts1rest += [band_size, top_size]

    for i in range(pts0rest.shape[0]):
        kp0rest[i].pt = tuple(pts0rest[i,:]);
        
    for i in range(pts1rest.shape[0]):
        kp1rest[i].pt = tuple(pts1rest[i,:]);

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
    for kp in kp0right :
        kp.pt = tuple(kp.pt + np.array([img0.shape[1] - band_size, top_size]))
    for kp in kp1left :
        kp.pt = tuple(kp.pt + np.array([0, top_size]))
        
    pts0right += [img0.shape[1] - band_size, top_size]
    pts1left += [0, top_size]

    for i in range(pts0right.shape[0]):
        kp0right[pts0indices[i]].pt = tuple(pts0right[i,:]);
        
    for i in range(pts1left.shape[0]):
        kp1left[pts1indices[i]].pt = tuple(pts1left[i,:]);

    final_matches = sorted(new_matches, key = lambda x:x.distance)
    
    dbg_img = cv2.drawMatches(img0, kp0right, img1, kp1left, final_matches, None,
                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('sorted matches w/ original images', dbg_img[:,:,[2,1,0]])
    cv2.waitKey()
    cv2.destroyAllWindows()
      
    return pts0right, pts1left, kp0right, kp1left, img0.shape, img1.shape, final_matches, kp0rest, kp1rest, pts0rest, pts1rest

def vertical_band_features(folder, file0, file1, overlapping_ratio=30, left = 0, right=20, top=30, bottom=30):
    if(folder == ''):
        folder = 'E:/Project Data/RFP_TEST/'
    if(file0 == '') :
        file0 = 'test_0001_0001_C.tiff'
    if(file1 == '') :
        file1 = 'test_0001_0004_C.tiff'
    
    print('==== 1st file: ', file0, '2nd file: ', file1)
    
    img0 = cv2.imread(folder + file0, cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(folder + file1, cv2.IMREAD_GRAYSCALE)
    
    band_size = int(img0.shape[0] * overlapping_ratio / 100)
    left_size = int(img0.shape[1] * left / 100)
    right_size = int(img0.shape[1] * right / 100)
    top_size = int(img0.shape[0] * top / 100)
    bottom_size = int(img1.shape[0] * bottom / 100)
    
    img0bottom = img0[img0.shape[0]-band_size:img0.shape[0],left_size:img0.shape[1]-right_size]
    img1top = img1[0:band_size,left_size:img1.shape[1]-right_size]
    img0rest = img0[top_size:img0.shape[0]-band_size,left_size:img0.shape[1]-right_size]
    img1rest = img1[band_size:img1.shape[0]-bottom_size,left_size:img1.shape[1]-right_size]
    
    orb = cv2.ORB_create(nfeatures=100)
    
    kp0bottom, desc0bottom = orb.detectAndCompute(img0bottom, None)
    kp1top, desc1top = orb.detectAndCompute(img1top, None)
    kp0rest, desc0rest = orb.detectAndCompute(img0rest, None)
    kp1rest, desc1rest = orb.detectAndCompute(img1rest, None)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    
    pts0rest = np.float32([k.pt for k in kp0rest]).reshape(-1,2)
    pts1rest = np.float32([k.pt for k in kp1rest]).reshape(-1,2)
    
    pts0rest = cv2.cornerSubPix(img0rest, pts0rest, (2,2), (-1,-1), criteria)
    pts1rest = cv2.cornerSubPix(img1rest, pts1rest, (2,2), (-1,-1), criteria)
    
    # Back to original coordinates
    pts0rest += [left_size, top_size]
    pts1rest += [left_size, band_size]

    for i in range(pts0rest.shape[0]):
        kp0rest[i].pt = tuple(pts0rest[i,:]);
        
    for i in range(pts1rest.shape[0]):
        kp1rest[i].pt = tuple(pts1rest[i,:]);

    # initial matches    
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, False)
    matches = matcher.match(desc0bottom, desc1top)
    
    pts0bottom = np.float32([kp0bottom[m.queryIdx].pt for m in matches]).reshape(-1,2)
    pts1top = np.float32([kp1top[m.trainIdx].pt for m in matches]).reshape(-1,2)
    
    # RANSAC filtering
    H, mask = cv2.findHomography(pts0bottom, pts1top, cv2.RANSAC, 3.0)
    
    new_matches = [m for i,m in enumerate(matches) if mask[i]]
    print('# of new matches: ', np.size(new_matches))

    dbg_img = cv2.drawMatches(img0bottom, kp0bottom, img1top, kp1top, new_matches, None,
                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('final matches', dbg_img[:,:,[2,1,0]])
#    cv2.waitKey()
#    cv2.destroyAllWindows()

    # Sub-Pixel position
    pts0bottom = np.float32([kp0bottom[m.queryIdx].pt for m in new_matches]).reshape(-1,2)
    pts1top = np.float32([kp1top[m.trainIdx].pt for m in new_matches]).reshape(-1,2)
    
    pts0indices = [m.queryIdx for m in new_matches]
    pts1indices = [m.trainIdx for m in new_matches]
    
    pts0bottom = cv2.cornerSubPix(img0bottom, pts0bottom, (1,1), (-1,-1), criteria)
    pts1top = cv2.cornerSubPix(img1top, pts1top, (1,1), (-1,-1), criteria)
    
    total_error = 0.0
    for i in range(pts0bottom.shape[0]):
        err = pts0bottom[i,:] - kp0bottom[pts0indices[i]].pt
        error = np.sum(np.sqrt(err*err))
        total_error += error
        if error > 0.25:
            print("err > 0.25:", error, i, pts0indices[i], pts0bottom[i,:], kp0bottom[pts0indices[i]].pt)
    print("total error @ img0: ", total_error)
    
    total_error = 0.0
    for i in range(pts1top.shape[0]):
        err = pts1top[i,:] - kp1top[pts1indices[i]].pt
        error = np.sum(np.sqrt(err*err))
        total_error += error
        if error > 0.25:
            print("err > 0.25:", error, i, pts1indices[i], pts1top[i,:], kp1top[pts1indices[i]].pt)
    print("total error @ img1: ", total_error)

    # Back to original coordinates
    for kp in kp0bottom :
        kp.pt = tuple(kp.pt + np.array([left_size, img0.shape[0] - band_size]))
    for kp in kp1top :
        kp.pt = tuple(kp.pt + np.array([left_size, 0]))
        
    pts0bottom += [left_size, img0.shape[0] - band_size]
    pts1top += [left_size, 0]

    for i in range(pts0bottom.shape[0]):
        kp0bottom[pts0indices[i]].pt = tuple(pts0bottom[i,:]);
        
    for i in range(pts1top.shape[0]):
        kp1top[pts1indices[i]].pt = tuple(pts1top[i,:]);

    final_matches = sorted(new_matches, key = lambda x:x.distance)

    dbg_img = cv2.drawMatches(img0, kp0bottom, img1, kp1top, final_matches, None,
                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('final matches w/ original images', dbg_img[:,:,[2,1,0]])
    cv2.waitKey()
    cv2.destroyAllWindows()
      
    return pts0bottom, pts1top, kp0bottom, kp1top, img0.shape, img1.shape, final_matches, kp0rest, kp1rest, pts0rest, pts1rest

def build_bal_data() :
    pts1right, pts2left, kp1right, kp2left, img1size, img2size, matches12, kp12rest, kp21rest, pts12rest, pts21rest = horizontal_band_features(
            'E:/Project Data/RFP_TEST/','test_0001_0001_C.tiff','test_0001_0002_C.tiff', 20, 0, 30, 0, 0)
    pts4right, pts5left, kp4right, kp5left, img4size, img5size, matches45, kp45rest, kp54rest, pts45rest, pts54rest = horizontal_band_features(
            'E:/Project Data/RFP_TEST/', 'test_0001_0004_C.tiff', 'test_0001_0005_C.tiff', 20, 30, 0, 0, 0)
    
    pts1bottom, pts4top, kp1bottom, kp4top, img1size, img4size, matches14, kp14rest, kp41rest, pts14rest, pts41rest = vertical_band_features(
            'E:/Project Data/RFP_TEST/', 'test_0001_0001_C.tiff', 'test_0001_0004_C.tiff', 30, 0, 20, 0, 0)
    pts2bottom, pts5top, kp2bottom, kp5top, img2size, img5size, matches25, kp25rest, kp52rest, pts25rest, pts52rest = vertical_band_features(
            'E:/Project Data/RFP_TEST/', 'test_0001_0002_C.tiff', 'test_0001_0005_C.tiff', 30, 20, 0, 0, 0)
      
    n_cameras = 4
    n_points = pts1right.shape[0] + pts4right.shape[0] + pts1bottom.shape[0] + pts2bottom.shape[0] + \
        pts12rest.shape[0] + pts45rest.shape[0] + pts21rest.shape[0] + pts54rest.shape[0]
    n_observations = pts1right.shape[0] + pts2left.shape[0] + pts4right.shape[0] + pts5left.shape[0] + \
        pts1bottom.shape[0] + pts4top.shape[0] + pts2bottom.shape[0] + pts5top.shape[0] + \
        pts12rest.shape[0] + pts45rest.shape[0] + pts21rest.shape[0] + pts54rest.shape[0]
        
    file = open("data.txt", 'w')
    str = "%d " % n_cameras
    str += "%d " % n_points
    str += "%d\n" % n_observations
    file.writelines(str)
    
    # observations
    for i in range(pts1right.shape[0]):
        str = "%d " % 0     # camera index
        str += "%d " % i    # point index
        str += "%f %f\n" % (pts1right[i,0], pts1right[i,1])
        file.writelines(str)
        
    for i in range(pts2left.shape[0]):
        str = "%d " % 1     # camera index
        str += "%d " % i    # point index
        str += "%f %f\n" % (pts2left[i,0], pts2left[i,1])
        file.writelines(str)
        
    total_points = pts1right.shape[0]
    for i in range(pts4right.shape[0]):
        str = "%d " % 2     # camera index
        str += "%d " % (i + total_points)    # point index
        str += "%f %f\n" % (pts4right[i,0], pts4right[i,1])
        file.writelines(str)
        
    for i in range(pts5left.shape[0]):
        str = "%d " % 3     # camera index
        str += "%d " % (i + total_points)   # point index
        str += "%f %f\n" % (pts5left[i,0], pts5left[i,1])
        file.writelines(str)
        
    total_points += pts4right.shape[0]
    for i in range(pts1bottom.shape[0]):
        str = "%d " % 0     # camera index
        str += "%d " % (i + total_points)     # point index
        str += "%f %f\n" % (pts1bottom[i,0], pts1bottom[i,1])
        file.writelines(str)
        
    for i in range(pts4top.shape[0]):
        str = "%d " % 2     # camera index
        str += "%d " % (i + total_points)     # point index
        str += "%f %f\n" % (pts4top[i,0], pts4top[i,1])
        file.writelines(str)
        
    total_points += pts1bottom.shape[0]
    for i in range(pts2bottom.shape[0]):
        str = "%d " % 1     # camera index
        str += "%d " % (i + total_points)    # point index
        str += "%f %f\n" % (pts2bottom[i,0], pts2bottom[i,1])
        file.writelines(str)
        
    for i in range(pts5top.shape[0]):
        str = "%d " % 3     # camera index
        str += "%d " % (i + total_points)     # point index
        str += "%f %f\n" % (pts5top[i,0], pts5top[i,1])
        file.writelines(str)
    
    total_points += pts2bottom.shape[0]
    for i in range(pts12rest.shape[0]):
        str = "%d " % 0     # camera index
        str += "%d " % (i + total_points) # point index
        str += "%f %f\n" % (pts12rest[i,0], pts12rest[i,1])
        file.writelines(str)
        
    total_points += pts12rest.shape[0]
    for i in range(pts45rest.shape[0]):
        str = "%d " % 2      # camera index
        str += "%d " % (i + total_points) # point index
        str += "%f %f\n" % (pts45rest[i,0], pts45rest[i,1])
        file.writelines(str)
        
    total_points += pts45rest.shape[0]
    for i in range(pts21rest.shape[0]):
        str = "%d " % 1      # camera index
        str += "%d " % (i + total_points) # point index
        str += "%f %f\n" % (pts21rest[i,0], pts21rest[i,1])
        file.writelines(str)
        
    total_points += pts21rest.shape[0]
    for i in range(pts54rest.shape[0]):
        str = "%d " % 3     # camera index
        str += "%d " % (i + total_points) # point index
        str += "%f %f\n" % (pts54rest[i,0], pts54rest[i,1])
        file.writelines(str)
        
    total_points += pts54rest.shape[0]
    
    # 9 camera parameters
    file.writelines("0.0\n") # camera 0
    file.writelines("0.0\n")
    file.writelines("0.0\n")
    file.writelines("0.0\n")
    file.writelines("0.0\n")
    file.writelines("0.0\n")
    file.writelines("1.0\n")
    file.writelines("0.0\n")
    file.writelines("0.0\n")

    file.writelines("0.0\n") # camera 1
    file.writelines("0.0\n")
    file.writelines("0.0\n")
    file.writelines("0.0\n")
    file.writelines("0.0\n")
    file.writelines("0.0\n")
    file.writelines("1.0\n")
    file.writelines("0.0\n")
    file.writelines("0.0\n")

    file.writelines("0.0\n") # camera 2
    file.writelines("0.0\n")
    file.writelines("0.0\n")
    file.writelines("0.0\n")
    file.writelines("0.0\n")
    file.writelines("0.0\n")
    file.writelines("1.0\n")
    file.writelines("0.0\n")
    file.writelines("0.0\n")

    file.writelines("0.0\n") # camera 3
    file.writelines("0.0\n")
    file.writelines("0.0\n")
    file.writelines("0.0\n")
    file.writelines("0.0\n")
    file.writelines("0.0\n")
    file.writelines("1.0\n")
    file.writelines("0.0\n")
    file.writelines("0.0\n")
     
    # points
    for i in range(pts1right.shape[0]):
        str = "%f %f %f\n" % (pts1right[i,0], pts1right[i,1], 1.0)
        file.writelines(str)
        
    offsetX = 0
    offsetY = int(img1size[0] - img1size[0] * 30 / 100)
    for i in range(pts4right.shape[0]):
        str = "%f %f %f\n" % (pts4right[i,0] + offsetX, pts4right[i,1] + offsetY, 1.0)
        file.writelines(str)
        
    for i in range(pts1bottom.shape[0]):
        str = "%f %f %f\n" % (pts1bottom[i,0], pts1bottom[i,1], 1.0)
        file.writelines(str)
    
    offsetX = int(img1size[1] - img1size[1] * 20 / 100)
    offsetY = 0
    for i in range(pts2bottom.shape[0]):
        str = "%f %f %f\n" % (pts2bottom[i,0] + offsetX, pts2bottom[i,1] + offsetY, 1.0)
        file.writelines(str)
    
    offsetX = 0
    offsetY = 0
    for i in range(pts12rest.shape[0]):
        str = "%f %f %f\n" % (pts12rest[i,0] + offsetX, pts12rest[i,1] + offsetY, 1.0)
        file.writelines(str)
        
    offsetX = 0
    offsetY = int(img1size[0] - img1size[0] * 30 / 100)
    for i in range(pts45rest.shape[0]):
        str = "%f %f %f\n" % (pts45rest[i,0] + offsetX, pts45rest[i,1] + offsetY, 1.0)
        file.writelines(str)
        
    offsetX = int(img1size[1] - img1size[1] * 20 / 100)
    offsetY = 0
    for i in range(pts21rest.shape[0]):
        str = "%f %f %f\n" % (pts21rest[i,0] + offsetX, pts21rest[i,1] + offsetY, 1.0)
        file.writelines(str)
        
    offsetX = int(img1size[1] - img1size[1] * 20 / 100)
    offsetY = int(img1size[0] - img1size[0] * 30 / 100)
    for i in range(pts54rest.shape[0]):
        str = "%f %f %f\n" % (pts54rest[i,0] + offsetX, pts54rest[i,1] + offsetY, 1.0)
        file.writelines(str)
        
    print('# of points: ', n_points, total_points)
          
    file.close()
    
    return

def read_bal_data(file_name):
    """
    20-12-09 https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
    """
    file = open(file_name, "rt")
    n_cameras, n_points, n_observations = map(int, file.readline().split())
    
    camera_indices = np.empty(n_observations, dtype=int)
    point_indices = np.empty(n_observations, dtype=int)
    points_2d = np.empty((n_observations, 2))
    
    for i in range(n_observations):
        camera_index, point_index, x, y = file.readline().split()
        camera_indices[i] = int(camera_index)
        point_indices[i] = int(point_index)
        points_2d[i] = [float(x), float(y)]
        
    camera_params = np.empty(n_cameras * 9)
    for i in range(n_cameras * 9):
        camera_params[i] = float(file.readline())
    camera_params = camera_params.reshape((n_cameras, -1))
    
    points_3d = np.empty(n_points * 3)
    for i in range(n_points):
        x, y, z = file.readline().split()
        points_3d[3*i] = float(x)
        points_3d[3*i+1] = float(y)
        points_3d[3*i+2] = float(z)
    points_3d = points_3d.reshape((n_points, -1))
        
    file.close()
    
    return camera_params, points_3d, camera_indices, point_indices, points_2d

def rotate(points, rot_vecs):
    """Rotate points by given rotation vectors.
    
    Rodrigues' rotation formular is used.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    return cos_theta * points + sin_theta * np.cross(v, points) + \
        dot * (1 - cos_theta) * v
        
def project(points, camera_params):
    """Convert 3-D points to 2-D by projecting onto images."""
    points_proj = rotate(points, camera_params[:,:3])
    points_proj += camera_params[:, 3:6]
    points_proj = points_proj[:, :2] / points_proj[:, 2, np.newaxis]
#    points_proj = points_proj[:,:2]
    f = camera_params[:, 6]
    k1 = camera_params[:, 7]
    k2 = camera_params[:, 8]
    n = np.sum(points_proj**2, axis=1)
    r = 1 + k1 * n + k2 * n ** 2
    points_proj *= (r * f)[:, np.newaxis]
    
    return points_proj

def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    """Compute residuals.
    
    'params' contains camera parameters and 3-D coordinates.
    """
    camera_params = params[:n_cameras * 9].reshape((n_cameras, 9))
    points_3d = params[n_cameras * 9:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices])
    
    return (points_proj - points_2d).ravel()

from scipy.sparse import lil_matrix

def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 9 + n_points * 3
    A = lil_matrix((m,n), dtype=int)
    
    i = np.arange(camera_indices.size)
    for s in range(9):
        A[2 * i, camera_indices * 9 + s] = 1
        A[2 * i + 1, camera_indices * 9 + s] = 1
        
    for s in range(3):
        A[2 * i, n_cameras * 9 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 9 + point_indices * 3 + s] = 1
        
    return A

def test():
    camera_params, points_3d, camera_indices, point_indices, points_2d = \
    read_bal_data('data.txt')
    
    n_cameras = camera_params.shape[0]
    n_points = points_3d.shape[0]
    
    n = 9 * n_cameras + 3 * n_points
    m = 2 * points_3d.shape[0]
    
    print("n_cameras: {}".format(n_cameras))
    print("n_points: {}".format(n_points))
    print("Total number of parameters: {}".format(n))
    print("Total number of residuals: {}".format(m))
    
    import matplotlib.pyplot as plt

    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
    
    f0 = fun(x0, n_cameras, n_points, camera_indices, point_indices, points_2d)
    
    plt.plot(f0)
    
    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)
    
    import time
    from scipy.optimize import least_squares
    print('start: ', time.strftime("%c"))
    res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                    args=(n_cameras, n_points, camera_indices, point_indices, points_2d))
    print('end: ', time.strftime("%c"))
    f1 = fun(res.x, n_cameras, n_points, camera_indices, point_indices, points_2d)
      
    return res, f1

def camera_calib(camera_params, points_3d, camera_indices, point_indices, points_2d):
    
    from scipy.spatial import Delaunay
    import matplotlib.pyplot as plt
    
    camera_params[:,7:9] = 0.0
    
    c1_indices = np.array([np.where(camera_indices==0)[0]]).reshape((227,))
    p1_indices = point_indices[c1_indices]
    zero_indices = np.zeros((227,),dtype=int)
    points_proj = project(points_3d[p1_indices], camera_params[zero_indices])
    world1pts = np.append(points_proj, np.array([[0,0],[0,1193],[1193,0],[1193,1193]]),axis=0)
    tri1 = Delaunay(world1pts)
    plt.triplot(world1pts[:,0], world1pts[:,1], tri1.simplices), plt.plot(world1pts[:,0], world1pts[:,1], 'o')

    c2_indices = np.array([np.where(camera_indices==1)[0]]).reshape(197,)
    p2_indices = point_indices[c2_indices]
    one_indices = np.ones((197,),dtype=int)
    points_proj = project(points_3d[p2_indices], camera_params[one_indices])
    world2pts = np.append(points_proj, np.array([[0,0],[0,1193],[1193,0],[1193,1193]]),axis=0)
    tri2 = Delaunay(world2pts)
    plt.triplot(world2pts[:,0], world2pts[:,1], tri2.simplices), plt.plot(world2pts[:,0], world2pts[:,1], 'o')

    c4_indices = np.array([np.where(camera_indices==2)[0]]).reshape(230,)
    p4_indices = point_indices[c4_indices]
    one_indices = np.ones((230,),dtype=int)
    points_proj = project(points_3d[p4_indices], camera_params[one_indices*2])
    world4pts = np.append(points_proj, np.array([[0,0],[0,1193],[1193,0],[1193,1193]]),axis=0)
    tri4 = Delaunay(world4pts)
    plt.triplot(world4pts[:,0], world4pts[:,1], tri4.simplices), plt.plot(world4pts[:,0], world4pts[:,1], 'o')

    c5_indices = np.array([np.where(camera_indices==3)[0]]).reshape(200,)
    p5_indices = point_indices[c5_indices]
    one_indices = np.ones((200,),dtype=int)
    points_proj = project(points_3d[p5_indices], camera_params[one_indices*3])
    world5pts = np.append(points_proj, np.array([[0,0],[0,1193],[1193,0],[1193,1193]]),axis=0)
    tri5 = Delaunay(world5pts)
    plt.triplot(world5pts[:,0], world5pts[:,1], tri5.simplices), plt.plot(world5pts[:,0], world5pts[:,1], 'o')

    return    