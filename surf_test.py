#!/usr/bin/python3
# 2021.22.06 00:00:0 WIB
# 2021.18.06 00:00:00 WIB

"""
KELOMPOK PENGENALAN WAJAH (FACE RECOGNITION) DENGAN
MENGGUNAKAN METODE SURF (SPEEDED UP ROBUST FEATURES)
1. Firmansyah Mukti Wijaya ( 19.1.03.02.0046 )
2. M. Yusuf Khoirul Huda ( 19.1.03.02.0172 )
3. Arip Dwi Cahyono ( 19.1.03.02.0081 )
4. Hendra Tri Kristanto ( 19.1.03.02.0087 )
UNIVERSITAS NUSANTARA PGRI KEDIRI
"""

"""IMPORT OPEN CV dan NUMPY"""
import cv2
import numpy as np
import os
import os.path
from matplotlib import pyplot as plt
# print('OpenCv Version:',cv2.__version__)
MIN_MATCH_COUNT = 4
def match(find, dir_train):
    imgname2 = dir_train
    imgname1 = find
    #imgname1 = "danu.png"
    #imgname2 = "1.png"

    ## (1) Persiapan Data
    img1 = cv2.imread(imgname1)
    img2 = cv2.imread(imgname2)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    ## (2) Membuat surf object
    surf = cv2.SIFT_create()
    # surf = cv2.xfeatures2d.SURF_create()
    
    ## (3) Membuat flann matcher
    matcher = cv2.FlannBasedMatcher(dict(algorithm = 1, trees = 5), {})
    
    ## (4) Detect keypoints and compute keypointer descriptors
    kpts1, descs1 = surf.detectAndCompute(gray1,None)
    kpts2, descs2 = surf.detectAndCompute(gray2,None)

    ## (5) knnMatch to get Top2
    matches = matcher.knnMatch(descs1, descs2, 2)

    # Sort by their distance.
    matches = sorted(matches, key = lambda x:x[0].distance)

    ## (6) Ratio test, to get good matches.
    good = [m1 for (m1, m2) in matches if m1.distance < 0.54 * m2.distance]
    canvas = img2.copy()
    ok = 0
    ## (7) find homography matrix
    if len(good)>MIN_MATCH_COUNT:
        ## (queryIndex for the small object, trainIndex for the scene )
        src_pts = np.float32([ kpts1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kpts2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        #print(src_pts)
        #print(dst_pts)
        ## find homography matrix in cv2.RANSAC using good match points
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        print(matchesMask)
        g = len(matchesMask)
        i = g
        for mm in matchesMask :
            if mm == 0 :
                i = i - 1
        print("JUMLAH = ", i)
        if i == 0 :
            ok == 0
        else :
            #print(M, "XXX")
            ok = 1
    else :
        print( "Not enough matches are found - {}/{}".format(len(good),MIN_MATCH_COUNT))
        matchesMask = None
        ok = 0

    if ok == 1:
        h,w = img1.shape[:2]
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        #print(pts)
        dst = cv2.perspectiveTransform(pts,M)
        cv2.polylines(canvas,[np.int32(dst)],True,(0,255,0),3, cv2.LINE_AA)
        ## (8) drawMatches
        matched = cv2.drawMatches(img1,kpts1,canvas,kpts2,good,None)#,**draw_params)

        ## (9) Crop the matched region from scene
        h,w = img1.shape[:2]
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)   
        dst = cv2.perspectiveTransform(pts,M)
        perspectiveM = cv2.getPerspectiveTransform(np.float32(dst),pts)
        found = cv2.warpPerspective(img2,perspectiveM,(w,h))
        
        p = find.split('/')
        #print(p[1])
        d = dir_train.split('/')
        #print(d)
        dir_match = "match/res_"+p[1]
        #print(dir_match)
        if os.path.isdir(dir_match) :
            print("ada")
        else :
            print("tidak")
            os.mkdir(dir_match)
        match_name = dir_match+"/matched-"+p[1]+"-with-"+d[1]+"+"+d[2]
        found_name = dir_match+"/found-"+p[1]+"-with-"+d[1]+"+"+d[2]
        #print(match_name)
        ## (10) save and display
        cv2.imwrite(match_name, matched)
        cv2.imwrite(found_name, found)
        #cv2.imshow(match_name, matched)
        #cv2.imshow(found_name, found)
        #cv2.waitKey()
        #cv2.destroyAllWindows()

print("SELAMAT DATANG DI FACE RECOGNITION WITH SURD METHOD")
print("Silahkan masukkan dulu data search dan train pada folder yang telah disediakan.")
print("SILAHKAN PILIH FILE Yang Akan Di Match :")
dir_search = os.listdir("search")
print(dir_search)
search = 0
while search == 0:
    find = input("Nama File dan Format yang di Match : ")
    for nf in dir_search:
        a = str(nf)
        if a == find:
            search = 1
    if search == 0:
        print("Nama File Salah")
    else:
        print("File ditemukan")
print("PROSES MATCH /n DATA TRAIN :")
dir_train = os.listdir("train")
for nf in dir_train:
    a = str(nf)
    b = "train/"+a+"/"
    dir_train2 = os.listdir(b)
    print(nf,dir_train2)
    for nf2 in dir_train2:
        c = str(nf2)
        d = "train/"+a+"/"+c
        e = "search/"+find
        match(e,d)
        



