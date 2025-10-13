import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from osgeo import gdal
from osgeo import osr
# 读取地理坐标数据

map = gdal.Open("1.tif")
width = map.RasterXSize
height = map.RasterYSize
trans = map.GetGeoTransform()
srs = osr.SpatialReference()
srs.ImportFromWkt(map.GetProjectionRef())

srslatlong = srs.CloneGeogCS()
cdtrans = osr.CoordinateTransformation(srs, srslatlong)

if True:
    print("width:", width)
    print("height:", height)
    print(cdtrans)

# 提取特征点


def tracker(
    input="our_data.mp4", output="output.mp4", mapName="1.tif", minMatch=10, gap=16
):
    writer = None
    H = None
    W = None
    nowFrame = 0
    video = cv.VideoCapture(input)
    label = False
    while True:
        frame = video.read()
        frame = frame[1]
        if frame is None:
            print("End of video")
            label = True
            break
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        rgb = cv.resize(rgb, (960, 540))

        if W is None or H is None:
            (H, W) = rgb.shape[:2]
        if writer is None:
            fourcc = cv.VideoWriter_fourcc(*"mp4v")
            writer = cv.VideoWriter(output, fourcc, 30, (W, H), True)
        detector = cv.xfeatures2d.SIFT_create()
        MIN_MATCH = minMatch

        if nowFrame % gap == 0:
            # sift 特征匹配和描述
            img = rgb
            conImg = plt.imread(mapName).copy()
            kp, des = detector.detectAndCompute(img, None)
            conkp, condes = detector.detectAndCompute(conImg, None)
            # Flann 特征匹配
            FLANN_INDEX_KDTREE = 0
            indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=2)
            searchParams = dict(checks=2)
            flann = cv.FlannBasedMatcher(indexParams, searchParams)

            matches = flann.knnMatch(des, condes, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good.append(m)

            if len(good) > MIN_MATCH:
                srcPts = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dstPts = np.float32([conkp[m.trainIdx].pt for m in good]).reshape(
                    -1, 1, 2
                )
                M, status = cv.findHomography(srcPts, dstPts, cv.RANSAC, 5.0)
                if M is not None:
                    print("Homography matrix updated at frame:", nowFrame)
                # 绘制相关标记
                h, w = img.shape[:2]
                pts = np.float32(
                    [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]
                ).reshape(-1, 1, 2)
                dst = cv.perspectiveTransform(pts, M)
                points = np.int32(dst)
                conImg = cv.polylines(conImg, [np.int32(dst)], True, 255, 5, cv.LINE_8)

                moment = cv.moments(points)
                cx = int(moment["m10"] / moment["m00"])
                cy = int(moment["m01"] / moment["m00"])

                conImg = cv.circle(conImg, [cx, cy], 4, (0, 255, 255), 10)
            if writer is not None:
                if conImg.shape[1] != W or conImg.shape[0] != H:
                    conImg = cv.resize(conImg, (W, H))
                conImg = cv.cvtColor(conImg, cv.COLOR_RGB2BGR)
                for i in range(16):
                    writer.write(conImg)
        nowFrame += 1
    writer.release()
    video.release()


tracker()
