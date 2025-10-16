from matplotlib.colors import to_rgb
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from osgeo import gdal
from osgeo import osr
from joblib import Parallel, delayed


class Tracker:
    def __init__(
        self,
        inputDir: str,
        outputDir: str,
        mapDir: str,
        minMatch: int,
        gap: int,
        coreNumber: int = 1,
    ):
        self.inputDir = inputDir  # 输入视频路径
        self.mapDir = mapDir  # 地图/参考图像路径
        self.minMatch = minMatch  # 最小匹配点数
        self.gap = gap  # 关键帧间隔
        self.outputDir = outputDir  # 输出路径
        self.coreNumber = coreNumber  # 并行处理核心数
        self.inputH = None  # 输入视频高宽
        self.inputW = None  # 输入视频高度
        self.inputF = None  # 输入视频帧数
        self.outputH = None  # 输出视频高宽
        self.outputW = None  # 输出视频高度
        self.video = None  # 输入视频对象
        self.map = None  # 地图/参考图像对象

    def trackFrame(self, frame: np.ndarray):
        # sift特征点检测
        frame = frame[1]
        if frame is None:
            print("invalid frame")
        detector = cv.xfeatures2d.SIFT_create()
        kp, des = detector.detectAndCompute(frame, None)
        kpMap, desMap = detector.detectAndCompute(self.map, None)
        # 特征点匹配
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=2)
        search_params = dict(checks=2)
        flann = cv.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des, desMap, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
        if len(good) > self.minMatch:
            srcPts = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dstPts = np.float32([kpMap[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, mask = cv.findHomography(srcPts, dstPts, cv.RANSAC, 5.0)
        return srcPts, dstPts, M, mask

    def init(self):
        self.video = cv.VideoCapture(self.inputDir)
        self.map = cv.imread(self.mapDir)
        self.inputH = int(self.video.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.inputW = int(self.video.get(cv.CAP_PROP_FRAME_WIDTH))
        self.inputF = int(self.video.get(cv.CAP_PROP_FRAME_COUNT))
        self.outputH = int(self.map.shape[0])
        self.outputW = int(self.map.shape[1])
        print(
            f"Input video info: width {self.inputW}, height {self.inputH}, frame {self.inputF}"
        )

    def drawFrame(self, frame: np.ndarray, srcPts, dstPts, M, mask):
        frame = frame[1]
        h, w = frame.shape[:2]
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(
            -1, 1, 2
        )
        dst = cv.perspectiveTransform(pts, M)
        frame = cv.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3, cv.LINE_AA)
        draw_params = dict(
            matchColor=(0, 255, 0),
            singlePointColor=None,
            matchesMask=mask.ravel().tolist(),
            flags=2,
        )
        # img3 = cv.drawMatches(frame, srcPts, self.map, dstPts, [], None, **draw_params)
        return frame

    def processFrame(self, frame):
        try:
            srcPts, dstPts, M, mask = self.trackFrame(frame)
            outFrame = self.drawFrame(frame, srcPts, dstPts, M, mask)
            return outFrame
        except:
            return None

    def findOutFrameSize(self, OutFrameList):
        maxH = max([frame.shape[0] for frame in OutFrameList])
        maxW = max([frame.shape[1] for frame in OutFrameList])
        return maxH, maxW

    def resizeOutFrame(self, frame, maxH, maxW):
        h, w = frame.shape[:2]
        top = (maxH - h) // 2
        bottom = maxH - h - top
        left = (maxW - w) // 2
        right = maxW - w - left
        resizedFrame = cv.copyMakeBorder(
            frame, top, bottom, left, right, cv.BORDER_CONSTANT, value=[0, 0, 0]
        )
        return resizedFrame

    def process(self):
        self.init()  # 初始化
        toDoList = []
        for i in range(self.inputF):  # 选取间隔帧处理
            frame = self.video.read()
            if i % self.gap == 0:
                toDoList.append(frame)
        results = [self.processFrame(frame) for frame in toDoList]
        print("All frames processed.")
        maxH, maxW = self.findOutFrameSize(results)
        results = [self.resizeOutFrame(frame, maxH, maxW) for frame in results]
        print("All frames resized.")
        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        outVideo = cv.VideoWriter(self.outputDir, fourcc, 20.0, (maxW, maxH))
        for frame in results:
            outVideo.write(frame)
        outVideo.release()

    def __del__(self):
        self.video.release()
        print("Done.")


if __name__ == "__main__":
    inputDir = "our_data.mp4"
    outputDir = "Output.mp4"
    mapDir = "1.tif"
    minMatch = 10
    gap = 5
    coreNumber = 4
    tracker = Tracker(inputDir, outputDir, mapDir, minMatch, gap, coreNumber)
    tracker.process()
