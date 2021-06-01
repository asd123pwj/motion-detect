import cv2 as cv
import numpy as np


def GMM_process(frame, gray_frame, mog, es):
    gray_frame = mog.apply(gray_frame)

    #   膨胀、腐蚀处理，将相近的运动区块连接
    # gray_frame = cv.erode(gray_frame, es, iterations=1)
    # gray_frame = cv.dilate(gray_frame, es, iterations=1)

    for i in range(20):
        gray_frame = cv.dilate(gray_frame, es, iterations=2)
        gray_frame = cv.erode(gray_frame, es, iterations=1)

    #   运动轮廓提取
    contours, hierarchy = cv.findContours(gray_frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #   轮廓画于图片
    for c in contours:
        #   去除小面积变化噪声
        if cv.contourArea(c) < 300:
            continue
        (x, y, w, h) = cv.boundingRect(c)
        cv.rectangle(gray_frame, (x, y), (x + w, y + h), (255, 255, 255), -1)

    gray_frame = cv.dilate(gray_frame, es, iterations=10)

    #   运动轮廓提取
    contours, hierarchy = cv.findContours(gray_frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #   轮廓画于图片
    for c in contours:
        #   去除小面积变化噪声
        if cv.contourArea(c) < 200:
            continue
        (x, y, w, h) = cv.boundingRect(c)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return frame


def get_edge(gray_frame, es, GB_thres, Canny_thres_1, Canny_thres_2):
    gray_frame = cv.GaussianBlur(gray_frame, (GB_thres, GB_thres), 0)
    # gray_frame = cv.bilateralFilter(gray_frame, 10, 175, 5)
    #   差别图像二值化
    # gray_frame = cv.threshold(gray_frame, 150, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
    edge_frame = cv.Canny(gray_frame, Canny_thres_1, Canny_thres_2, True)
    # gray_frame = cv.GaussianBlur(gray_frame, (GB_thres, GB_thres), 0)
    return edge_frame


def get_inter_diff(frame, present_frame, last_frame, es):
    inter_diff = cv.absdiff(last_frame, present_frame)

    #   膨胀、腐蚀处理，将相近的运动区块连接
    inter_diff = cv.erode(inter_diff, es, iterations=1)
    inter_diff = cv.dilate(inter_diff, es, iterations=1)

    for i in range(5):
        inter_diff = cv.dilate(inter_diff, es, iterations=2)
        inter_diff = cv.erode(inter_diff, es, iterations=1)

    #   运动轮廓提取
    contours, hierarchy = cv.findContours(inter_diff, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #   轮廓画于图片
    for c in contours:
        #   去除小面积变化噪声
        if cv.contourArea(c) < 300:
            continue
        (x, y, w, h) = cv.boundingRect(c)
        cv.rectangle(inter_diff, (x, y), (x + w, y + h), (255, 255, 255), -1)

    inter_diff = cv.dilate(inter_diff, es, iterations=5)

    #   运动轮廓提取
    contours, hierarchy = cv.findContours(inter_diff, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #   轮廓画于图片
    for c in contours:
        #   去除小面积变化噪声
        if cv.contourArea(c) < 400:
            continue
        (x, y, w, h) = cv.boundingRect(c)
        cv.rectangle(inter_diff, (x, y), (x + w, y + h), (255, 255, 255), -1)

    inter_diff = cv.dilate(inter_diff, es, iterations=10)
    inter_diff = cv.erode(inter_diff, es, iterations=4)

    #   运动轮廓提取
    contours, hierarchy = cv.findContours(inter_diff, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #   轮廓画于图片
    for c in contours:
        #   去除小面积变化噪声
        if cv.contourArea(c) < 800:
            continue
        (x, y, w, h) = cv.boundingRect(c)
        cv.rectangle(inter_diff, (x, y), (x + w, y + h), (255, 255, 255), -1)

    frame_origin = cv.absdiff(last_frame, last_frame)

    #   运动轮廓提取
    contours, hierarchy = cv.findContours(inter_diff, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #   轮廓画于图片
    for c in contours:
        #   去除小面积变化噪声
        if cv.contourArea(c) < 800:
            continue
        (x, y, w, h) = cv.boundingRect(c)
        cv.rectangle(frame_origin, (x, y), (x + w, y + h), (255, 255, 255), -1)

    es2 = cv.getStructuringElement(cv.MORPH_CROSS, (3, 8))
    frame_origin = cv.dilate(frame_origin, es2, iterations=12)
    es2 = cv.getStructuringElement(cv.MORPH_CROSS, (8, 3))
    frame_origin = cv.dilate(frame_origin, es2, iterations=7)
    es2 = cv.getStructuringElement(cv.MORPH_CROSS, (5, 7))
    frame_origin = cv.erode(frame_origin, es2, iterations=18)
    # frame_origin = cv.erode(frame_origin, es, iterations=4)
    # inter_diff = cv.absdiff(frame_origin, frame)

    #   运动轮廓提取
    contours, hierarchy = cv.findContours(frame_origin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #   轮廓画于图片
    for c in contours:
        #   去除小面积变化噪声
        if cv.contourArea(c) < 200:
            continue
        (x, y, w, h) = cv.boundingRect(c)
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return frame


def main():
    #   视频读取
    cap = cv.VideoCapture('testVideo.mp4')
    #   视频fps读取
    fps = cap.get(cv.CAP_PROP_FPS)
    print(fps)
    #   视频大小读取
    size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
    #   视频保存编码
    fourcc_mp4 = cv.VideoWriter_fourcc(*'mp4v')
    fourcc_avi = cv.VideoWriter_fourcc(*'XVID')
    #   保存视频
    out_detect = cv.VideoWriter('output_detect.mp4', fourcc_mp4, fps, size, True)

    #   形态学模板创建
    # es = cv.getStructuringElement(cv.MORPH_ELLIPSE, (12, 12))
    es = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    print(es)

    GB_thres = 7
    medianBlur_Threshold = 3
    Canny_thres_1 = 20
    Canny_thres_2 = 60

    inter_diff = None

    mog = cv.createBackgroundSubtractorMOG2()
    last_frame = None
    times = 1
    show = 0
    while cap.isOpened():
        #   处理第times帧
        print(times)
        times += 1
        #   读入视频帧
        ret, frame = cap.read()
        #   视频结束，跳出循环
        if frame is None:
            break
        #   转灰度图
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        #   GMM识别
        # GMM_frame = GMM_process(frame, gray_frame.copy(), mog, es)
        #   GMM识别区内不同
        # intra_diff = cv.absdiff(GMM_frame, gray_frame)
        # intra_edge = get_edge(intra_diff, es, GB_thres, Canny_thres_1, Canny_thres_2)

        if last_frame is not None:
            present_edge = get_edge(gray_frame, es, GB_thres, Canny_thres_1, Canny_thres_2)
            last_edge = get_edge(last_frame, es, GB_thres, Canny_thres_1, Canny_thres_2)
            present_GMM = mog.apply(present_edge)
            last_GMM = mog.apply(last_edge)
            inter_diff = get_inter_diff(frame.copy(), present_GMM, last_GMM, es)
            # inter_diff = cv.absdiff(frame, inter_diff)
            # inter_gray_frame = cv.cvtColor(inter_diff, cv.COLOR_BGR2GRAY)
            # GMM_frame = GMM_process(frame, inter_gray_frame.copy(), mog, es)
            # GMM_frame = GMM_process(frame, inter_diff.copy(), mog, es)
        last_frame = gray_frame

        #   图像实时显示
        if show:
            # cv.namedWindow("frame", cv.WINDOW_NORMAL)
            # cv.imshow("frame", frame)
            # cv.namedWindow("gray_frame", cv.WINDOW_NORMAL)
            # cv.imshow("gray_frame", gray_frame)
            # cv.namedWindow("edge", cv.WINDOW_NORMAL)
            # cv.imshow("edge", edge)

            if inter_diff is not None:
                cv.namedWindow("diff", cv.WINDOW_NORMAL)
                cv.imshow("diff", inter_diff)

            #   按键退出
            if cv.waitKey(1) == ord('q'):
                break

        #   处理结果写入
        out_detect.write(inter_diff)
    #   资源释放
    cap.release()
    cv.destroyAllWindows()
    out_detect.release()


if __name__ == '__main__':
    main()
