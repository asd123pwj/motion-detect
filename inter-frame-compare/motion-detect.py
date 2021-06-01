import cv2 as cv
import numpy as np
import argparse


def model_process(gray_frame, model, noise_thres):
    """
    使用模型检测运动区域，并对检测的区域去噪、连接
    :param gray_frame: 当前帧的灰度图
    :param model: 检测模型
    :param noise_thres: 被视作噪声面积的阈值，视频分辨率越高，该值应越大
    :return: 运动区域
    """
    model_frame = model.apply(gray_frame)
    contours, hierarchy = cv.findContours(model_frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for c in contours:
        #   去除小面积变化噪声
        if cv.contourArea(c) < noise_thres * 0.5:
            continue
        (x, y, w, h) = cv.boundingRect(c)
        cv.rectangle(model_frame, (x, y), (x + w, y + h), (255, 255, 255), -1)

    #   连接相近区域并调整形状
    es2 = cv.getStructuringElement(cv.MORPH_CROSS, (3, 5))
    model_frame = cv.dilate(model_frame, es2, iterations=1)

    contours, hierarchy = cv.findContours(model_frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for c in contours:
        #   去除小面积变化噪声
        if cv.contourArea(c) < noise_thres:
            continue
        (x, y, w, h) = cv.boundingRect(c)
        cv.rectangle(model_frame, (x, y), (x + w, y + h), (255, 255, 255), -1)

    #   去除小面积噪声
    contours, hierarchy = cv.findContours(model_frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    moderate_contours = []
    for c in contours:
        if cv.contourArea(c) > noise_thres * 2:
            moderate_contours.append(c)

    return moderate_contours


def get_contours_area(contours):
    """
    计算区域集的总面积
    :param contours: 区域集
    :return: 区域集总面积
    """
    area_sum = 0
    for c in contours:
        area_sum += cv.contourArea(c)
    return area_sum


def is_contours_intersect(contour1, contour2):
    """
    判断两区域是否有交集
    :param contour1: 区域1
    :param contour2: 区域2
    :return: 区域有交集则True，否则False
    """
    (x1, y1, w1, h1) = cv.boundingRect(contour1)
    (x2, y2, w2, h2) = cv.boundingRect(contour2)
    if x1 + w1 < x2 or x2 + w2 < x1:
        return False
    if y1 + h1 < y2 or y2 + h2 < y1:
        return False
    return True


def is_big_motion(queue_contours, area_thres, slope_thres, num_thres):
    """
    判断是否出现运动区域巨大变化的帧
    对于斜率的判断，我认为，在不到一秒的的时间内，开关灯的影响会让变化区域先变大，再变小，而正常运动，则是要么变大，要么变小。但也可能会因为时间过于短暂或拍摄设备的问题让灯光残留时间过久，导致这个判断不准确。如果开关灯的变化刚好在关联帧内，那么理论上阈值取得高点比较好，反之则取低一点。
    对于面积的判断，开光灯必然使得被视作运动的区域的面积变大，因此可以将变化幅度过大的帧视作出现巨大变化的帧。
    对于区域数目的判断，开关灯结束后的一段时间，运动面积趋于平稳，但个数可能起伏不定，因此如果变化过大，可能是噪声。
    对这三个参数的调整，可以将显示日志的参数打开，观察判断错误的帧中，这三个参数是如何变化的。
    值得一提的是，如果关联比例relevance与关联时间relevance_time取得好，可以不判断是否要处理，而是都进行前后帧关联处理，这样可以减少一些错误噪声，但是会降低对运动的敏感性。
    :param queue_contours: 前后一系列帧的区域的集
    :param area_thres: 面积变化阈值。
    :param slope_thres: 面积变化斜率阈值
    :param num_thres: 区域数目变化阈值
    :return: 日志与判断真假结果
    """
    #   直接判断需要处理
    if area_thres == 0 or slope_thres == 0 or num_thres == 0:
        return 'Always process.', True
    #   初始化
    q_len = len(queue_contours)
    contours = queue_contours[q_len//2]
    c_len = len(contours)
    area_average = 0
    area = get_contours_area(contours)
    slope = 0
    num = 0
    confidence_slope = False
    confidence_area = False
    confidence_num = False
    area_list = []
    for pos in range(q_len):
        area_list.append(get_contours_area(queue_contours[pos]))
    #   面积变化斜率判断
    for pos in range(q_len // 2 + 1):
        if area_list[pos] < 1.5 * area_list[pos + 1]:
            slope += 1
    for pos in range(q_len // 2 + 1, q_len):
        if area_list[pos - 1] > 1.5 * area_list[pos]:
            slope += 1
    if slope / q_len > slope_thres:
        confidence_slope = True
    #   面积变化大小判断
    for pos in range(q_len):
        area_average += area_list[pos]
    area_average = area_average / q_len
    if area_average and (area > area_average * area_thres or area * area_thres < area_average):
        confidence_area = True
    #   区域数目变化幅度判断
    for pos in range(q_len):
        num += len(queue_contours[pos])
    num /= q_len
    if c_len > num * num_thres or c_len * num_thres < num:
        confidence_num = True
    #   日志
    log = 'Area:' + str(area // 1) + '  Area_average:' + str(area_average // 1)
    log = log + '  Slope_conf: ' + str(confidence_slope) + '  Area_conf:' + str(confidence_area)
    log = log + '  Num_conf: ' + str(confidence_num)
    if confidence_area or confidence_slope or confidence_num:
        return log, True
    else:
        return log, False


def get_contours_intersect(contours, contours_other):
    """
    获取contours区域集中与contours_other有交集的区域
    :param contours: 当前帧被检测到的运动区域
    :param contours_other: 非当前帧的运动区域
    :return: 返回与contours等长的列表，列表中元素值为1，代表该contour与contours_other有交集
    """
    c_len = len(contours)
    contours_intersect = np.zeros(c_len)
    for c1 in range(c_len):
        for c2 in contours_other:
            if is_contours_intersect(contours[c1], c2):
                contours_intersect[c1] = 1
                break
    return contours_intersect


def get_better_contours(queue_contours,  area_thres, slope_thres, num_thres, relevance):
    """
    根据前后帧区域判断是否出现大面积变动，并与前后帧关联处理。
    我认为，一个正常的运动物体，一般运动持续时间比灯光变化的时间要长，因此，如果关联帧中某区域一直运动，那么可以认为该区域属于运动区域，relevance越大，运动区域的识别准确率越高
    :param queue_contours:  前后一系列帧的区域的列表
    :param relevance:   关联比例，关联比例越小，被视作正常运动的帧越多，取值范围[0,1]
    :param area_thres:  面积变化阈值
    :param slope_thres: 面积变化斜率阈值
    :param num_thres: 区域数目变化阈值
    :return: 更可能是正常运动的区域
    """
    #   初始化
    q_len = len(queue_contours)
    c_len = len(queue_contours[q_len//2])
    contours_related = queue_contours[q_len//2]
    num_pre_contours = len(contours_related)
    confidence_contours = np.zeros(c_len)
    #   区域变化情况检测
    log, is_big = is_big_motion(queue_contours, area_thres, slope_thres, num_thres)
    logs = [log]
    #   区域相关性检测
    if is_big:
        for pos in range(q_len):
            if pos != q_len//2:
                contours_intersect = get_contours_intersect(contours_related, queue_contours[pos])
                confidence_contours = np.sum([confidence_contours, contours_intersect], axis=0)
        #   相关区域整理
        contours_related = []
        for pos in range(c_len):
            if confidence_contours[pos] > relevance * c_len:
                contours_related.append(queue_contours[q_len//2][pos])
        #   日志
        log2 = 'big change: '
        queue_contours[q_len//2] = contours_related
        log2 = log2 + ' contours number: after process:' + str(len(contours_related))
        log2 = log2 + ' before: ' + str(num_pre_contours)
        logs.append(log2)
    return logs, contours_related


def enque(q, ele, max_len):
    """
    入队
    :param q: 队列
    :param ele: 元素
    :param max_len: 队列最大长度
    :return: 无返回值，直接修改队列
    """
    q.append(ele)
    if len(q) == max_len + 1:
        q.pop(0)


def add_log(logs, frame, size):
    """
    根据视频大小动态设置日志字体大小，为帧添加文本
    :param logs: 文本集，每个元素为一个字符串
    :param frame: 待作画区域
    :param size: 视频大小
    :return: 无返回值，在frame上直接修改
    """
    width = size[0] // 50
    per_height = size[1] // 40
    font_size = size[0] / 1000
    height = per_height
    for log in logs:
        height += per_height
        cv.putText(frame, log, (width, height), cv.FONT_HERSHEY_PLAIN, font_size, (255, 0, 0), 2)


def motion_detect(src, area_thres, slope_thres, num_thres, relevance, relevance_time, noise_proportion, show_frame, show_log, output, skip_time):
    """
    :param src: 待检测视频路径
    :param area_thres: 面积变化阈值，非负数
    :param slope_thres: 面积变化斜率阈值，取值范围[0,1]
    :param num_thres: 区域数目变化阈值，取值范围[0,1]
    :param relevance: 关联比例，关联比例越小，被视作正常运动的帧越多，取值范围[0,1]
    :param relevance_time: 关联时间，单位秒，以该帧为中心，前后n/2秒的帧被视作相关帧，非负数
    :param noise_proportion: 噪声比例，可被视作噪声的区域占视频面积的比例，不应过大也不应过小，一般以万分一为调整单位。非负数
    :param show_frame: 实时显示，为真时实时显示处理结果
    :param show_log: 日志显示，为真时将日志写入视频帧
    :param output: 输出路径
    :param skip_time: 跳过时间，从指定时间开始检测，非负数
    :return: 无返回值。
    """
    #   视频读取
    cap = cv.VideoCapture(src)
    #   视频fps读取
    fps = cap.get(cv.CAP_PROP_FPS)
    print(fps)
    #   视频大小读取
    size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
    print(size)
    #   视频保存编码
    fourcc_mp4 = cv.VideoWriter_fourcc(*'mp4v')
    #   保存视频
    out_detect = cv.VideoWriter(output, fourcc_mp4, fps, size, True)

    #   参数初始化
    #   前景检测模型
    mog = cv.createBackgroundSubtractorMOG2()
    knn = cv.createBackgroundSubtractorKNN(detectShadows=False, history=5)
    #   帧数
    frame_count = 0
    skip_frame = skip_time * fps
    #   阈值
    noise_thres = size[0] * size[1] * noise_proportion
    queue_max_len = fps * relevance_time // 2 * 2 + 1
    print(queue_max_len)
    #   相关帧队列
    queue_frame = []
    queue_contours = []
    #   视频处理
    while cap.isOpened():
        #   日志初始化
        logs = []
        frame_count += 1
        # print(frame_count)
        log = 'Frame:' + str(frame_count) + '  noise_thres:' + str(noise_thres) + '  noise_proportion:' + str(noise_proportion)
        logs.append(log)
        log = 'area_thres:' + str(area_thres) + '  slope_thres:' + str(slope_thres) + ' num_thres:' + str(num_thres)
        log = log + '  relevance:' + str(relevance) + '  relevance_time:' + str(relevance_time)
        logs.append(log)
        #   读入视频帧
        ret, frame = cap.read()
        #   跳过前n帧
        if frame_count < skip_frame:
            continue
        enque(queue_frame, frame, queue_max_len)
        #   视频结束，跳出循环
        if frame is None:
            break
        #   转灰度图
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        #   前景检测
        contours_KNN = model_process(gray_frame, knn, noise_thres)
        #   光照影响消减
        if len(queue_contours) < queue_max_len or queue_max_len <= 1:
            contours_better = contours_KNN
        else:
            log, contours_better = get_better_contours(queue_contours, area_thres, slope_thres, num_thres, relevance)
            for l in log:
                logs.append(l)   
        enque(queue_contours, contours_KNN, queue_max_len)
        frame = queue_frame[len(queue_frame) // 2]
        #   日志写入视频帧
        if show_log:
            add_log(logs, frame, size)

        #   绘制检测区
        for c in contours_better:
            (x, y, w, h) = cv.boundingRect(c)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        #   图像实时显示
        if show_frame:
            cv.namedWindow("frame", cv.WINDOW_NORMAL)
            cv.imshow("frame", frame)
            #   按q键退出
            if cv.waitKey(1) == ord('q'):
                break

        #   处理结果写入
        out_detect.write(frame)

    #   资源释放
    cap.release()
    cv.destroyAllWindows()
    out_detect.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default='testVideo.mp4')
    parser.add_argument('--area_thres', type=float, default=2)
    parser.add_argument('--slope_thres', type=float, default=0.5)
    parser.add_argument('--num_thres', type=float, default=3)
    parser.add_argument('--relevance', type=float, default=0.8)
    parser.add_argument('--relevance_time', type=float, default=0.5)
    parser.add_argument('--noise_proportion', type=float, default=0.00002)
    parser.add_argument('--show_frame', type=bool, default=0)
    parser.add_argument('--show_log', type=bool, default=0)
    parser.add_argument('--output', type=str, default='output.mp4')
    parser.add_argument('--skip_time', type=float, default=0)
    args = parser.parse_args()
    motion_detect(args.src,
                  args.area_thres,
                  args.slope_thres,
                  args.num_thres,
                  args.relevance,
                  args.relevance_time,
                  args.noise_proportion,
                  args.show_frame,
                  args.show_log,
                  args.output,
                  args.skip_time)


if __name__ == '__main__':
    main()
