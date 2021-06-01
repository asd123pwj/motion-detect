# 中文介绍
详细介绍见我的博客：https://mwhls.top/2485.html

我文章的格式不是markdown，复制粘贴过来效果不好，所以就写上使用方式。

也许哪天我会再写一个WordPress转markdown格式的。

## 使用方式：

Python版本及库需求：
Python 3.7 
opencv 4.5.1.48 
numpy 1.20.2 
argparse

## 工作目录：

motion_detect.py    #运动检测代码

testVideo.mp4    #测试视频

## 启动命令： 
默认参数执行：python motion_detect.py 
自订参数执行：python motion_detect.py --relevance 0.9
 
## 输出：
首先命令行输出视频fps与关联帧数，随后将处理好的帧数的序号打印出。

结束后会生成output.mp4视频文件。

实时输出及日志：

当show_frame参数为True时，实时显示处理结果，按q键提前退出，并正常保存视频。

当show_log参数为True时，将当前帧的处理日志写入视频帧中。

当skip_time参数有正数值时，将会跳过一定秒数

推荐在测试时将上面三个参数按需使用，方便根据结果修改阈值参数。


## 参数

src: 待检测视频路径
默认值：'testVideo.mp4'

area_thres: 面积变化阈值，非负数
默认值：2

slope_thres: 面积变化斜率阈值，取值范围[0,1]
默认值：0.5

num_thres: 区域数目变化阈值，取值范围[0,1]
默认值：3

relevance: 关联比例，关联比例越小，被视作正常运动的帧越多，取值范围[0,1]
默认值：0.8

relevance_time: 关联时间，单位秒，以该帧为中心，前后n/2秒的帧被视作相关帧，非负数
默认值：0.5

noise_proportion: 噪声比例，可被视作噪声的区域占视频面积的比例，不应过大也不应过小，一般以万分一为调整单位。非负数
默认值：0.00002

show_frame: 实时显示，为真时实时显示处理结果
默认值：0

show_log: 日志显示，为真时将日志写入视频帧
默认值：0

output: 输出路径
默认值：'output.mp4'

skip_time: 跳过时间，从指定时间开始检测，非负数
默认值：0
