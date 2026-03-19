import cv2
import numpy as np

# 1. 打开视频文件
cap = cv2.VideoCapture('1.avi')

# 配置 Harris 角点检测参数
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7, useHarrisDetector=True)

# 配置 LK 光流参数 (包含金字塔层数 maxLevel)
lk_params = dict(winSize=(15, 15), maxLevel=2, 
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# 2. 读取第一帧并提取角点
ret, old_frame = cap.read()
if not ret:
    print("无法读取视频")
    cap.release()
    exit()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# 创建一个用于绘制轨迹的掩膜图
mask = np.zeros_like(old_frame)

while True:
    # 3. 循环读取后续帧
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 4. 计算金字塔 LK 光流
    # p1 是新帧中的点位置，st 是状态(1表示跟踪成功)，err 是误差
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # 5. 筛选有效点
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # 6. 绘制运动轨迹
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            # 绘制线条（在掩膜上）
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
            # 绘制当前点（在当前帧上）
            frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

        img = cv2.add(frame, mask)
        cv2.imshow('LK Optical Flow - Pyramid', img)

        # 7. 更新上一帧的数据
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
        
        # 如果跟踪的点太少，可以重新检测角点（可选）
        if len(good_new) < 10:
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
            mask = np.zeros_like(old_frame) # 重置轨迹

    if cv2.waitKey(30) & 0xFF == 27: # 按 ESC 退出
        break

cap.release()
cv2.destroyAllWindows()
