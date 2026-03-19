import cv2
import numpy as np

def stabilize_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    
    # 获取视频属性
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 1. 预计算所有帧间的运动轨迹
    _, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # 存储每帧相对于前一帧的 [dx, dy, da] (平移x, 平移y, 角度)
    transforms = np.zeros((n_frames-1, 3), np.float32)

    for i in range(n_frames - 1):
        ret, curr_frame = cap.read()
        if not ret: break
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # LK 光流找对应点
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

        # 筛选有效点
        idx = np.where(status == 1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]

        # 计算仿射变换矩阵 (仅估计平移和旋转)
        m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
        
        dx = m[0, 2]
        dy = m[1, 2]
        da = np.arctan2(m[1, 0], m[0, 0])
        
        transforms[i] = [dx, dy, da]
        prev_gray = curr_gray

    # 2. 计算累积轨迹并进行平滑处理 (均值滤波)
    trajectory = np.cumsum(transforms, axis=0)
    
    def moving_average(curve, radius):
        window_size = 2 * radius + 1
        f = np.ones(window_size) / window_size
        curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
        return np.convolve(curve_pad, f, mode='valid')

    # 平滑半径，越大越平稳但黑边越多
    SMOOTH_RADIUS = 30
    smoothed_trajectory = np.copy(trajectory)
    for i in range(3):
        smoothed_trajectory[:, i] = moving_average(trajectory[:, i], radius=SMOOTH_RADIUS)

    # 计算平滑后的增量
    difference = smoothed_trajectory - trajectory
    transforms_smooth = transforms + difference

    # 3. 应用平滑后的变换并写入新视频
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))
    
    for i in range(n_frames - 1):
        ret, frame = cap.read()
        if not ret: break
        
        dx, dy, da = transforms_smooth[i]
        
        # 构建新的仿射矩阵
        m = np.zeros((2, 3), np.float32)
        m[0, 0] = np.cos(da)
        m[0, 1] = -np.sin(da)
        m[1, 0] = np.sin(da)
        m[1, 1] = np.cos(da)
        m[0, 2] = dx
        m[1, 2] = dy
        
        # 应用变换
        frame_stabilized = cv2.warpAffine(frame, m, (w, h))
        
        # 稍微放大图片以去除黑边 (可选)
        # s = 1.1
        # T = cv2.getRotationMatrix2D((w/2, h/2), 0, s)
        # frame_stabilized = cv2.warpAffine(frame_stabilized, T, (w, h))

        out.write(frame_stabilized)
        cv2.imshow("Stabilized", frame_stabilized)
        cv2.waitKey(1)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

stabilize_video('1.avi', 'stable_output.avi')
