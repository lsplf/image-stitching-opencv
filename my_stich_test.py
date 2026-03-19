import cv2
import numpy as np

class FourCorners:
    def __init__(self):
        self.left_top = (0, 0)
        self.left_bottom = (0, 0)
        self.right_top = (0, 0)
        self.right_bottom = (0, 0)

corners = FourCorners()

def calc_corners(H, src):
    h, w = src.shape[:2]
    # 定义源图像的四个角点坐标 (x, y)
    pts = np.float32([[0, 0], [0, h], [w, 0], [w, h]]).reshape(-1, 1, 2)
    # 使用单应性矩阵进行透视变换
    dst_pts = cv2.perspectiveTransform(pts, H)
    
    corners.left_top = dst_pts[0][0]
    corners.left_bottom = dst_pts[1][0]
    corners.right_top = dst_pts[2][0]
    corners.right_bottom = dst_pts[3][0]

def optimize_seam(img1, trans, dst):
    start = int(min(corners.left_top[0], corners.left_bottom[0]))
    process_width = img1.shape[1] - start
    rows, cols = dst.shape[:2]
    img1_cols = img1.shape[1]

    # 线性渐变融合 (Alpha Blending)
    for i in range(rows):
        for j in range(start, img1_cols):
            # 如果 trans 对应位置是黑色（无数据），完全使用 img1
            if not trans[i, j].any():
                alpha = 1
            else:
                # 距离重叠左边界越远，img1 权重越小
                alpha = (process_width - (j - start)) / process_width
            
            dst[i, j] = img1[i, j] * alpha + trans[i, j] * (1 - alpha)

def main():
    # 1. 读取图像
    img_r = cv2.imread("scottsdale_right_01.png")  # 右图
    img_l = cv2.imread("scottsdale_left_01.png")  # 左图
    
    # 2. 灰度化与特征提取 (SURF)
    # 注意：新版本 OpenCV 可能需要使用 SIFT 或 ORB，如果 SURF 不可用
    sift = cv2.SIFT_create() # 参数根据需要调整，通常默认即可
    kp1, des1 = sift.detectAndCompute(img_r, None)
    kp2, des2 = sift.detectAndCompute(img_l, None)

    # 3. 特征匹配 (FLANN)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(des2, des1, k=2)

    # Lowe's ratio test 筛选优秀点
    good_matches = [m for m, n in matches if m.distance < 0.4 * n.distance]

    # 4. 计算单应性矩阵
    if len(good_matches) > 4:
        src_pts = np.float32([kp2[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # 这里的 imagePoints1 和 imagePoints2 在原代码中顺序是 1->2
        # 我们计算从 img_r 到 img_l 的变换
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        
        # 5. 计算变换后的顶点
        calc_corners(H, img_r)
        
        # 6. 图像配准
        dst_width = int(max(corners.right_top[0], corners.right_bottom[0]))
        image_transform_r = cv2.warpPerspective(img_r, H, (dst_width, img_l.shape[0]))
        
           # 【可视化 2】：右图变换后的样子
        cv2.imshow("Step 2: Warped Right Image", image_transform_r)
        cv2.imwrite("wrap_right.png", image_transform_r)
        
        # 7. 拼接初步处理
        dst = image_transform_r.copy()
        # 将左图 img_l 覆盖上去
        dst[0:img_l.shape[0], 0:img_l.shape[1]] = img_l
        
        # 8. 优化连接处
        optimize_seam(img_l, image_transform_r, dst)
        cv2.imwrite("stitch_result.png", dst)
        cv2.imshow("Result", dst)
        cv2.waitKey(0)
    else:
        print("Not enough matches are found!")

if __name__ == "__main__":
    main()
