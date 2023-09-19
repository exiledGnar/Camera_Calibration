import numpy as np
import cv2

# 设置棋盘参数
board_size = (9, 6)  # 棋盘上的角点数目(这里直接用9和6了，比打印出的图小一点)
square_size = 1.0  # 每个棋盘格子的大小
num_samples = 18  # 用于标定的图像样本数量

# 生成棋盘角点的世界坐标
chessboard_points_3d = np.zeros((np.prod(board_size), 3), np.float32)
chessboard_points_3d[:,:2] = np.indices(board_size).T.reshape(-1, 2)
chessboard_points_3d *= square_size

# 存储角点的世界坐标和图像坐标
world_points = []  # 世界坐标
image_points = []  # 图像坐标

# 读取标定图像
for i in range(num_samples):
    # 读取图像
    img_path = f'E:\\{i+1}.jpg'  # 图像路径
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 查找棋盘角点
    ret, corners = cv2.findChessboardCorners(gray, board_size, None)

    if ret:
        world_points.append(chessboard_points_3d)
        image_points.append(corners)

        # 绘制角点
        img = cv2.drawChessboardCorners(img, board_size, corners, ret)
        cv2.imshow('Chessboard', img)
        cv2.waitKey(0)


# 进行相机标定
ret, camera_matrix, distortion_coeffs, rvecs, tvecs = cv2.calibrateCamera(world_points, image_points, gray.shape[::-1], None, None)

# 打印标定结果
print('Camera matrix:')
print(camera_matrix)
print('\nDistortion coefficients:')
print(distortion_coeffs)