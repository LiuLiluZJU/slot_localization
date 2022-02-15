from curses.panel import new_panel
import os
import numpy as np
import cv2
import random
from timeit import default_timer as timer
import image_processing


def on_mouse(event, x, y, flags, param):
    global img, point1, point2, g_rect
    img2 = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击,则在原图打点
        # print("1-EVENT_LBUTTONDOWN")
        point1 = (x, y)
        cv2.circle(img2, point1, 10, (0, 255, 0), 5)
        cv2.imshow('image', img2)

    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # 按住左键拖曳，画框
        # print("2-EVENT_FLAG_LBUTTON")
        cv2.rectangle(img2, point1, (x, y), (255, 0, 0), thickness=2)
        cv2.imshow('image', img2)

    elif event == cv2.EVENT_LBUTTONUP:  # 左键释放，显示
        # print("3-EVENT_LBUTTONUP")
        point2 = (x, y)
        cv2.rectangle(img2, point1, point2, (0, 0, 255), thickness=2)
        cv2.imshow('image', img2)
        if point1 != point2:
            min_x = min(point1[0], point2[0])
            min_y = min(point1[1], point2[1])
            width = abs(point1[0] - point2[0])
            height = abs(point1[1] - point2[1])
            g_rect = [min_x, min_y, width, height]
            cut_img = img[min_y:min_y + height, min_x:min_x + width]
            cv2.imshow('ROI', cut_img)


def get_image_roi(rgb_image):
    '''
    获得用户ROI区域的rect=[x,y,w,h]
    :param rgb_image:
    :return:
    '''
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    global img
    img = bgr_image
    cv2.namedWindow('image')
    while True:
        cv2.setMouseCallback('image', on_mouse)
        # cv2.startWindowThread()  # 加在这个位置
        cv2.imshow('image', img)
        key = cv2.waitKey(0)
        if key == 13 or key == 32:  # 按空格和回车键退出
            break
    cv2.destroyAllWindows()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return g_rect


def select_user_roi(orig_image):
    '''
    由于原图的分辨率较大，这里缩小后获取ROI，返回时需要重新scale对应原图
    :param image_path:
    :return:
    '''
    orig_shape = np.shape(orig_image)
    resize_image = image_processing.resize_image(orig_image, resize_height=800, resize_width=None)
    re_shape = np.shape(resize_image)
    g_rect = get_image_roi(resize_image)
    orgi_rect = image_processing.scale_rect(g_rect, re_shape, orig_shape)
    return orgi_rect


def GetROI(image, background, show_image_flag=0):
    """返回图像中感兴趣物体的包围框和掩模
    
    @param image：输入图像
    @param background：图像背景
    @param show_image_flag：是否显示中间结果，默认为不显示（0）
    @return bboxes：感兴趣物体的矩形包围框
    @return mask：感兴趣物体的掩模

    """

    # 完整图减去背景
    backsub_res = cv2.subtract(image, background)

    # 灰度化
    gray_img = cv2.cvtColor(backsub_res, cv2.COLOR_BGR2GRAY)
    
    # 二值化
    retval,threshold = cv2.threshold(gray_img, 10, 255, cv2.THRESH_BINARY)

    # 中值滤波
    threshold = cv2.medianBlur(threshold, 3)

    # 先膨胀后腐蚀
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
    
    # 先腐蚀后膨胀
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    # 寻找轮廓（掩模形式）
    mask = np.zeros_like(image)
    _, contours, hierarchy = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(mask, contours, contourIdx=-1, color=(255, 255, 255), thickness=-1)
    mask[mask > 0] = 1

    # 寻找轮廓的包围框（轮廓包围面积大于1000的被选择）
    image_show = image.copy()
    bboxes = []
    for c in contours:
        if cv2.contourArea(c) > 1000:
            x, y, w, h = cv2.boundingRect(c)
            bboxes.append((x, y, w, h))
            cv2.rectangle(image_show, (x, y), (x + w, y + h), (0, 255, 0), 1)
    
    # 显示
    if show_image_flag:
        cv2.imshow('backsub_res', backsub_res)
        cv2.waitKey(0)
        cv2.imshow('gray_img', gray_img)
        cv2.waitKey(0)
        cv2.imshow('threshold', threshold)
        cv2.waitKey(0)
        cv2.imshow('closed', closed)
        cv2.waitKey(0)
        cv2.imshow('opened', opened)
        cv2.waitKey(0)
        cv2.imshow('scene_img', image_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return bboxes, mask


def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    # assert A.shape == B.shape #去掉断言

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def findEucTransformation(src_pts, dst_pts):
    assert src_pts.shape == dst_pts.shape
    src_pts = np.squeeze(src_pts)
    dst_pts = np.squeeze(dst_pts)
    num = src_pts.shape[0]
    inlier_num_max = 0
    R_ret = np.identity(2)
    t_ret = np.zeros(2)
    T_ret = np.identity(3)
    for i in range(100):
        random_index = np.random.choice(np.arange(num), (2), replace=False)
        random_src_pts = src_pts[random_index]
        random_dst_pts = dst_pts[random_index]
        T, R, t = best_fit_transform(random_src_pts, random_dst_pts)
        pts_diff = np.dot(R, src_pts.T).T + t - dst_pts
        pts_norm = np.linalg.norm(pts_diff, axis=1)
        inlier_num = np.sum(pts_norm < 10)
        if inlier_num > inlier_num_max:
            inlier_num_max = inlier_num
            R_ret = R
            t_ret = t
    T_ret[:2, :2] = R_ret
    T_ret[:2, 2] = t_ret
    return T_ret, R_ret, t_ret



if __name__ == "__main__":

    tic = timer()

    # 读取图像
    background = cv2.imread("./images2/background.jpeg")
    full_jigsaw_1 = cv2.imread("./images/full.bmp")
    full_jigsaw = cv2.imread("./images/full2.png")
    full_jigsaw = cv2.resize(full_jigsaw, (full_jigsaw_1.shape[1], full_jigsaw_1.shape[0]))
    scene_img = cv2.imread("./images2/scene_1.jpeg")

    full_jigsaw_1_origin = np.array([2981, 1658])
    full_jigsaw_origin = np.array([2967, 1653])

    # cv2.namedWindow("target", 0)
    # cv2.imshow("target", full_jigsaw_1)
    # cv2.waitKey(0)

    template_rect = select_user_roi(full_jigsaw_1)
    template = full_jigsaw_1[template_rect[1] : template_rect[1] + template_rect[3], template_rect[0] : template_rect[0] + template_rect[2]]
    # template = template[::-1, ::-1]
    cv2.imshow("template", template)
    cv2.waitKey(0)

    # 初始化SIFT提取器
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=0, nOctaveLayers=1, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
    # sift = cv2.xfeatures2d.SIFT_create(nfeatures=0, nOctaveLayers=1, contrastThreshold=0.04, edgeThreshold=10, sigma=10)

    # 使用SIFT提取关键点和描述子
    # full_jigsaw = full_jigsaw[:, int(full_jigsaw.shape[1] / 2) : full_jigsaw.shape[1]]
    kp2, des2 = sift.detectAndCompute(full_jigsaw, None)
    target_img = cv2.drawKeypoints(full_jigsaw, kp2, None)

    # 使用SIFT提取关键点和描述子
    kp1, des1 = sift.detectAndCompute(template, None)
    template_img = cv2.drawKeypoints(template, kp1, None)
    # cv2.imshow('template_img', template_img)
    # cv2.waitKey(0)

    # 关键点匹配
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    flann = cv2.BFMatcher()
    matches = flann.knnMatch(des1, des2, k=2)

    # 保存匹配良好的点对
    good = []
    for m, n in matches:
        if m.distance < 0.95 * n.distance:
            good.append(m)

    # 计算变换矩阵
    MIN_MATCH_COUNT = 5
    if len(good) > MIN_MATCH_COUNT:
        # 获取关键点坐标
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # 保存匹配点对坐标
        # src_pts_global = src_pts + np.array([box_template[0], box_template[1]])
        # dst_pts_global = dst_pts + np.array([box_target[0], box_target[1]])
        # with open("./matching.txt", 'w') as f:
        #     f.write('map index + query index \r\n')
        #     for k in range(dst_pts_global.shape[0]):
        #         line1 = np.squeeze(dst_pts_global[k])
        #         line2 = np.squeeze(src_pts_global[k])
        #         f.write(str(line1[0]) + ' ' + str(line1[1]) + ' ' + str(line2[0]) + ' ' + str(line2[1]) + '\r\n')

        T, R, t = findEucTransformation(src_pts, dst_pts)
        
        # 计算单应矩阵
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h, w, c = template.shape

        # 变换模板图
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, T)
        cv2.polylines(full_jigsaw, [np.int32(dst)], True, 0, 2, cv2.LINE_AA)  # 画出变换后的外包围框
    else:
        print( "Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
        matchesMask = None

    # 计算匹配后的像素误差
    pred_full_jigsaw_origin = np.dot(R, full_jigsaw_1_origin - template_rect[:2]) + t
    diff = np.linalg.norm(pred_full_jigsaw_origin - full_jigsaw_origin)
    pred_full_jigsaw_origin = pred_full_jigsaw_origin.astype(np.int)
    cv2.line(full_jigsaw, (pred_full_jigsaw_origin[0], 0), (pred_full_jigsaw_origin[0], full_jigsaw.shape[0]), (0, 0, 255), 10)
    cv2.line(full_jigsaw, (0, pred_full_jigsaw_origin[1]), (full_jigsaw.shape[1], pred_full_jigsaw_origin[1]), (0, 0, 255), 10)
    print("Predicted pixel error:", diff)

    # 画匹配点
    draw_params = dict(matchColor=(0, 255, 0), 
                    singlePointColor=None,
                    matchesMask=matchesMask, 
                    flags=2)
    result = cv2.drawMatches(template, kp1, full_jigsaw, kp2, good, None, **draw_params)
    cv2.namedWindow("result", 0)
    cv2.resizeWindow("result", 640, 480)
    cv2.imshow('result', result)
    cv2.waitKey(0)

    # # 提取原图中的ROI(包围框、掩模)及目标图
    # ROIs_target, mask_target = GetROI(full_jigsaw, background, 0)
    # box_target = ROIs_target[0]
    # full_jigsaw = np.multiply(full_jigsaw, mask_target)  # 使用掩模
    # cv2.imshow('full_jigsaw', full_jigsaw)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # # 提取散乱图中的ROI(包围框、掩模)
    # ROIs_template, mask_template = GetROI(scene_img, background, 1)
    # scene_img = np.multiply(scene_img, mask_template)  # 使用掩模

    # # 初始化SIFT提取器
    # sift = cv2.xfeatures2d.SIFT_create(nfeatures=0,nOctaveLayers=1, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
    # # sift = cv2.xfeatures2d.SIFT_create()

    # # 切分目标图，一共num_x * num_y块子图，相邻子图之间有一半重叠
    # num_x = 4
    # num_y = 4
    # for i in range(num_x):
    #     for j in range(num_y):

    #         # 利用原图ROI提取目标图
    #         interval_x = int(box_target[2] / (num_x + 1))
    #         interval_y = int(box_target[3] / (num_y + 1))
    #         start_x = box_target[0] + interval_x * i
    #         start_y = box_target[1] + interval_y * j
    #         stop_x = start_x + 2 * interval_x
    #         stop_y = start_y + 2 * interval_y
    #         target = full_jigsaw[start_y : stop_y, start_x : stop_x]

    #         # 使用SIFT提取关键点和描述子
    #         kp2, des2 = sift.detectAndCompute(target, None)
    #         target_img = cv2.drawKeypoints(target, kp2, None)

    #         for box_template in ROIs_template:
    #             # 利用散乱图的ROI裁剪模板图
    #             template = scene_img[box_template[1] : box_template[1] + box_template[3], box_template[0] : box_template[0] + box_template[2]]

    #             # 使用SIFT提取关键点和描述子
    #             kp1, des1 = sift.detectAndCompute(template,None)
    #             template_img = cv2.drawKeypoints(template, kp1, None)
    #             # cv2.imshow('template_img', template_img)
    #             # cv2.waitKey(0)

    #             # 关键点匹配
    #             FLANN_INDEX_KDTREE = 0
    #             index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    #             search_params = dict(checks=50)
    #             # flann = cv2.FlannBasedMatcher(index_params, search_params)
    #             flann = cv2.BFMatcher()
    #             matches = flann.knnMatch(des1, des2, k=2)

    #             # 保存匹配良好的点对
    #             good = []
    #             for m, n in matches:
    #                 if m.distance < 0.95 * n.distance:
    #                     good.append(m)

    #             # 计算变换矩阵
    #             MIN_MATCH_COUNT = 10
    #             if len(good) > MIN_MATCH_COUNT:
    #                 # 获取关键点坐标
    #                 src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    #                 dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    #                 # 保存匹配点对坐标
    #                 # src_pts_global = src_pts + np.array([box_template[0], box_template[1]])
    #                 # dst_pts_global = dst_pts + np.array([box_target[0], box_target[1]])
    #                 # with open("./matching.txt", 'w') as f:
    #                 #     f.write('map index + query index \r\n')
    #                 #     for k in range(dst_pts_global.shape[0]):
    #                 #         line1 = np.squeeze(dst_pts_global[k])
    #                 #         line2 = np.squeeze(src_pts_global[k])
    #                 #         f.write(str(line1[0]) + ' ' + str(line1[1]) + ' ' + str(line2[0]) + ' ' + str(line2[1]) + '\r\n')
                    
    #                 # 计算单应矩阵
    #                 M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    #                 matchesMask = mask.ravel().tolist()
    #                 h,w,c = template.shape

    #                 # 变换模板图
    #                 pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    #                 dst = cv2.perspectiveTransform(pts,M)
    #                 # cv2.polylines(target,[np.int32(dst)],True,0,2, cv2.LINE_AA)  # 画出变换后的外包围框
    #             else:
    #                 print( "Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
    #                 matchesMask = None

    #             draw_params = dict(matchColor=(0, 255, 0), 
    #                             singlePointColor=None,
    #                             matchesMask=matchesMask, 
    #                             flags=2)
    #             result = cv2.drawMatches(template,kp1,target,kp2,good,None,**draw_params)
    #             cv2.imshow('result', result)
    #             cv2.waitKey(0)

    #         cv2.destroyAllWindows()

    toc = timer() 
    print('template match using time {}'.format(toc - tic))  
            

