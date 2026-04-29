import cv2
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def add_salt_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    """
    添加椒盐噪声
    salt_prob: 盐噪声（白点）概率
    pepper_prob: 椒噪声（黑点）概率
    """
    noisy_image = image.copy()
    total_pixels = image.shape[0] * image.shape[1]
    
    # 添加盐噪声（白点）
    num_salt = int(total_pixels * salt_prob)
    salt_coords = [np.random.randint(0, i-1, num_salt) for i in image.shape[:2]]
    if len(image.shape) == 3:  # 彩色图像
        noisy_image[salt_coords[0], salt_coords[1], :] = 255
    else:  # 灰度图像
        noisy_image[salt_coords[0], salt_coords[1]] = 255
    
    # 添加椒噪声（黑点）
    num_pepper = int(total_pixels * pepper_prob)
    pepper_coords = [np.random.randint(0, i-1, num_pepper) for i in image.shape[:2]]
    if len(image.shape) == 3:  # 彩色图像
        noisy_image[pepper_coords[0], pepper_coords[1], :] = 0
    else:  # 灰度图像
        noisy_image[pepper_coords[0], pepper_coords[1]] = 0
    
    return noisy_image

def sift_match(img1, img2, title="Feature Matching"):
    """
    SIFT特征提取与匹配
    """
    # 转换为灰度图像
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = img1
        
    if len(img2.shape) == 3:
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = img2
    
    # 创建SIFT对象
    sift = cv2.SIFT_create()
    
    # 检测关键点和计算描述子
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    print(f"图像1关键点数量: {len(kp1)}")
    print(f"图像2关键点数量: {len(kp2)}")
    
    # 如果描述子为空，返回空结果
    if des1 is None or des2 is None:
        print("警告: 无法检测到足够的关键点")
        return None, None, None, None, 0
    
    # 使用FLANN匹配器
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    # 使用Lowe's ratio test筛选好的匹配点
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    print(f"良好匹配点数量: {len(good_matches)}")
    
    # 绘制匹配结果
    matched_img = cv2.drawMatches(
        img1, kp1, img2, kp2, good_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    return matched_img, kp1, kp2, good_matches, len(good_matches)

def display_images_with_matches(original, transformed, matched_img, title, transform_name):
    """
    显示原始图像、变换图像和匹配结果
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 显示原始图像
    if len(original.shape) == 3:
        axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    else:
        axes[0].imshow(original, cmap='gray')
    axes[0].set_title('原始图像')
    axes[0].axis('off')
    
    # 显示变换后的图像
    if len(transformed.shape) == 3:
        axes[1].imshow(cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB))
    else:
        axes[1].imshow(transformed, cmap='gray')
    axes[1].set_title(f'{transform_name}')
    axes[1].axis('off')
    
    # 显示匹配结果
    if matched_img is not None:
        # 转换颜色空间用于显示
        if len(matched_img.shape) == 3:
            matched_img_rgb = cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB)
        else:
            matched_img_rgb = matched_img
        axes[2].imshow(matched_img_rgb)
    else:
        axes[2].text(0.5, 0.5, '匹配失败\n无法检测到足够的关键点', 
                     ha='center', va='center', transform=axes[2].transAxes)
    axes[2].set_title(f'匹配结果: {title}')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    """
    主函数：读取图像，进行变换，并进行SIFT特征匹配
    """
    # 读取图像
    image_path = input("请输入图像路径（直接回车使用示例图像）: ").strip()
    
    if not image_path:
        # 如果没有输入，创建一个简单的示例图像
        print("未输入图像路径，创建示例图像...")
        img = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.rectangle(img, (50, 100), (200, 300), (0, 0, 255), -1)
        cv2.rectangle(img, (250, 150), (400, 250), (0, 255, 0), -1)
        cv2.circle(img, (500, 200), 80, (255, 0, 0), -1)
        cv2.putText(img, 'SIFT', (200, 380), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    else:
        img = cv2.imread(image_path)
        if img is None:
            print(f"错误：无法读取图像 {image_path}")
            return
    
    print(f"原始图像尺寸: {img.shape}")
    
    # 1. 缩放图像
    scale_factor = 0.5
    img_scaled = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)
    print(f"缩放后图像尺寸: {img_scaled.shape}")
    
    # 2. 水平翻转
    img_flipped_h = cv2.flip(img, 1)  # 1表示水平翻转
    img_flipped_v = cv2.flip(img, 0)  # 0表示垂直翻转
    
    # 3. 添加椒盐噪声
    img_noisy = add_salt_pepper_noise(img, salt_prob=0.02, pepper_prob=0.02)
    
    print("\n" + "="*60)
    print("开始SIFT特征匹配")
    print("="*60)
    
    # 创建结果存储
    results = []
    
    # 测试1: 原始图像 vs 缩放图像
    print("\n--- 测试1: 原始图像 vs 缩放图像 ---")
    matched_img1, kp1, kp2, good_matches1, num_matches1 = sift_match(img, img_scaled)
    results.append({
        'transformed': img_scaled,
        'matched_img': matched_img1,
        'title': f'缩放匹配 ({num_matches1}个匹配点)',
        'transform_name': f'缩放 (scale={scale_factor})'
    })
    
    # 测试2: 原始图像 vs 水平翻转图像
    print("\n--- 测试2: 原始图像 vs 水平翻转图像 ---")
    matched_img2, kp1, kp2, good_matches2, num_matches2 = sift_match(img, img_flipped_h)
    results.append({
        'transformed': img_flipped_h,
        'matched_img': matched_img2,
        'title': f'水平翻转匹配 ({num_matches2}个匹配点)',
        'transform_name': '水平翻转'
    })
    
    # 测试3: 原始图像 vs 椒盐噪声图像
    print("\n--- 测试3: 原始图像 vs 椒盐噪声图像 ---")
    matched_img3, kp1, kp2, good_matches3, num_matches3 = sift_match(img, img_noisy)
    results.append({
        'transformed': img_noisy,
        'matched_img': matched_img3,
        'title': f'椒盐噪声匹配 ({num_matches3}个匹配点)',
        'transform_name': '椒盐噪声'
    })
    
    # 显示所有匹配结果
    for i, result in enumerate(results):
        display_images_with_matches(
            img, 
            result['transformed'],
            result['matched_img'],
            result['title'],
            result['transform_name']
        )
    
    # 绘制匹配点数量对比柱状图
    plot_match_comparison(results)
    
    print("\n" + "="*60)
    print("匹配完成！")
    print("="*60)

def plot_match_comparison(results):
    """
    绘制不同变换下的匹配点数量对比图
    """
    transform_names = ['缩放', '水平翻转', '椒盐噪声']
    match_counts = []
    
    for result in results:
        # 从title中提取匹配点数量
        title = result['title']
        import re
        match = re.search(r'\((\d+)个匹配点\)', title)
        if match:
            match_counts.append(int(match.group(1)))
        else:
            match_counts.append(0)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(transform_names, match_counts, color=['#3498db', '#e74c3c', '#2ecc71'])
    
    # 在柱状图上添加数值标签
    for bar, count in zip(bars, match_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{count}', ha='center', va='bottom', fontsize=12)
    
    ax.set_xlabel('变换类型', fontsize=12)
    ax.set_ylabel('匹配点数量', fontsize=12)
    ax.set_title('不同变换下的SIFT特征匹配点数量对比', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()

def advanced_sift_match_with_homography(img1, img2, title="Feature Matching"):
    """
    高级SIFT匹配：使用RANSAC计算单应性矩阵并绘制更详细的匹配结果
    """
    # 转换为灰度图像
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = img1
        
    if len(img2.shape) == 3:
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = img2
    
    # 创建SIFT对象
    sift = cv2.SIFT_create()
    
    # 检测关键点和计算描述子
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    if des1 is None or des2 is None:
        print("无法检测到足够的关键点")
        return None, 0, None
    
    # 使用BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    
    # 筛选好的匹配点
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    print(f"良好匹配点数量: {len(good_matches)}")
    
    # 如果匹配点足够多，计算单应性矩阵
    H = None
    inlier_matches = good_matches
    
    if len(good_matches) >= 4:
        # 获取匹配点的坐标
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # 使用RANSAC计算单应性矩阵
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if H is not None:
            inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i]]
            print(f"RANSAC内点数量: {len(inlier_matches)}")
    
    # 绘制匹配结果
    matched_img = cv2.drawMatches(
        img1, kp1, img2, kp2, inlier_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    return matched_img, len(inlier_matches), H

if __name__ == "__main__":
    main()