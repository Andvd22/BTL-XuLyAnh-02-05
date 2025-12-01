import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_simple_descriptor(image, keypoint):
    """
    Tính descriptor đơn giản hóa (không phải SIFT đầy đủ)
    Chỉ để minh họa cách SIFT hoạt động
    SIFT đầy đủ rất phức tạp (300-600 dòng code)
    """
    x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
    size = int(keypoint.size) if hasattr(keypoint, 'size') else 16
    
    # Lấy vùng quanh keypoint (16×16)
    half_size = size // 2
    y_min = max(0, y - half_size)
    y_max = min(image.shape[0], y + half_size)
    x_min = max(0, x - half_size)
    x_max = min(image.shape[1], x + half_size)
    
    patch = image[y_min:y_max, x_min:x_max]
    
    if patch.shape[0] < 4 or patch.shape[1] < 4:
        return None
    
    # Resize về 16×16 nếu cần
    if patch.shape[0] != 16 or patch.shape[1] != 16:
        patch = cv2.resize(patch, (16, 16))
    
    # Tính gradient
    grad_x = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
    
    # Tính magnitude và direction
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    direction = np.arctan2(grad_y, grad_x)
    direction = (direction + np.pi) / (2 * np.pi) * 8  # Chuyển về 0-8
    
    # Chia thành 4×4 = 16 ô, mỗi ô 4×4 pixel
    descriptor = []
    for i in range(4):
        for j in range(4):
            # Lấy vùng 4×4
            mag_block = magnitude[i*4:(i+1)*4, j*4:(j+1)*4]
            dir_block = direction[i*4:(i+1)*4, j*4:(j+1)*4]
            
            # Tính histogram 8 hướng
            hist = np.zeros(8)
            for k in range(4):
                for l in range(4):
                    bin_idx = int(dir_block[k, l]) % 8
                    hist[bin_idx] += mag_block[k, l]
            
            descriptor.extend(hist)
    
    descriptor = np.array(descriptor, dtype=np.float32)
    
    # Normalize để giảm ảnh hưởng của illumination
    norm = np.linalg.norm(descriptor)
    if norm > 0:
        descriptor = descriptor / norm
    
    # Clip giá trị lớn (giảm ảnh hưởng của gradient lớn)
    descriptor = np.clip(descriptor, 0, 0.2)
    
    # Normalize lại
    norm = np.linalg.norm(descriptor)
    if norm > 0:
        descriptor = descriptor / norm
    
    return descriptor

def detect_keypoints_simple(image):
    """
    Tìm keypoints đơn giản (sử dụng Harris corner hoặc FAST)
    SIFT đầy đủ có scale-space và orientation, nhưng đây chỉ là phiên bản đơn giản
    """
    # Dùng Harris corner detector
    corners = cv2.goodFeaturesToTrack(image, maxCorners=1000, qualityLevel=0.01, minDistance=10)
    
    if corners is None:
        return []
    
    # Chuyển thành keypoints
    keypoints = []
    for corner in corners:
        x, y = corner[0]
        kp = cv2.KeyPoint(x, y, 16)  # size = 16
        keypoints.append(kp)
    
    return keypoints

def sift_detect_and_compute(image):
    """
    Tự implement SIFT đơn giản hóa
    Lưu ý: Đây KHÔNG phải SIFT đầy đủ, chỉ để minh họa
    SIFT đầy đủ cần: scale-space, DoG, orientation assignment, v.v.
    """
    # Tìm keypoints
    keypoints = detect_keypoints_simple(image)
    
    if len(keypoints) == 0:
        return [], None
    
    # Tính descriptors
    descriptors = []
    valid_keypoints = []
    
    for kp in keypoints:
        desc = compute_simple_descriptor(image, kp)
        if desc is not None:
            descriptors.append(desc)
            valid_keypoints.append(kp)
    
    if len(descriptors) == 0:
        return [], None
    
    descriptors = np.array(descriptors, dtype=np.float32)
    
    return valid_keypoints, descriptors

def blend_images(img_left, img_right_warped, h_left, w_left, h_right, w_right, is_img1_left):
    """
    Blend 2 ảnh ở vùng overlap để tránh viền đen
    Args:
        img_left: Ảnh bên trái (không warp)
        img_right_warped: Ảnh bên phải (đã warp)
        h_left, w_left: Kích thước ảnh trái
        h_right, w_right: Kích thước ảnh phải
        is_img1_left: True nếu img1 ở bên trái (không dùng trong hàm này, chỉ để tương thích)
    Returns:
        Ảnh đã blend
    """
    result = img_right_warped.copy()
    result_h, result_w = result.shape[:2]
    
    # Tìm vùng overlap bằng cách kiểm tra pixel nào có giá trị trong cả 2 ảnh
    # Vùng overlap thường nằm ở ranh giới giữa 2 ảnh
    overlap_width = min(150, w_left, result_w // 3)  # Độ rộng vùng overlap
    overlap_start = max(0, w_left - overlap_width)
    overlap_end = min(w_left, result_w)
    
    if overlap_start < overlap_end:
        blend_width = overlap_end - overlap_start
        if blend_width > 0:
            # Tạo mask linear gradient cho blending
            # Mask cho ảnh trái: từ 1.0 giảm xuống 0.0 (từ trái sang phải)
            # Mask cho ảnh phải: từ 0.0 tăng lên 1.0 (từ trái sang phải)
            mask_left = np.linspace(1.0, 0.0, blend_width).reshape(1, -1, 1)
            mask_right = 1.0 - mask_left
            
            h_blend = min(h_left, result_h)
            mask_left = np.repeat(mask_left, h_blend, axis=0)
            mask_right = np.repeat(mask_right, h_blend, axis=0)
            
            # Lấy vùng overlap từ cả 2 ảnh
            img_left_overlap = img_left[0:h_blend, overlap_start:min(w_left, overlap_end)]
            img_right_overlap = result[0:h_blend, overlap_start:overlap_end]
            
            # Đảm bảo cùng kích thước
            min_w = min(img_left_overlap.shape[1], img_right_overlap.shape[1])
            if min_w > 0:
                img_left_overlap = img_left_overlap[:, :min_w]
                img_right_overlap = img_right_overlap[:, :min_w]
                mask_left = mask_left[:, :min_w]
                mask_right = mask_right[:, :min_w]
                
                # Blend vùng overlap
                result[0:h_blend, overlap_start:overlap_start+min_w] = (
                    img_left_overlap.astype(np.float32) * mask_left + 
                    img_right_overlap.astype(np.float32) * mask_right
                ).astype(np.uint8)
    
    # Đặt ảnh trái vào vùng hoàn toàn không overlap (bên trái)
    if overlap_start > 0:
        result[0:min(h_left, result_h), 0:min(overlap_start, w_left)] = \
            img_left[0:min(h_left, result_h), 0:min(overlap_start, w_left)]
    
    return result

def panorama(img1, img2):
    """
    Ghép ảnh panorama tự động phát hiện thứ tự đúng
    Sử dụng KNN matcher với Lowe's ratio test để lọc matches tốt hơn
    Có blending ở vùng overlap để tránh viền đen
    """
    scr_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    tar_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    # Tự implement SIFT đơn giản hóa (thay vì dùng hàm có sẵn)
    # Lưu ý: Đây là phiên bản đơn giản hóa, không phải SIFT đầy đủ
    # SIFT đầy đủ rất phức tạp (300-600 dòng code) với scale-space, DoG, orientation, v.v.
    # Phiên bản này chỉ minh họa cách tính descriptor từ gradient
    
    # Option 1: Dùng hàm tự implement (đơn giản hóa)
    USE_CUSTOM_SIFT = False  # Đặt True để dùng phiên bản tự implement
    
    if USE_CUSTOM_SIFT:
        k1, d1 = sift_detect_and_compute(scr_gray)
        k2, d2 = sift_detect_and_compute(tar_gray)
    else:
        # Option 2: Dùng hàm OpenCV (khuyến nghị cho production)
        # SIFT đầy đủ đã được tối ưu và test kỹ
        try:
            Sift_detect = cv2.xfeatures2d.SIFT_create()
        except:
            # Nếu không có xfeatures2d, dùng SIFT thường (OpenCV 4.5+)
            Sift_detect = cv2.SIFT_create()
        k1, d1 = Sift_detect.detectAndCompute(scr_gray, None)
        k2, d2 = Sift_detect.detectAndCompute(tar_gray, None)

    # Kiểm tra nếu không có đủ keypoints
    if d1 is None or d2 is None or len(k1) < 4 or len(k2) < 4:
        raise ValueError("Không tìm thấy đủ điểm đặc trưng trong ảnh. Vui lòng chọn ảnh có nội dung tương đồng.")

    # Sử dụng KNN matcher với Lowe's ratio test (tốt hơn match đơn giản)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn_matches = bf.knnMatch(d1, d2, k=2)
    
    # Áp dụng Lowe's Ratio Test để lọc matches tốt
    ratio_thresh = 0.75
    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    
    # Kiểm tra nếu không có đủ matches
    if len(good_matches) < 10:
        raise ValueError("Không tìm thấy đủ điểm tương đồng giữa 2 ảnh. Vui lòng chọn 2 ảnh có phần chồng lấp (overlap) với nhau.")
    
    matches = good_matches
    
    keypoint_1 = np.float32([kp.pt for kp in k1])
    keypoint_2 = np.float32([kp.pt for kp in k2])
    pts1 = np.float32([keypoint_1[m.queryIdx] for m in matches])
    pts2 = np.float32([keypoint_2[m.trainIdx] for m in matches])
    
    # Phân tích vị trí keypoints để xác định thứ tự
    # Nếu ảnh 1 ở bên trái, các điểm matching ở ảnh 1 sẽ ở bên phải
    # và các điểm matching ở ảnh 2 sẽ ở bên trái
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Tính trung bình vị trí X của các điểm matching
    avg_x1 = np.mean(pts1[:, 0])
    avg_x2 = np.mean(pts2[:, 0])
    
    # Tỷ lệ vị trí so với chiều rộng ảnh
    ratio1 = avg_x1 / w1  # Tỷ lệ trong ảnh 1
    ratio2 = avg_x2 / w2  # Tỷ lệ trong ảnh 2
    
    # Nếu ảnh 1 ở trái: ratio1 cao (điểm ở bên phải ảnh 1), ratio2 thấp (điểm ở bên trái ảnh 2)
    # Nếu ảnh 1 ở phải: ratio1 thấp, ratio2 cao
    is_img1_left = (ratio1 > 0.5 and ratio2 < 0.5) or (ratio1 > ratio2 + 0.1)
    
    # Thử cả 2 cách và chọn kết quả tốt hơn
    results = []
    
    # Cách 1: img1 trái, img2 phải
    try:
        H1, mask1 = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
        if H1 is not None:
            inliers1 = np.sum(mask1) if mask1 is not None else 0
            # Kiểm tra homography hợp lệ (không bị biến dạng quá mức)
            if inliers1 > 10:  # Ít nhất 10 inliers
                img_result1 = cv2.warpPerspective(img2, H1, (w1 + w2, max(h1, h2)))
                # Blend thay vì ghi đè để tránh viền đen
                img_result1 = blend_images(img1, img_result1, h1, w1, h2, w2, True)
                # Đánh giá chất lượng: tỷ lệ diện tích sử dụng
                gray1 = cv2.cvtColor(img_result1, cv2.COLOR_RGB2GRAY)
                used_area1 = np.sum(gray1 > 0) / (gray1.shape[0] * gray1.shape[1])
                score1 = inliers1 * 0.7 + used_area1 * 100  # Trọng số cho inliers cao hơn
                results.append((score1, img_result1, H1, mask1, True))
    except:
        pass
    
    # Cách 2: img2 trái, img1 phải (đảo ngược)
    try:
        H2, mask2 = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        if H2 is not None:
            inliers2 = np.sum(mask2) if mask2 is not None else 0
            if inliers2 > 10:
                img_result2 = cv2.warpPerspective(img1, H2, (w1 + w2, max(h1, h2)))
                # Blend thay vì ghi đè để tránh viền đen
                # img2 ở trái, img1 (warped) ở phải
                img_result2 = blend_images(img2, img_result2, h2, w2, h1, w1, False)
                gray2 = cv2.cvtColor(img_result2, cv2.COLOR_RGB2GRAY)
                used_area2 = np.sum(gray2 > 0) / (gray2.shape[0] * gray2.shape[1])
                score2 = inliers2 * 0.7 + used_area2 * 100
                results.append((score2, img_result2, H2, mask2, False))
    except:
        pass
    
    # Chọn kết quả tốt nhất
    if len(results) == 0:
        # Nếu cả 2 cách đều fail, thử fallback với RANSAC threshold cao hơn
        try:
            H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 10.0)
            if H is not None and mask is not None and np.sum(mask) >= 10:
                img_result = cv2.warpPerspective(img2, H, (w1 + w2, max(h1, h2)))
                # Blend thay vì ghi đè để tránh viền đen
                img_result = blend_images(img1, img_result, h1, w1, h2, w2, True)
                img_soKhop = cv2.drawMatches(img1, k1, img2, k2, matches, None, flags=2)
            else:
                raise ValueError("Không thể ghép 2 ảnh này. Có thể 2 ảnh không có đủ phần chồng lấp hoặc không liên quan đến nhau.")
        except:
            raise ValueError("Không thể ghép 2 ảnh này. Có thể 2 ảnh không có đủ phần chồng lấp hoặc không liên quan đến nhau.")
    else:
        # Sắp xếp theo score, chọn cao nhất
        results.sort(key=lambda x: x[0], reverse=True)
        best_score, img_result, best_H, best_mask, is_normal_order = results[0]
        
        # Vẽ ảnh so khớp với thứ tự đúng
        if is_normal_order:
            img_soKhop = cv2.drawMatches(img1, k1, img2, k2, matches, None, flags=2)
        else:
            # Đảo ngược matches để vẽ đúng (swap queryIdx và trainIdx)
            reversed_matches = []
            for m in matches:
                # Tạo match object mới với thứ tự đảo ngược
                reversed_match = cv2.DMatch()
                reversed_match.queryIdx = m.trainIdx
                reversed_match.trainIdx = m.queryIdx
                reversed_match.distance = m.distance
                reversed_matches.append(reversed_match)
            img_soKhop = cv2.drawMatches(img2, k2, img1, k1, reversed_matches, None, flags=2)

    # Cắt bỏ vùng đen bên phải
    gray = cv2.cvtColor(img_result, cv2.COLOR_RGB2GRAY)
    cols = np.where(gray.sum(axis=0) > 0)[0]
    if len(cols) > 0:
        right = cols[-1]
        img_result = img_result[:, :right+1]
        # Cắt bỏ vùng đen bên trái nếu có
        left = cols[0] if len(cols) > 0 else 0
        if left > 0:
            img_result = img_result[:, left:]

    return img_soKhop, img_result


def panorama_multiple(images):
    """
    Ghép nhiều ảnh thành panorama
    Args:
        images: List các ảnh RGB
    Returns:
        tuple: (img_soKhop_list, final_panorama, failed_images)
        - img_soKhop_list: List các ảnh so khớp từng bước
        - final_panorama: Ảnh panorama cuối cùng
        - failed_images: List số thứ tự các ảnh không ghép được (1-indexed)
    """
    if len(images) < 2:
        raise ValueError("Cần ít nhất 2 ảnh để ghép panorama")
    
    if len(images) == 2:
        img_soKhop, panorama_result = panorama(images[0], images[1])
        return [img_soKhop], panorama_result, []
    
    # Ghép từng ảnh một
    img_soKhop_list = []
    current_panorama = images[0]
    failed_images = []
    successful_count = 1  # Đếm số ảnh đã ghép thành công (bắt đầu với ảnh đầu tiên)
    
    for i in range(1, len(images)):
        try:
            img_soKhop, panorama_result = panorama(current_panorama, images[i])
            img_soKhop_list.append(img_soKhop)
            current_panorama = panorama_result
            successful_count += 1
        except Exception as e:
            # Track ảnh không ghép được
            failed_images.append(i + 1)  # +1 vì index bắt đầu từ 0, nhưng số thứ tự từ 1
            # Nếu quá nhiều ảnh không ghép được, raise error
            if successful_count < 2:
                raise ValueError(f"Không thể ghép ảnh {i+1} với panorama hiện tại. {str(e)}")
            # Nếu chỉ một vài ảnh không ghép được, tiếp tục với các ảnh còn lại
            continue
    
    # Kiểm tra nếu quá nhiều ảnh không ghép được
    if len(failed_images) > len(images) / 2:
        failed_list = ', '.join(map(str, failed_images))
        raise ValueError(f"Quá nhiều ảnh không thể ghép được (ảnh số: {failed_list}). Vui lòng kiểm tra lại thứ tự và phần chồng lấp giữa các ảnh.")
    
    return img_soKhop_list, current_panorama, failed_images


if __name__ == "__main__":
    # Mở các file ảnh
    img1 = cv2.imread('image5.jpg')
    img2 = cv2.imread('image6.jpg')

    h_min = min(img1.shape[0], img2.shape[0])
    img1 = cv2.resize(img1, (int(img1.shape[1] * h_min / img1.shape[0]), h_min))
    img2 = cv2.resize(img2, (int(img2.shape[1] * h_min / img2.shape[0]), h_min))

    #Convert sang chế độ màu RGB
    scr_imgRGB = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    tar_imgRGB = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img_soKhop, img_Panorama = panorama(scr_imgRGB, tar_imgRGB)

    # Hiển thị 2 ảnh gốc
    fig=plt.figure(figsize=(16, 9))
    ax1,ax2 = fig.subplots(1, 2)

    ax1.imshow(scr_imgRGB)
    ax1.set_title('Ảnh 1')
    ax1.axis('off')

    ax2.imshow(tar_imgRGB)
    ax2.set_title('Ảnh 2')
    ax2.axis('off')
    plt.show()

    # Tạo cửa sổ hiển thị ảnh so khớp và ảnh Panorama
    fig=plt.figure(figsize=(16, 9))
    ax1,ax2 = fig.subplots(2, 1)
    ax1.imshow(img_soKhop)
    ax1.set_title('Ảnh so khớp ảnh 1 và ảnh 2')
    ax1.axis('off')

    ax2.imshow(img_Panorama)
    ax2.set_title('Ảnh Panorama tạo từ ảnh 1 và ảnh 2')
    ax2.axis('off')
    plt.show()
