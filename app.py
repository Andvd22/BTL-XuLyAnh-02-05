from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
from Panorama import panorama, panorama_multiple
import os
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size (tăng từ 16MB)

# Tạo thư mục uploads nếu chưa có
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def read_image(file):
    """Đọc ảnh từ file upload, hỗ trợ JPG, PNG, và các định dạng khác"""
    try:
        # Đọc bytes
        img_bytes = file.read()
        file.seek(0)  # Reset file pointer để có thể đọc lại nếu cần
        
        # Kiểm tra định dạng file
        filename = file.filename.lower() if hasattr(file, 'filename') else ''
        is_raw = any(filename.endswith(ext) for ext in ['.dng', '.cr2', '.nef', '.arw', '.orf', '.raf', '.rw2', '.srw', '.pef', '.x3f'])
        
        if is_raw:
            # Thử dùng PIL để đọc RAW (một số RAW được hỗ trợ)
            try:
                pil_image = Image.open(BytesIO(img_bytes))
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                img_array = np.array(pil_image)
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                return img_bgr
            except Exception as e:
                raise Exception(f"Không thể đọc file RAW. Vui lòng chuyển đổi sang JPG/PNG trước. Lỗi: {str(e)}")
        
        # Thử dùng PIL để đọc (hỗ trợ tốt PNG có alpha)
        try:
            pil_image = Image.open(BytesIO(img_bytes))
            # Chuyển RGBA sang RGB nếu có alpha channel
            if pil_image.mode == 'RGBA':
                # Tạo background trắng
                rgb_image = Image.new('RGB', pil_image.size, (255, 255, 255))
                rgb_image.paste(pil_image, mask=pil_image.split()[3])  # Dùng alpha làm mask
                pil_image = rgb_image
            elif pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Chuyển PIL Image sang numpy array
            img_array = np.array(pil_image)
            # PIL trả về RGB, OpenCV cần BGR
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            return img_bgr
        except:
            # Nếu PIL không đọc được, thử OpenCV
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                # Thử đọc với flag khác
                img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                if img is not None and len(img.shape) == 3 and img.shape[2] == 4:
                    # Ảnh có alpha channel (BGRA), chuyển sang BGR
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            if img is None:
                raise Exception("Không thể đọc ảnh. Vui lòng kiểm tra định dạng file (hỗ trợ: JPG, PNG, BMP, TIFF)")
            return img
    except Exception as e:
        raise Exception(f"Không thể đọc ảnh: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_images():
    try:
        # Lấy tất cả các file ảnh từ request
        files = []
        file_keys = [key for key in request.files.keys() if key.startswith('image')]
        file_keys.sort(key=lambda x: int(x.replace('image', '')) if x.replace('image', '').isdigit() else 0)
        
        if len(file_keys) < 2:
            return jsonify({'error': 'Vui lòng upload ít nhất 2 ảnh'}), 400
        
        # Đọc tất cả ảnh
        images = []
        for key in file_keys:
            file = request.files[key]
            if file.filename == '':
                continue
            
            try:
                img = read_image(file)
                if img is not None:
                    images.append(img)
            except Exception as e:
                return jsonify({'error': f'Lỗi đọc ảnh {key}: {str(e)}'}), 400
        
        if len(images) < 2:
            return jsonify({'error': 'Cần ít nhất 2 ảnh hợp lệ để ghép panorama'}), 400
        
        # Resize tất cả ảnh về cùng chiều cao
        h_min = min(img.shape[0] for img in images)
        images_resized = []
        for img in images:
            resized = cv2.resize(img, (int(img.shape[1] * h_min / img.shape[0]), h_min))
            images_resized.append(resized)
        
        # Chuyển BGR sang RGB
        images_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images_resized]
        
        # Ghép panorama
        failed_images = []
        if len(images_rgb) == 2:
            # Ghép 2 ảnh
            img_soKhop, img_panorama = panorama(images_rgb[0], images_rgb[1])
            match_images_list = [img_soKhop]
        else:
            # Ghép nhiều ảnh
            match_images_list, img_panorama, failed_images = panorama_multiple(images_rgb)
        
        # Chuyển ảnh kết quả về BGR để encode
        match_images_bgr = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in match_images_list]
        img_panorama_bgr = cv2.cvtColor(img_panorama, cv2.COLOR_RGB2BGR)
        
        # Encode ảnh thành base64
        encode_format = '.jpg'
        mime_type = 'image/jpeg'
        
        # Encode ảnh gốc
        images_base64 = []
        for img in images_resized:
            _, buffer = cv2.imencode(encode_format, img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            images_base64.append(f'data:{mime_type};base64,' + img_base64)
        
        # Encode ảnh so khớp
        match_images_base64 = []
        for img in match_images_bgr:
            _, buffer = cv2.imencode(encode_format, img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            match_images_base64.append(f'data:{mime_type};base64,' + img_base64)
        
        # Encode ảnh panorama
        _, buffer_panorama = cv2.imencode(encode_format, img_panorama_bgr)
        panorama_base64 = base64.b64encode(buffer_panorama).decode('utf-8')
        
        response_data = {
            'success': True,
            'images': images_base64,
            'match_images': match_images_base64,
            'panorama': f'data:{mime_type};base64,' + panorama_base64,
            'num_images': len(images)
        }
        
        # Thêm thông tin về ảnh không ghép được (nếu có)
        if failed_images:
            response_data['warning'] = f"Lưu ý: {len(failed_images)} ảnh không thể ghép được (ảnh số: {', '.join(map(str, failed_images))}). Panorama được tạo từ các ảnh còn lại."
            response_data['failed_images'] = failed_images
        
        return jsonify(response_data)
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Lỗi xử lý: {str(e)}'}), 500

# Xử lý lỗi 413 (Request Entity Too Large)
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'Ảnh quá lớn! Vui lòng chọn ảnh nhỏ hơn 100MB hoặc resize ảnh trước khi upload.'}), 413

if __name__ == '__main__':
    app.run(debug=True, port=5000)