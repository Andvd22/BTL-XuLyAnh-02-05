// Quản lý số lượng ảnh
let imageCount = 2;
const MAX_IMAGES = 10;
const MIN_IMAGES = 2;

// Khởi tạo 2 ảnh đầu tiên
document.addEventListener('DOMContentLoaded', function() {
    initializeUploadBox();
});

function initializeUploadBox() {
    const uploadBox = document.getElementById('uploadBox');
    uploadBox.innerHTML = '';
    
    for (let i = 1; i <= imageCount; i++) {
        addImageInput(i);
    }
    
    updateImageCount();
    setupEventListeners();
}

function addImageInput(index) {
    const uploadBox = document.getElementById('uploadBox');
    const uploadItem = document.createElement('div');
    uploadItem.className = 'upload-item';
    uploadItem.id = `upload-item-${index}`;
    
    uploadItem.innerHTML = `
        <label for="image${index}" class="upload-label">
            <div class="upload-placeholder">
                <span class="upload-icon">📷</span>
                <span class="upload-text">Chọn Ảnh ${index}</span>
            </div>
            <input type="file" id="image${index}" name="image${index}" accept="image/*" required>
        </label>
        <div id="preview${index}" class="preview"></div>
    `;
    
    uploadBox.appendChild(uploadItem);
    
    // Setup preview cho ảnh mới
    const input = document.getElementById(`image${index}`);
    input.addEventListener('change', function(e) {
        previewImage(e.target.files[0], `preview${index}`);
    });
}

function setupEventListeners() {
    // Thêm ảnh
    document.getElementById('addImageBtn').addEventListener('click', function() {
        if (imageCount < MAX_IMAGES) {
            imageCount++;
            addImageInput(imageCount);
            updateImageCount();
            updateRemoveButton();
        } else {
            showError(`Tối đa ${MAX_IMAGES} ảnh`);
        }
    });
    
    // Xóa ảnh cuối
    document.getElementById('removeImageBtn').addEventListener('click', function() {
        if (imageCount > MIN_IMAGES) {
            const uploadItem = document.getElementById(`upload-item-${imageCount}`);
            if (uploadItem) {
                uploadItem.remove();
            }
            imageCount--;
            updateImageCount();
            updateRemoveButton();
        }
    });
    
    // Form submit
    document.getElementById('uploadForm').addEventListener('submit', handleSubmit);
    
    // Download button
    document.getElementById('downloadBtn').addEventListener('click', function() {
        if (window.panoramaImageData) {
            const link = document.createElement('a');
            link.href = window.panoramaImageData;
            link.download = 'panorama.jpg';
            link.click();
        }
    });
}

function updateImageCount() {
    document.getElementById('imageCount').textContent = imageCount;
}

function updateRemoveButton() {
    const removeBtn = document.getElementById('removeImageBtn');
    removeBtn.style.display = imageCount > MIN_IMAGES ? 'block' : 'none';
}

function previewImage(file, previewId) {
    if (file) {
        const maxSize = 100 * 1024 * 1024; // 100MB
        if (file.size > maxSize) {
            showError(`Ảnh quá lớn (${(file.size / 1024 / 1024).toFixed(2)}MB). Vui lòng chọn ảnh nhỏ hơn 100MB.`);
            const inputId = previewId.replace('preview', 'image');
            document.getElementById(inputId).value = '';
            return;
        }
        
        const preview = document.getElementById(previewId);
        const fileSizeMB = (file.size / 1024 / 1024).toFixed(2);
        const fileName = file.name.toLowerCase();
        
        const rawFormats = ['.dng', '.cr2', '.nef', '.arw', '.orf', '.raf', '.rw2', '.srw', '.pef', '.x3f'];
        const isRawFile = rawFormats.some(format => fileName.endsWith(format));
        
        if (isRawFile || file.size > 20 * 1024 * 1024) {
            const fileType = isRawFile ? 'RAW' : 'lớn';
            preview.innerHTML = `
                <div style="text-align: center; padding: 15px; color: #b8c5d6; border: 2px dashed rgba(255, 107, 53, 0.5); border-radius: 12px; background: rgba(255, 107, 53, 0.1); margin-top: 10px;">
                    <div style="font-size: 2em; margin-bottom: 8px;">📷</div>
                    <div style="font-weight: 600; margin-bottom: 5px; font-size: 0.9em; color: #ffffff;">${file.name}</div>
                    <div style="font-size: 0.8em; color: #6b7a8f;">${fileSizeMB} MB ${isRawFile ? '(RAW)' : ''}</div>
                    <div style="font-size: 0.7em; color: #00d4aa; margin-top: 5px;">✓ Hợp lệ</div>
                </div>
            `;
            preview.classList.add('show');
            return;
        }
        
        preview.innerHTML = '<div style="text-align: center; padding: 10px; color: #ff6b35;">Đang tải...</div>';
        preview.classList.add('show');
        
        const reader = new FileReader();
        reader.onload = function(e) {
            const img = new Image();
            img.onload = function() {
                try {
                    const canvas = document.createElement('canvas');
                    const ctx = canvas.getContext('2d');
                    const maxWidth = 300;
                    let width = img.width;
                    let height = img.height;
                    
                    if (width > maxWidth) {
                        height = (height * maxWidth) / width;
                        width = maxWidth;
                    }
                    
                    canvas.width = width;
                    canvas.height = height;
                    ctx.drawImage(img, 0, 0, width, height);
                    const resizedDataUrl = canvas.toDataURL('image/jpeg', 0.85);
                    
                    preview.innerHTML = `
                        <img src="${resizedDataUrl}" alt="Preview" style="max-width: 100%; height: auto; border-radius: 12px; margin-top: 10px; border: 1px solid rgba(255, 255, 255, 0.1);">
                        <div style="text-align: center; margin-top: 8px; font-size: 0.8em; color: #6b7a8f;">${fileSizeMB} MB</div>
                    `;
                } catch (error) {
                    preview.innerHTML = `
                        <img src="${e.target.result}" alt="Preview" style="max-width: 100%; height: auto; max-height: 200px; border-radius: 12px; margin-top: 10px; border: 1px solid rgba(255, 255, 255, 0.1);">
                        <div style="text-align: center; margin-top: 8px; font-size: 0.8em; color: #6b7a8f;">${fileSizeMB} MB</div>
                    `;
                }
            };
            img.onerror = function() {
                preview.innerHTML = `
                    <div style="text-align: center; padding: 15px; color: #b8c5d6; border: 2px dashed rgba(255, 107, 53, 0.5); border-radius: 12px; background: rgba(255, 107, 53, 0.1); margin-top: 10px;">
                        <div style="font-size: 2em; margin-bottom: 8px;">📷</div>
                        <div style="font-weight: 600; font-size: 0.9em; color: #ffffff;">${file.name}</div>
                        <div style="font-size: 0.8em; color: #6b7a8f; margin-top: 5px;">${fileSizeMB} MB</div>
                        <div style="font-size: 0.7em; color: #00d4aa; margin-top: 5px;">✓ Hợp lệ</div>
                    </div>
                `;
            };
            img.src = e.target.result;
        };
        reader.onerror = function() {
            preview.innerHTML = '<div style="text-align: center; padding: 10px; color: #ff4757;">Lỗi đọc file</div>';
        };
        reader.readAsDataURL(file);
    }
}

async function handleSubmit(e) {
    e.preventDefault();
    
    const formData = new FormData();
    let hasFiles = false;
    
    // Lấy tất cả file ảnh
    for (let i = 1; i <= imageCount; i++) {
        const input = document.getElementById(`image${i}`);
        if (input && input.files[0]) {
            formData.append(`image${i}`, input.files[0]);
            hasFiles = true;
        }
    }
    
    if (!hasFiles) {
        showError('Vui lòng chọn ít nhất 2 ảnh');
        return;
    }
    
    // Hiển thị loading
    const submitBtn = document.getElementById('submitBtn');
    const btnText = document.getElementById('btnText');
    const btnLoader = document.getElementById('btnLoader');
    
    submitBtn.disabled = true;
    btnText.textContent = 'Đang xử lý...';
    btnLoader.style.display = 'inline-block';
    
    document.getElementById('results').style.display = 'none';
    document.getElementById('errorMessage').style.display = 'none';
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        if (response.status === 413) {
            showError('Ảnh quá lớn! Vui lòng chọn ảnh nhỏ hơn 100MB.');
            return;
        }
        
        const data = await response.json();
        
        if (data.success) {
            // Hiển thị cảnh báo nếu có ảnh không ghép được
            if (data.warning) {
                showWarning(data.warning, data.failed_images);
            }
            displayResults(data);
        } else {
            showError(data.error || 'Có lỗi xảy ra khi xử lý ảnh');
        }
    } catch (error) {
        showError('Lỗi kết nối: ' + error.message);
    } finally {
        submitBtn.disabled = false;
        btnText.textContent = 'Bắt Đầu Ghép Ảnh';
        btnLoader.style.display = 'none';
    }
}

function displayResults(data) {
    // Hiển thị ảnh gốc
    const originalImagesDiv = document.getElementById('originalImages');
    originalImagesDiv.innerHTML = '';
    
    data.images.forEach((imgSrc, index) => {
        const imageCard = document.createElement('div');
        imageCard.className = 'image-card';
        imageCard.innerHTML = `
            <img src="${imgSrc}" alt="Ảnh ${index + 1}">
            <p>Ảnh ${index + 1}</p>
        `;
        originalImagesDiv.appendChild(imageCard);
    });
    
    // Hiển thị ảnh so khớp (nếu có)
    const matchSection = document.getElementById('matchSection');
    const matchImagesDiv = document.getElementById('matchImages');
    
    if (data.match_images && data.match_images.length > 0) {
        matchSection.style.display = 'block';
        matchImagesDiv.innerHTML = '';
        
        data.match_images.forEach((imgSrc, index) => {
            const imageCard = document.createElement('div');
            imageCard.className = 'image-card';
            imageCard.innerHTML = `
                <img src="${imgSrc}" alt="So khớp ${index + 1}">
                <p>Bước ${index + 1}</p>
            `;
            matchImagesDiv.appendChild(imageCard);
        });
    } else {
        matchSection.style.display = 'none';
    }
    
    // Hiển thị panorama
    document.getElementById('panoramaImage').src = data.panorama;
    window.panoramaImageData = data.panorama;
    
    document.getElementById('results').style.display = 'block';
    document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
    
    // Reset form về trạng thái ban đầu
    resetUploadForm();
}

function resetUploadForm() {
    // Reset về 2 ảnh ban đầu
    const uploadBox = document.getElementById('uploadBox');
    const currentItems = uploadBox.querySelectorAll('.upload-item');
    
    // Xóa tất cả ảnh hiện tại
    currentItems.forEach(item => item.remove());
    
    // Reset số lượng ảnh về 2
    imageCount = 2;
    updateImageCount();
    updateRemoveButton();
    
    // Tạo lại 2 input ảnh ban đầu
    for (let i = 1; i <= 2; i++) {
        addImageInput(i);
    }
    
    // Reset form
    document.getElementById('uploadForm').reset();
}

function showError(message) {
    const errorDiv = document.getElementById('errorMessage');
    errorDiv.className = 'alert error';
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
    errorDiv.scrollIntoView({ behavior: 'smooth' });
}

function showWarning(message, failedImages) {
    const errorDiv = document.getElementById('errorMessage');
    errorDiv.className = 'alert warning';
    errorDiv.innerHTML = `
        <strong>⚠️ Lưu ý:</strong> ${message}
        ${failedImages ? `<br><small>Ảnh không ghép được: ${failedImages.join(', ')}</small>` : ''}
    `;
    errorDiv.style.display = 'block';
    errorDiv.scrollIntoView({ behavior: 'smooth' });
}
