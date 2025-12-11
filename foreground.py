import torch
import cv2
import os
import glob
import numpy as np
import urllib.request
import sys
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from tqdm import tqdm

INPUT_DIR = r"/kaggle/input/hackathon-2025"

OUTPUT_DIR = r"/kaggle/working/denso-ai-factory-2025/datasets/RealIAD/PCB5/DISthresh/good"

CHECKPOINT_PATH = "sam_vit_h_4b8939.pth" 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f" Đang chạy trên thiết bị: {DEVICE}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
if not os.path.exists(CHECKPOINT_PATH):
    print(f"Đang tải model SAM (2.4GB) về máy... Vui lòng chờ.")
    try:
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = (downloaded / total_size) * 100
            sys.stdout.write(f"\r   Tiến độ: {percent:.1f}%")
            sys.stdout.flush()
            
        urllib.request.urlretrieve(CHECKPOINT_URL, CHECKPOINT_PATH, show_progress)
        print("\n✅ Đã tải xong model!")
    except Exception as e:
        print(f"\nLỗi tải model: {e}")
        sys.exit(1)

print("Đang load model vào bộ nhớ...")
sam = sam_model_registry["vit_h"](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=16,         # Quét 32x32 điểm lưới
    pred_iou_thresh=0.88,         # Lọc mask chất lượng kém
    stability_score_thresh=0.95,  # Lọc mask không ổn định
    crop_n_layers=0,              # Tắt crop để nhanh hơn
    min_mask_region_area=500,     # Bỏ qua vùng quá nhỏ (rác)
)
print("Model đã sẵn sàng!")

def process_image(img_path):
    image = cv2.imread(img_path)
    if image is None: 
        return None
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image_rgb)

    if len(masks) == 0:
        return None
    sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    img_h, img_w = image.shape[:2]
    img_area = img_h * img_w
    best_mask = sorted_masks[0]
    if best_mask['area'] > (img_area * 0.95) and len(sorted_masks) > 1:
        best_mask = sorted_masks[1]
    mask_bool = best_mask['segmentation']
    mask_uint8 = (mask_bool * 255).astype(np.uint8)

    return mask_uint8
extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
image_paths = []
for ext in extensions:
    image_paths.extend(glob.glob(os.path.join(INPUT_DIR, ext)))

print(f"\n📂 Tìm thấy {len(image_paths)} ảnh trong thư mục đầu vào.")
print(f"📂 Kết quả sẽ lưu tại: {OUTPUT_DIR}")

count_success = 0
count_fail = 0

for img_path in tqdm(image_paths, desc="Đang xử lý"):
    # Lấy tên file gốc
    filename = os.path.basename(img_path)
    name_no_ext = os.path.splitext(filename)[0]
    output_filename = f"{name_no_ext}_mask.png"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    if os.path.exists(output_path):
        continue
        
    try:
        mask = process_image(img_path)
        
        if mask is not None:
            cv2.imwrite(output_path, mask)
            count_success += 1
        else:
            print(f"Không tìm thấy đối tượng trong: {filename}")
            count_fail += 1
            
    except Exception as e:
        print(f"Lỗi ngoại lệ file {filename}: {e}")
        count_fail += 1
        if "out of memory" in str(e).lower():
            print("Hết VRAM! Hãy thử giảm 'points_per_side' xuống 16 hoặc 12.")
            break

print("\n" + "="*50)
print(f"✅ HOÀN TẤT!")
print(f"📊 Thành công: {count_success}")
print(f"📊 Thất bại/Bỏ qua: {count_fail}")
print(f"📂 Thư mục chứa mask: {OUTPUT_DIR}")
print("="*50)

# pip install torch torchvision
# pip install git+https://github.com/facebookresearch/segment-anything.git
# pip install opencv-python matplotlib tqdm