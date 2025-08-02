import os
import cv2
import random
import numpy as np

def augement_image(img):
    return[
        cv2.flip(img, 1),  # Horizontal flip
        cv2.flip(img, 0),  # Vertical flip
        cv2.flip(img, -1)
    ]

def deblur_image(img):
    # De-blur with sharpen filter.
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(img, -1, kernel)
    return sharpened

def process_images(folder_path, target_size=(512, 512), target_count=100):
    folder_name = os.path.basename(folder_path.rstrip('/\\'))
    prefix = folder_name.lower().replace(" ", "_")
    print(f"\n🔧 Đang xử lý thư mục: {folder_name}")

    images = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue
            resized = cv2.resize(img, target_size)
            deblurred = deblur_image(resized)
            images.append(deblurred)

    # Record resized and deblurred image
    for idx, img in enumerate(images):
        filename = f"img_{prefix}_{idx + 1:03d}.jpg"
        cv2.imwrite(os.path.join(folder_path, filename), img)

    current_count = len(images)
    next_idx = current_count + 1

    while current_count < target_count:
        img = random.choice(images)
        for aug in augement_image(img):
            if current_count >= target_count:
                break
            filename = f"{prefix}_{next_idx:03d}.jpg"
            cv2.imwrite(os.path.join(folder_path, filename), aug)
            next_idx += 1
            current_count += 1

    print(f"Xong! {target_count} ảnh sau khi resize + khử mờ + augment.")

    folder_name = os.path.basename(folder_path.rstrip('/\\'))
    prefix = folder_name.lower().replace(' ', '_')
    print(f"Processing folder: {folder_name}")

    images = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue
            resized = cv2.resize(img, target_size)
            images.append(resized)

    for idx, img in enumerate(images):
        filename = f"{prefix}_{idx + 1:03d}.jpg"
        cv2.imwrite(os.path.join(folder_path, filename), img)

    current_count = len(images)
    next_idx = current_count + 1

    while current_count < target_count:
        img = random.choice(images)
        for aug in augement_image(img):
            if current_count >= target_count:
                break
            filename = f"{prefix}_{next_idx:03d}.jpg"
            cv2.imwrite(os.path.join(folder_path, filename), aug)
            next_idx += 1
            current_count += 1
    print(f"Processed {current_count} images in folder: {folder_name}")

# process_images("root/childs")
