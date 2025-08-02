import os

def rename_images_in_folder(folder_path, prefix="image"):
    # Get list of image files
    files = os.listdir(folder_path)
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    image_files.sort()

    # Step 1: Change all photos to temporary names to avoid duplicate names
    temp_names = []
    for idx, filename in enumerate(image_files):
        ext = os.path.splitext(filename)[1]
        temp_name = f"__temp_{idx}{ext}"
        src = os.path.join(folder_path, filename)
        dst = os.path.join(folder_path, temp_name)
        os.rename(src, dst)
        temp_names.append(temp_name)

    # Step 2: Change from temporary name to new standard name
    for idx, temp_name in enumerate(temp_names, start=1):
        ext = os.path.splitext(temp_name)[1]
        new_name = f"{prefix}_{idx}{ext}"
        src = os.path.join(folder_path, temp_name)
        dst = os.path.join(folder_path, new_name)
        os.rename(src, dst)

    print("All photos have been renamed.")

# rename_images_in_folder("root/childs", prefix="child")