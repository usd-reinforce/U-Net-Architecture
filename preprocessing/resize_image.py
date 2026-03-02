import cv2

def resize_image(image_path, target_size=(256, 256)):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Gambar tidak ditemukan. Cek kembali path/lokasi file.")
    resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
    return resized_img