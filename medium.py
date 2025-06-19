import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2


def threshold_image(im, th):
    """Threshold pada gambar, menghasilkan gambar biner (0 atau 255)."""
    # Diubah agar menghasilkan 0 dan 255 untuk kompatibilitas dengan cv2 dan plotting
    thresholded_im = np.zeros(im.shape, dtype=np.uint8)
    thresholded_im[im >= th] = 255
    return thresholded_im

def compute_otsu_criteria(im, th):
    """Hitung kriteria Otsu (varians intra-kelas) untuk threshold tertentu."""
    # Mirip kaya medium yg dikasih ibunya di hebat, tapi lebih efisien
    foreground = im[im >= th]
    background = im[im < th]
    
    if foreground.size == 0 or background.size == 0:
        return np.inf
        
    weight_fg = foreground.size / im.size
    weight_bg = background.size / im.size
    
    variance_fg = np.var(foreground) if foreground.size > 0 else 0
    variance_bg = np.var(background) if background.size > 0 else 0
    
    return (weight_fg * variance_fg) + (weight_bg * variance_bg)

def find_best_threshold(im):
    """Mencari threshold terbaik dengan mencoba semua kemungkinan dan meminimalkan kriteria Otsu."""
    threshold_range = range(np.max(im) + 1)
    criterias = [compute_otsu_criteria(im, th) for th in threshold_range]
    best_threshold = threshold_range[np.argmin(criterias)]
    return best_threshold


def apply_post_processing(binary_image):
    """Bersihin gambar biner pake operasi morfologi"""
    # Kernel 3x3 lebih 'lembut' untuk menghindari penghapusan detail objek.
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    return opening


def main():
    # Ganti path gambar di bawah ini sesuai dengan file yang ingin diproses.
    # ganti: pake os.path biar konsisten buat Windows ataupun Ubuntu
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, 'merpati.jpg') 
    
    try:
        original_image = Image.open(image_path)
        gray_image = original_image.convert('L')
        gray_image_array = np.asarray(gray_image)

        # 1. Pra-pemrosesan
        blurred_image_array = cv2.GaussianBlur(gray_image_array, (5, 5), 0)

        # 2. Segmentasi otsu (menggunakan implementasi manual)
        print("Mencari threshold optimal dengan metode manual (dari medium.py)...")
        best_thresh = find_best_threshold(blurred_image_array)
        raw_binary = threshold_image(gray_image_array, best_thresh)

        # 3. Penyesuaian cerdas (Inversi jika perlu)
        foreground_pixels = np.count_nonzero(raw_binary)
        if foreground_pixels > raw_binary.size / 2:
            print("Info: Hasil dibalik untuk memastikan objek (area kecil) berwarna putih.")
            raw_binary = cv2.bitwise_not(raw_binary)

        # 4. Pasca-pemrosesan
        print("Membersihkan hasil dengan operasi morfologi...")
        final_image = apply_post_processing(raw_binary)

        print(f"Threshold optimal yang ditemukan: {best_thresh}")
        
        # 5. Menampilkan hasil
        plt.figure(figsize=(20, 5))

        plt.subplot(1, 4, 1)
        plt.title('Gambar Asli')
        plt.imshow(original_image)
        plt.axis('off')

        plt.subplot(1, 4, 2)
        plt.title('Gambar Grayscale')
        plt.imshow(gray_image, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 4, 3)
        plt.title('Hasil Otsu (Mentah)')
        plt.imshow(raw_binary, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 4, 4)
        plt.title('Hasil Akhir (Dibersihkan)')
        plt.imshow(final_image, cmap='gray')
        plt.axis('off')

        plt.suptitle('Segmentasi dengan Otsu Manual & Pasca-pemrosesan', fontsize=16)
        plt.show()

    except FileNotFoundError:
        print(f"Error: Gambar tidak ada di path: {image_path}")
    except Exception as e:
        print(f"Ada error: {e}")

if __name__ == "__main__":
    main()
