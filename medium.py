# Vito Fajar Wibawa Solin        -    163221020
# Patrick Andrasena Tumengkol    -    163221077
# Fellysha Fernanda              -    163221098

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os
import random

class GeneticAlgorithmOtsu:
    def __init__(self, population_size=50, generations=100, mutation_rate=0.1, crossover_rate=0.8):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
    def create_initial_population(self, min_val, max_val):
        """Membuat populasi awal dengan threshold acak"""
        population = []
        for _ in range(self.population_size):
            threshold = random.randint(min_val, max_val)
            population.append(threshold)
        return population
    
    def fitness_function(self, image, threshold):
        """Fungsi fitness berdasarkan kriteria Otsu (minimize intra-class variance)"""
        foreground = image[image >= threshold]
        background = image[image < threshold]
        
        if foreground.size == 0 or background.size == 0:
            return 0  # Fitness rendah untuk threshold yang tidak valid
            
        # Hitung bobot
        weight_fg = foreground.size / image.size
        weight_bg = background.size / image.size
        
        # Hitung varians
        variance_fg = np.var(foreground) if foreground.size > 0 else 0
        variance_bg = np.var(background) if background.size > 0 else 0
        
        # Intra-class variance (yang ingin diminimalkan)
        intra_class_variance = (weight_fg * variance_fg) + (weight_bg * variance_bg)
        
        # Return fitness (semakin kecil intra-class variance, semakin tinggi fitness)
        return 1 / (1 + intra_class_variance)
    
    def selection(self, population, fitness_scores):
        """Tournament selection untuk memilih parent"""
        tournament_size = 3
        selected = []
        
        for _ in range(2):  # Pilih 2 parent
            tournament = random.sample(list(zip(population, fitness_scores)), tournament_size)
            winner = max(tournament, key=lambda x: x[1])
            selected.append(winner[0])
        
        return selected
    
    def crossover(self, parent1, parent2):
        """Single-point crossover untuk threshold values"""
        if random.random() < self.crossover_rate:
            # Arithmetic crossover untuk nilai threshold
            alpha = random.random()
            child1 = int(alpha * parent1 + (1 - alpha) * parent2)
            child2 = int(alpha * parent2 + (1 - alpha) * parent1)
            return child1, child2
        else:
            return parent1, parent2
    
    def mutation(self, individual, min_val, max_val):
        """Mutasi dengan penambahan noise kecil"""
        if random.random() < self.mutation_rate:
            # Mutasi Gaussian
            noise = int(np.random.normal(0, 5))  # Noise kecil
            mutated = individual + noise
            # Pastikan dalam range yang valid
            mutated = max(min_val, min(max_val, mutated))
            return mutated
        return individual
    
    def optimize_threshold(self, image):
        """Optimasi threshold menggunakan genetic algorithm"""
        min_val = int(np.min(image))
        max_val = int(np.max(image))
        
        # Inisialisasi populasi
        population = self.create_initial_population(min_val, max_val)
        best_fitness_history = []
        
        print(f"Memulai optimasi GA dengan populasi {self.population_size} untuk {self.generations} generasi...")
        
        for generation in range(self.generations):
            # Evaluasi fitness
            fitness_scores = [self.fitness_function(image, threshold) for threshold in population]
            
            # Tracking best fitness
            best_fitness = max(fitness_scores)
            best_fitness_history.append(best_fitness)
            
            if generation % 20 == 0:
                best_threshold = population[np.argmax(fitness_scores)]
                print(f"Generasi {generation}: Best threshold = {best_threshold}, Fitness = {best_fitness:.6f}")
            
            new_population = []
            #menetapkan individu yang terbaik
            best_idx = np.argmax(fitness_scores)
            new_population.append(population[best_idx])
            
            # Generate rest of population
            while len(new_population) < self.population_size:
                parent1, parent2 = self.selection(population, fitness_scores)
                child1, child2 = self.crossover(parent1, parent2)
                
                child1 = self.mutation(child1, min_val, max_val)
                child2 = self.mutation(child2, min_val, max_val)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
        
        final_fitness_scores = [self.fitness_function(image, threshold) for threshold in population]
        best_threshold = population[np.argmax(final_fitness_scores)]
        
        return best_threshold, best_fitness_history

def threshold_image(im, th):
    """Threshold pada gambar, menghasilkan gambar biner (0 atau 255)."""
    # Diubah agar menghasilkan 0 dan 255 untuk kompatibilitas dengan cv2 dan plotting
    thresholded_im = np.zeros(im.shape, dtype=np.uint8)
    thresholded_im[im >= th] = 255
    return thresholded_im

# metode kalkulasi otsu utama
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

def find_traditional_otsu_threshold(im):
    # sama seperti sebelumnya, exhaustive search (penelusuran menyeluruh) = otsu + exhaustive
    """Otsu tradisional untuk dibandingkan."""
    threshold_range = range(int(np.min(im)), int(np.max(im)) + 1)
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
    # Ganti path gambar di bawah ini sesuai dengan file yang ingin diproses. (merpati.jpg)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, 'merpati.jpg') 
    
    try:
        original_image = Image.open(image_path)
        gray_image = original_image.convert('L')
        gray_image_array = np.asarray(gray_image)

        # 1. Pra-pemrosesan
        blurred_image_array = cv2.GaussianBlur(gray_image_array, (5, 5), 0)

        # 2. Segmentasi Genetic Algorithm
        print("=== OPTIMASI THRESHOLD DENGAN GENETIC ALGORITHM ===")
        ga_optimizer = GeneticAlgorithmOtsu(
            population_size=50,
            generations=100,
            mutation_rate=0.1,
            crossover_rate=0.8
        )
        
        ga_threshold, fitness_history = ga_optimizer.optimize_threshold(blurred_image_array)
        ga_binary = threshold_image(gray_image_array, ga_threshold)
        
        # 3. Perbandingan dengan Otsu tradisional
        print("\n=== PERBANDINGAN DENGAN METODE TRADISIONAL ===")
        traditional_threshold = find_traditional_otsu_threshold(blurred_image_array)
        traditional_binary = threshold_image(gray_image_array, traditional_threshold)
        
        print(f"GA Threshold: {ga_threshold}")
        print(f"Traditional Otsu Threshold: {traditional_threshold}")
        print(f"Perbedaan: {abs(ga_threshold - traditional_threshold)}")
        
        # 4. Penyesuaian untuk metode GA
        foreground_pixels_ga = np.count_nonzero(ga_binary)
        if foreground_pixels_ga > ga_binary.size / 2:
            print("Info: Hasil GA dibalik untuk memastikan objek berwarna putih.")
            ga_binary = cv2.bitwise_not(ga_binary)
        
        # 5. Penyesuaian untuk metode Traditional
        foreground_pixels_trad = np.count_nonzero(traditional_binary)
        if foreground_pixels_trad > traditional_binary.size / 2:
            print("Info: Hasil Traditional dibalik untuk memastikan objek berwarna putih.")
            traditional_binary = cv2.bitwise_not(traditional_binary)

        # 6. Pasca-pemrosesan
        ga_final = apply_post_processing(ga_binary)
        traditional_final = apply_post_processing(traditional_binary)
        
        # 7. Visualisasi hasil
        
        # Slide 1: Perbandingan hasil
        fig1, axes1 = plt.subplots(1, 6, figsize=(24, 5))
        fig1.suptitle('Perbandingan Genetic Algorithm vs Traditional Otsu Thresholding', fontsize=16)

        axes1[0].set_title('Gambar Asli', fontsize=12)
        axes1[0].imshow(original_image)
        axes1[0].axis('off')

        axes1[1].set_title('Gambar Grayscale', fontsize=12)
        axes1[1].imshow(gray_image, cmap='gray')
        axes1[1].axis('off')

        axes1[2].set_title(f'GA Raw Binary\n(Threshold: {ga_threshold})', fontsize=12)
        axes1[2].imshow(ga_binary, cmap='gray')
        axes1[2].axis('off')

        axes1[3].set_title('GA Final Result\n(Cleaned)', fontsize=12)
        axes1[3].imshow(ga_final, cmap='gray')
        axes1[3].axis('off')

        axes1[4].set_title(f'Traditional Raw Binary\n(Threshold: {traditional_threshold})', fontsize=12)
        axes1[4].imshow(traditional_binary, cmap='gray')
        axes1[4].axis('off')

        axes1[5].set_title('Traditional Final Result\n(Cleaned)', fontsize=12)
        axes1[5].imshow(traditional_final, cmap='gray')
        axes1[5].axis('off')

        fig1.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle

        # Slide 2: Analisis
        fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

        axes2[0].set_title('GA Fitness Evolution', fontsize=12)
        axes2[0].plot(fitness_history)
        axes2[0].set_xlabel('Generation')
        axes2[0].set_ylabel('Best Fitness')
        axes2[0].grid(True)

        axes2[1].set_title('Histogram Perbandingan', fontsize=12)
        axes2[1].hist(gray_image_array.flatten(), bins=256, alpha=0.7, label='Original', color='blue')
        axes2[1].axvline(x=ga_threshold, color='red', linestyle='--', label=f'GA Threshold ({ga_threshold})')
        axes2[1].axvline(x=traditional_threshold, color='green', linestyle='--', label=f'Traditional ({traditional_threshold})')
        axes2[1].set_xlabel('Pixel Intensity')
        axes2[1].set_ylabel('Frequency')
        axes2[1].legend()
        axes2[1].grid(True)

        axes2[2].set_title('Perbedaan Hasil', fontsize=12)
        diff = np.abs(ga_final.astype(int) - traditional_final.astype(int))
        im = axes2[2].imshow(diff, cmap='hot')
        fig2.colorbar(im, ax=axes2[2], fraction=0.046, pad=0.04, label='Difference')
        axes2[2].axis('off')

        fig2.tight_layout()

        plt.show()

        # 8. Analisis kuantitatif
        print("\n=== ANALISIS KUANTITATIF ===")
        
        # Hitung kriteria Otsu untuk kedua hasil
        ga_criteria = compute_otsu_criteria(gray_image_array, ga_threshold)
        traditional_criteria = compute_otsu_criteria(gray_image_array, traditional_threshold)
        
        print(f"GA Otsu Criteria (Intra-class variance): {ga_criteria:.6f}")
        print(f"Traditional Otsu Criteria: {traditional_criteria:.6f}")
        
        if ga_criteria < traditional_criteria:
            print("✓ GA menghasilkan kriteria Otsu yang lebih baik (variance lebih kecil)")
        else:
            print("✓ Traditional Otsu menghasilkan kriteria yang lebih baik")
        
        # Hitung jumlah piksel yang berbeda
        pixels_different = np.sum(ga_final != traditional_final)
        total_pixels = ga_final.size
        difference_percentage = (pixels_different / total_pixels) * 100
        
        print(f"Piksel yang berbeda: {pixels_different} dari {total_pixels} ({difference_percentage:.2f}%)")

    except FileNotFoundError:
        print(f"Error: Gambar tidak ada di path: {image_path}")
    except Exception as e:
        print(f"Ada error: {e}")

if __name__ == "__main__":
    main()
