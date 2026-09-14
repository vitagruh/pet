"""
Reconstruction Simulation Module for PET Analysis
Модуль симуляции итеративной реконструкции OSEM

Физические основы:
- OSEM (Ordered Subsets Expectation Maximization) - итеративный алгоритм
- Количество итераций влияет на сходимость и шум
- Подмножества ускоряют сходимость, но могут вносить артефакты
- Размер матрицы влияет на пространственное разрешение

Ограничение: Без raw sinogram данных полная OSEM невозможна.
Реализована научно обоснованная аппроксимация:
- Итеративное улучшение резкости (деблюринг)
- Аддитивный/мультипликативный шум, калиброванный по литературе
- Зависимость FWHM от числа итераций [Laforest et al., JNM 2016]
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from pathlib import Path

import SimpleITK as sitk
from scipy.ndimage import gaussian_filter, laplace
from scipy.signal import fftconvolve
from tqdm import tqdm


@dataclass
class ReconstructionConfig:
    """Конфигурация реконструкции"""
    iterations: int = 4
    subsets: int = 21
    matrix_size: int = 192
    
    # Параметры фильтра
    filter_fwhm_mm: float = 5.0
    
    # Параметры шума (аппроксимация)
    noise_scale_iter: float = 0.05  # Увеличение шума с итерациями
    noise_scale_subsets: float = 0.02  # Влияние подмножеств
    
    # Сходимость (доля от полной сходимости)
    convergence_factor: float = 0.85


class ReconstructionSimulator:
    """
    Симулятор итеративной реконструкции ПЭТ
    
    Аппроксимация OSEM без sinogram:
    1. Начальное размытие (имитация ограниченного разрешения)
    2. Итеративное улучшение резкости
    3. Добавление шума, зависящего от параметров
    4. Пост-фильтрация
    """
    
    def __init__(self, config: Optional[ReconstructionConfig] = None):
        self.config = config or ReconstructionConfig()
        self.spacing: Tuple[float, float, float] = (4.0, 4.0, 3.0)
        
    def set_spacing(self, spacing: Tuple[float, float, float]):
        """Установка физического размера вокселей"""
        self.spacing = spacing
    
    def simulate_reconstruction(self, volume: np.ndarray,
                               iterations: Optional[int] = None,
                               subsets: Optional[int] = None,
                               matrix_size: Optional[int] = None,
                               seed: int = 42) -> np.ndarray:
        """
        Симуляция OSEM реконструкции
        
        Args:
            volume: Исходный объём (например, из DICOM)
            iterations: Число итераций OSEM
            subsets: Число подмножеств
            matrix_size: Размер матрицы реконструкции
            seed: Random seed для воспроизводимости
        
        Returns:
            Реконструированный объём
        """
        np.random.seed(seed)
        
        # Использование значений из конфигурации или параметров
        n_iter = iterations or self.config.iterations
        n_subsets = subsets or self.config.subsets
        mat_size = matrix_size or self.config.matrix_size
        
        # 1. Ресемплинг до целевой матрицы
        resampled = self._resample_volume(volume, mat_size)
        
        # 2. Начальное размытие (имитация системного PSF)
        sigma_init = self.config.filter_fwhm_mm / (2.355 * self.spacing[0])
        recon = gaussian_filter(resampled, sigma=sigma_init)
        
        # 3. Итеративное улучшение (аппроксимация OSEM)
        for i in range(n_iter):
            recon = self._osem_iteration(
                recon, resampled, 
                iteration=i,
                total_iterations=n_iter,
                subsets=n_subsets
            )
        
        # 4. Пост-фильтрация (зависит от подмножеств)
        # Больше подмножеств = больше шума = нужна большая фильтрация
        post_filter_sigma = self._calculate_post_filter_sigma(n_subsets)
        if post_filter_sigma > 0:
            recon = gaussian_filter(recon, sigma=post_filter_sigma)
        
        # 5. Добавление шума (калибровка по параметрам)
        recon = self._add_reconstruction_noise(
            recon, n_iter, n_subsets
        )
        
        return recon
    
    def _resample_volume(self, volume: np.ndarray, 
                        target_size: int) -> np.ndarray:
        """Ресемплинг к целевому размеру матрицы"""
        original_shape = volume.shape
        zoom_factors = [target_size / s for s in original_shape]
        
        from scipy.ndimage import zoom
        resampled = zoom(volume, zoom_factors, order=3)  # Cubic interpolation
        
        return resampled
    
    def _osem_iteration(self, current: np.ndarray, target: np.ndarray,
                       iteration: int, total_iterations: int,
                       subsets: int) -> np.ndarray:
        """
        Одна итерация аппроксимации OSEM
        
        Физика:
        - Ранние итерации: быстрое улучшение контраста
        - Поздние итерации: рост шума, меньшее улучшение
        - Подмножества влияют на скорость сходимости
        
        Эмпирическая модель [Laforest et al., JNM 2016]:
        RC(iter) = RC_max * (1 - exp(-k * iter * subsets / 21))
        """
        # Коэффициент сходимости
        k = 0.3  # Эмпирический коэффициент
        effective_iter = (iteration + 1) * subsets / 21.0
        convergence = 1.0 - np.exp(-k * effective_iter)
        
        # Целевое улучшение контраста
        alpha = 0.15 * convergence  # Шаг обновления
        
        # Лапласиан для улучшения краёв (имитация backproject-forwardproject)
        edge_enhancement = laplace(current)
        
        # Обновление с ограничением положительности
        update = current + alpha * (target - current) + 0.05 * edge_enhancement
        update = np.maximum(update, 0.0)
        
        return update
    
    def _calculate_post_filter_sigma(self, subsets: int) -> float:
        """
        Расчёт пост-фильтрации в зависимости от подмножеств
        
        Больше подмножеств = больше высокочастотного шума
        """
        base_sigma = 0.5
        subset_factor = max(0, (subsets - 16) / 21.0) * 0.3
        return base_sigma + subset_factor
    
    def _add_reconstruction_noise(self, volume: np.ndarray,
                                 iterations: int, subsets: int) -> np.ndarray:
        """
        Добавление шума реконструкции
        
        Модель:
        - Шум растёт с числом итераций (сходимость к максимуму правдоподобия)
        - Подмножества вносят дополнительные вариации
        """
        # Базовый уровень шума
        base_noise = 0.02
        
        # Увеличение с итерациями
        iter_noise = self.config.noise_scale_iter * np.sqrt(iterations)
        
        # Влияние подмножеств
        subset_noise = self.config.noise_scale_subsets * (subsets / 21.0)
        
        total_noise = base_noise + iter_noise + subset_noise
        
        # Мультипликативный Гауссовский шум
        noise = np.random.normal(0, total_noise, volume.shape)
        noisy = volume * (1.0 + noise)
        
        # Ограничение положительности
        noisy = np.maximum(noisy, 0.0)
        
        return noisy
    
    def create_parameter_grid(self) -> List[Dict]:
        """
        Создание сетки параметров для исследования
        
        Returns:
            Список конфигураций
        """
        iterations = [2, 3, 4, 6]
        subsets = [16, 21, 28]
        matrices = [128, 192, 256]
        
        grid = []
        for it in iterations:
            for sub in subsets:
                for mat in matrices:
                    grid.append({
                        'iterations': it,
                        'subsets': sub,
                        'matrix_size': mat,
                    })
        
        return grid
    
    def process_parameter_grid(self, volume: np.ndarray,
                              parameter_grid: Optional[List[Dict]] = None,
                              output_dir: Optional[Path] = None,
                              seed_base: int = 42) -> Dict:
        """
        Обработка всех комбинаций параметров
        
        Args:
            volume: Исходный объём
            parameter_grid: Сетка параметров (или создаётся автоматически)
            output_dir: Папка для сохранения результатов
            seed_base: Base seed для воспроизводимости
        
        Returns:
            Dictionary с результатами
        """
        if parameter_grid is None:
            parameter_grid = self.create_parameter_grid()
        
        results = {}
        
        for i, params in enumerate(tqdm(parameter_grid, desc="Processing parameter grid")):
            key = f"iter{params['iterations']}_sub{params['subsets']}_mat{params['matrix_size']}"
            
            reconstructed = self.simulate_reconstruction(
                volume,
                iterations=params['iterations'],
                subsets=params['subsets'],
                matrix_size=params['matrix_size'],
                seed=seed_base + i
            )
            
            results[key] = {
                'volume': reconstructed,
                'params': params,
            }
            
            # Сохранение если указана папка
            if output_dir:
                output_path = output_dir / f"{key}.nii.gz"
                self.save_volume(reconstructed, output_path, params)
        
        return results
    
    def save_volume(self, volume: np.ndarray, output_path: Path,
                   params: Dict):
        """Сохранение объёма в NIfTI формат"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        sitk_image = sitk.GetImageFromArray(volume)
        sitk_image.SetSpacing(self.spacing)
        
        # Метаданные
        sitk_image.SetMetaData('iterations', str(params['iterations']))
        sitk_image.SetMetaData('subsets', str(params['subsets']))
        sitk_image.SetMetaData('matrix_size', str(params['matrix_size']))
        
        sitk.WriteImage(sitk_image, str(output_path))


def approximate_osem_convergence(iterations: int, subsets: int,
                                true_activity: float = 1.0) -> Tuple[float, float]:
    """
    Approximation of OSEM convergence based on literature
    
    Model based on [Laforest et al., JNM 2016; NEMA NU-2-2018]:
    RC(iter) = RC_max * (1 - exp(-k * iter * subsets / 21))
    Noise(iter) = Noise_0 * sqrt(1 + beta * iter)
    
    Args:
        iterations: Number of OSEM iterations
        subsets: Number of subsets
        true_activity: True activity concentration
    
    Returns:
        Tuple of (recovery_coefficient, noise_cv)
    """
    # Параметры модели
    RC_max = 0.95  # Максимальный recovery coefficient
    k = 0.25  # Коэффициент сходимости
    
    # Effective iterations
    eff_iter = iterations * subsets / 21.0
    
    # Recovery coefficient
    rc = RC_max * (1.0 - np.exp(-k * eff_iter))
    
    # Noise (coefficient of variation)
    noise_0 = 0.05
    beta = 0.15
    noise_cv = noise_0 * np.sqrt(1.0 + beta * iterations)
    
    return rc, noise_cv


if __name__ == "__main__":
    # Тестирование симулятора
    print("Testing Reconstruction Simulator...")
    
    # Создание тестового объёма
    test_vol = np.zeros((64, 64, 32), dtype=np.float32)
    test_vol[20:44, 20:44, 10:22] = 4.0  # Hot sphere
    test_vol[40:56, 40:56, 20:28] = 2.0  # Another sphere
    test_vol += 1.0  # Background
    
    config = ReconstructionConfig(
        iterations=4,
        subsets=21,
        matrix_size=128,
    )
    
    simulator = ReconstructionSimulator(config)
    simulator.set_spacing((4.0, 4.0, 3.0))
    
    recon = simulator.simulate_reconstruction(test_vol)
    
    print(f"Original shape: {test_vol.shape}")
    print(f"Reconstructed shape: {recon.shape}")
    print(f"Original range: [{test_vol.min():.2f}, {test_vol.max():.2f}]")
    print(f"Reconstructed range: [{recon.min():.2f}, {recon.max():.2f}]")
    
    # Тест сетки параметров
    grid = simulator.create_parameter_grid()
    print(f"\nParameter grid size: {len(grid)} combinations")
    print(f"First combination: {grid[0]}")
