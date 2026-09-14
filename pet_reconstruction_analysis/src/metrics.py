"""
Metrics Calculation Module for PET Analysis
Модуль расчёта количественных метрик ПЭТ

Физические основы:
- SUV (Standardized Uptake Value): нормализованная активность
- SUVmax: максимальное значение в ROI
- SUVmean: среднее значение в ROI
- SUVpeak: среднее в 1ml сфере вокруг максимума (EANM рекомендация)
- Recovery Coefficient (RC): отношение измеренного SUV к истинному
- %CV (Coefficient of Variation): мера шума = 100 * std / mean
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from scipy.ndimage import center_of_mass, binary_erosion


@dataclass
class SphereMetrics:
    """Метрики для одной сферы"""
    sphere_id: int
    diameter_mm: float
    true_suv: float
    
    # Измеренные метрики
    suv_max: float
    suv_mean: float
    suv_peak: float
    volume_ml: float
    
    # Recovery coefficients
    rc_max: float
    rc_mean: float
    rc_peak: float
    
    # Дополнительно
    contrast_recovery: float


@dataclass
class BackgroundMetrics:
    """Метрики фона"""
    suv_mean: float
    suv_std: float
    noise_cv: float  # %CV
    uniformity: float  # NEMA uniformity


class MetricsCalculator:
    """
    Калькулятор количественных метрик ПЭТ
    
    Поддерживает:
    - Автоматическое обнаружение сфер thresholding'ом
    - Расчёт SUVmax, SUVmean, SUVpeak
    - Recovery Coefficients
    - Анализ шума (%CV)
    """
    
    def __init__(self, sphere_diameters: Optional[List[float]] = None,
                 true_sphere_suvs: Optional[List[float]] = None,
                 background_suv: float = 1.0):
        """
        Args:
            sphere_diameters: Диаметры сфер в мм (NEMA: [37, 28, 22, 17, 13, 10])
            true_sphere_suvs: Истинные SUV значения сфер
            background_suv: Ожидаемый SUV фона
        """
        self.sphere_diameters = sphere_diameters or [37, 28, 22, 17, 13, 10]
        self.true_sphere_suvs = true_sphere_suvs or [4.0] * len(self.sphere_diameters)
        self.background_suv = background_suv
        
        self.volume: Optional[np.ndarray] = None
        self.spacing: Tuple[float, float, float] = (4.0, 4.0, 3.0)
        self.sphere_masks: Dict[int, np.ndarray] = {}
        self.background_mask: Optional[np.ndarray] = None
        
    def set_volume(self, volume: np.ndarray, 
                   spacing: Tuple[float, float, float] = (4.0, 4.0, 3.0)):
        """Установка объёма для анализа"""
        self.volume = volume
        self.spacing = spacing
    
    def detect_spheres(self, threshold_factor: float = 1.5,
                      min_volume_mm3: float = 100.0) -> Dict[int, np.ndarray]:
        """
        Автоматическое обнаружение сфер через thresholding
        
        Args:
            threshold_factor: Порог относительно фона (e.g., 1.5 = 50% выше фона)
            min_volume_mm3: Минимальный объём сферы в мм³
        
        Returns:
            Dictionary {sphere_id: mask}
        """
        if self.volume is None:
            raise ValueError("Set volume first")
        
        # Базовый порог
        threshold = self.background_suv * threshold_factor
        
        # Бинарная маска "горячих" областей
        hot_mask = self.volume > threshold
        
        # Поиск связных компонент
        from scipy.ndimage import label, find_objects
        
        labeled, n_labels = label(hot_mask)
        
        # Фильтрация по размеру и сортировка по объёму
        sphere_candidates = []
        voxel_volume = np.prod(self.spacing)
        
        for i in range(1, n_labels + 1):
            component_mask = (labeled == i)
            volume_mm3 = np.sum(component_mask) * voxel_volume
            
            if volume_mm3 >= min_volume_mm3:
                # Вычисление центра масс
                center = center_of_mass(component_mask)
                sphere_candidates.append({
                    'id': i,
                    'mask': component_mask,
                    'volume_mm3': volume_mm3,
                    'center': center,
                })
        
        # Сортировка по объёму (большие сферы первыми)
        sphere_candidates.sort(key=lambda x: x['volume_mm3'], reverse=True)
        
        # Назначение ID сферам по соответствию диаметрам
        self.sphere_masks = {}
        for i, candidate in enumerate(sphere_candidates):
            if i < len(self.sphere_diameters):
                self.sphere_masks[i] = candidate['mask']
        
        return self.sphere_masks
    
    def load_sphere_masks(self, masks: Dict[int, np.ndarray]):
        """Загрузка готовых масок сфер (например, из генератора фантома)"""
        self.sphere_masks = masks
    
    def create_background_roi(self, margin_voxels: int = 3) -> np.ndarray:
        """
        Создание ROI фона (вне сфер)
        
        Args:
            margin_voxels: Отступ от сфер в вокселях
        """
        if self.volume is None:
            raise ValueError("Set volume first")
        
        # Объединение всех сфер
        if self.sphere_masks:
            all_spheres = np.zeros_like(self.volume, dtype=bool)
            for mask in self.sphere_masks.values():
                all_spheres |= mask
        else:
            # Если маски не заданы, используем thresholding
            all_spheres = self.volume > (self.background_suv * 1.5)
        
        # Маска тела (ненулевые значения)
        body_mask = self.volume > 0
        
        # Фон = тело минус сферы
        background = body_mask & (~all_spheres)
        
        # Применение отступа
        if margin_voxels > 0 and background.any():
            background = binary_erosion(background, iterations=margin_voxels)
        
        self.background_mask = background
        return background
    
    def calculate_suv_peak(self, mask: np.ndarray, 
                          peak_volume_ml: float = 1.0) -> float:
        """
        Расчёт SUVpeak - среднее в 1ml сфере вокруг максимума
        
        EANM рекомендация: сфера 1ml вокруг вокселя с максимумом
        
        Args:
            mask: ROI маска
            peak_volume_ml: Объём для усреднения (мл)
        
        Returns:
            SUVpeak значение
        """
        if self.volume is None:
            return 0.0
        
        # Нахождение вокселя с максимумом в ROI
        roi_values = self.volume[mask]
        if len(roi_values) == 0:
            return 0.0
        
        # Получение координат максимума
        roi_indices = np.where(mask)
        max_local_idx = np.argmax(roi_values)
        
        # 3D координаты максимума
        max_pos = (roi_indices[0][max_local_idx], 
                   roi_indices[1][max_local_idx], 
                   roi_indices[2][max_local_idx])
        
        # Расчёт радиуса сферы 1ml
        voxel_volume_ml = np.prod(self.spacing) / 1000.0  # мм³ -> мл
        sphere_radius_vox = int(np.ceil((peak_volume_ml / voxel_volume_ml) ** (1/3)))
        sphere_radius_vox = max(1, min(sphere_radius_vox, 5))  # Ограничение радиуса
        
        # Создание куба вокруг максимума (упрощённая аппроксимация сферы)
        z0, y0, x0 = max_pos
        r = sphere_radius_vox
        
        z_min, z_max = max(0, z0 - r), min(self.volume.shape[0], z0 + r + 1)
        y_min, y_max = max(0, y0 - r), min(self.volume.shape[1], y0 + r + 1)
        x_min, x_max = max(0, x0 - r), min(self.volume.shape[2], x0 + r + 1)
        
        peak_region = self.volume[z_min:z_max, y_min:y_max, x_min:x_max]
        
        if peak_region.size == 0:
            return float(self.volume[max_pos])
        
        suv_peak = float(np.mean(peak_region))
        
        return suv_peak
    
    def calculate_sphere_metrics(self, sphere_id: int) -> Optional[SphereMetrics]:
        """
        Расчёт всех метрик для одной сферы
        
        Args:
            sphere_id: ID сферы
        
        Returns:
            SphereMetrics объект или None если маска не найдена
        """
        if sphere_id not in self.sphere_masks:
            return None
        
        mask = self.sphere_masks[sphere_id]
        true_suv = self.true_sphere_suvs[sphere_id] if sphere_id < len(self.true_sphere_suvs) else 4.0
        diameter = self.sphere_diameters[sphere_id] if sphere_id < len(self.sphere_diameters) else 20.0
        
        # Извлечение значений
        values = self.volume[mask]
        
        if len(values) == 0:
            return None
        
        # Базовые метрики
        suv_max = float(np.max(values))
        suv_mean = float(np.mean(values))
        suv_peak = self.calculate_suv_peak(mask)
        
        # Объём сферы в мл
        voxel_volume_ml = np.prod(self.spacing) / 1000.0
        volume_ml = float(np.sum(mask)) * voxel_volume_ml
        
        # Recovery coefficients
        rc_max = suv_max / true_suv if true_suv > 0 else 0.0
        rc_mean = suv_mean / true_suv if true_suv > 0 else 0.0
        rc_peak = suv_peak / true_suv if true_suv > 0 else 0.0
        
        # Contrast recovery
        contrast_recovery = (suv_mean - self.background_suv) / (true_suv - self.background_suv) \
                           if (true_suv - self.background_suv) > 0 else 0.0
        
        return SphereMetrics(
            sphere_id=sphere_id,
            diameter_mm=diameter,
            true_suv=true_suv,
            suv_max=suv_max,
            suv_mean=suv_mean,
            suv_peak=suv_peak,
            volume_ml=volume_ml,
            rc_max=rc_max,
            rc_mean=rc_mean,
            rc_peak=rc_peak,
            contrast_recovery=contrast_recovery
        )
    
    def calculate_background_metrics(self) -> Optional[BackgroundMetrics]:
        """
        Расчёт метрик фона (шум, однородность)
        
        Returns:
            BackgroundMetrics объект
        """
        if self.background_mask is None:
            self.create_background_roi()
        
        if self.background_mask is None or not self.background_mask.any():
            return None
        
        values = self.volume[self.background_mask]
        
        if len(values) == 0:
            return None
        
        suv_mean = float(np.mean(values))
        suv_std = float(np.std(values))
        
        # Коэффициент вариации (%)
        noise_cv = 100.0 * suv_std / suv_mean if suv_mean > 0 else 0.0
        
        # NEMA uniformity (variability)
        uniformity = 100.0 * (np.max(values) - np.min(values)) / (np.max(values) + np.min(values)) \
                    if (np.max(values) + np.min(values)) > 0 else 0.0
        
        return BackgroundMetrics(
            suv_mean=suv_mean,
            suv_std=suv_std,
            noise_cv=noise_cv,
            uniformity=uniformity
        )
    
    def analyze_all_spheres(self) -> pd.DataFrame:
        """
        Анализ всех сфер и создание DataFrame
        
        Returns:
            DataFrame со всеми метриками
        """
        metrics_list = []
        
        for sphere_id in range(len(self.sphere_diameters)):
            metrics = self.calculate_sphere_metrics(sphere_id)
            if metrics:
                metrics_list.append({
                    'sphere_id': metrics.sphere_id,
                    'diameter_mm': metrics.diameter_mm,
                    'true_suv': metrics.true_suv,
                    'suv_max': metrics.suv_max,
                    'suv_mean': metrics.suv_mean,
                    'suv_peak': metrics.suv_peak,
                    'volume_ml': metrics.volume_ml,
                    'rc_max': metrics.rc_max,
                    'rc_mean': metrics.rc_mean,
                    'rc_peak': metrics.rc_peak,
                    'contrast_recovery': metrics.contrast_recovery,
                })
        
        df = pd.DataFrame(metrics_list)
        return df
    
    def full_analysis(self, params: Dict = None) -> Dict:
        """
        Полный анализ: сферы + фон
        
        Args:
            params: Дополнительные параметры (итерации, подмножества, матрица)
        
        Returns:
            Dictionary с результатами
        """
        # Фон
        bg_metrics = self.calculate_background_metrics()
        
        # Сферы
        spheres_df = self.analyze_all_spheres()
        
        # Добавление параметров реконструкции
        if params:
            spheres_df = spheres_df.assign(**params)
            if bg_metrics:
                bg_dict = {
                    'noise_cv': bg_metrics.noise_cv,
                    'background_suv_mean': bg_metrics.suv_mean,
                    'background_suv_std': bg_metrics.suv_std,
                    'uniformity': bg_metrics.uniformity,
                }
                bg_dict.update(params)
            else:
                bg_dict = None
        else:
            bg_dict = {
                'noise_cv': bg_metrics.noise_cv if bg_metrics else None,
                'background_suv_mean': bg_metrics.suv_mean if bg_metrics else None,
            } if bg_metrics else None
        
        return {
            'spheres': spheres_df,
            'background': bg_dict,
        }


def batch_analyze_reconstructions(reconstructions: Dict[str, Dict],
                                  sphere_diameters: List[float] = None,
                                  true_suvs: List[float] = None,
                                  background_suv: float = 1.0) -> pd.DataFrame:
    """
    Массовый анализ всех реконструкций
    
    Args:
        reconstructions: Dict {key: {'volume': array, 'params': dict}}
        sphere_diameters: Диаметры сфер
        true_suvs: Истинные SUV сфер
        background_suv: SUV фона
    
    Returns:
        Сводный DataFrame
    """
    all_results = []
    
    for key, data in reconstructions.items():
        volume = data['volume']
        params = data.get('params', {})
        
        calculator = MetricsCalculator(
            sphere_diameters=sphere_diameters,
            true_sphere_suvs=true_suvs,
            background_suv=background_suv
        )
        
        calculator.set_volume(volume)
        
        # Попытка загрузить маски сфер если доступны
        if 'sphere_masks' in data:
            calculator.load_sphere_masks(data['sphere_masks'])
        else:
            calculator.detect_spheres()
        
        results = calculator.full_analysis(params)
        
        # Добавление ключа реконструкции
        results['spheres']['reconstruction_key'] = key
        all_results.append(results['spheres'])
    
    combined_df = pd.concat(all_results, ignore_index=True)
    return combined_df


if __name__ == "__main__":
    # Тестирование
    print("Testing Metrics Calculator...")
    
    # Создание тестового объёма
    test_vol = np.ones((64, 64, 32), dtype=np.float32)
    test_vol[20:44, 20:44, 10:22] = 4.0  # Sphere 1
    test_vol[40:56, 40:56, 20:28] = 3.0  # Sphere 2
    
    calc = MetricsCalculator(
        sphere_diameters=[30, 20],
        true_sphere_suvs=[4.0, 3.0],
        background_suv=1.0
    )
    
    calc.set_volume(test_vol, spacing=(4.0, 4.0, 3.0))
    calc.detect_spheres(threshold_factor=1.5)
    
    results = calc.full_analysis({'test': 'param'})
    
    print("\nSphere metrics:")
    print(results['spheres'])
    print(f"\nBackground: {results['background']}")
