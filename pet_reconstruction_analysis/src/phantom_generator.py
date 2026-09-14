"""
Phantom Generator Module for PET Reconstruction Analysis
Генератор синтетических ПЭТ-фантомов NEMA IEC Body

Физические основы:
- Фантом NEMA IEC содержит 6 сфер разного диаметра (10, 13, 17, 22, 28, 37 мм)
- Отношение активности сфер к фону: обычно 4:1 или 8:1
- Добавляется реалистичный шум (Пуассоновский + Гауссовский)
- Учитывается разрешение системы (PSF blur)
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from pathlib import Path

import SimpleITK as sitk
from scipy.ndimage import gaussian_filter
from tqdm import tqdm


@dataclass
class SphereDefinition:
    """Определение сферы фантома"""
    diameter_mm: float
    center_x: float  # mm от центра
    center_y: float
    center_z: float
    activity_ratio: float  # Отношение к фону (e.g., 4.0 = 4:1)
    true_suv: float  # Истинное SUV значение


@dataclass
class PhantomConfig:
    """Конфигурация фантома"""
    # Размеры матрицы
    matrix_x: int = 192
    matrix_y: int = 192
    matrix_z: int = 100
    
    # Разрешение вокселей (мм)
    spacing_x: float = 4.0
    spacing_y: float = 4.0
    spacing_z: float = 3.0
    
    # Активность
    background_suv: float = 1.0
    sphere_activity_ratio: float = 4.0
    
    # Шум
    noise_level: float = 0.1  # Коэффициент вариации
    poisson_scale: float = 1000.0  # Масштаб для Пуассоновского шума
    
    # Разрешение системы (PSF)
    psf_fwhm_mm: float = 5.0  # Full Width at Half Maximum
    
    # Размеры фантома
    phantom_width_mm: float = 235  # Типичный размер тела
    phantom_height_mm: float = 185
    phantom_length_mm: float = 300
    
    # Определение сфер NEMA IEC
    spheres: List[SphereDefinition] = None
    
    def __post_init__(self):
        if self.spheres is None:
            self.spheres = self._default_spheres()
    
    def _default_spheres(self) -> List[SphereDefinition]:
        """
        Сферы NEMA IEC Body Phantom
        Диаметры: 10, 13, 17, 22, 28, 37 мм
        """
        # Типичные позиции в левой части фантома
        sphere_positions = [
            (37, -60, 0, 0),   # Самая большая
            (28, -30, 30, 0),
            (22, 30, 30, 0),
            (17, -30, -30, 0),
            (13, 30, -30, 0),
            (10, 0, 0, 0),     # Самая маленькая в центре
        ]
        
        spheres = []
        for i, (diam, cx, cy, cz) in enumerate(sphere_positions):
            spheres.append(SphereDefinition(
                diameter_mm=diam,
                center_x=cx,
                center_y=cy,
                center_z=cz,
                activity_ratio=self.sphere_activity_ratio,
                true_suv=self.background_suv * self.sphere_activity_ratio
            ))
        
        return spheres


class PhantomGenerator:
    """
    Генератор синтетических ПЭТ-фантомов
    
    Создаёт реалистичные 3D данные с:
    - Эллипсоидальным телом
    - Шестью сферами NEMA IEC
    - Реалистичным шумом
    - Размытием PSF
    """
    
    def __init__(self, config: Optional[PhantomConfig] = None):
        """
        Args:
            config: Конфигурация фантома (по умолчанию NEMA IEC)
        """
        self.config = config or PhantomConfig()
        self.volume: Optional[np.ndarray] = None
        self.mask: Optional[np.ndarray] = None
        self.sphere_masks: Dict[int, np.ndarray] = {}
        
    def generate_phantom(self, seed: int = 42) -> np.ndarray:
        """
        Генерация полного фантома
        
        Args:
            seed: Seed для воспроизводимости
        
        Returns:
            3D numpy array в единицах SUV
        """
        np.random.seed(seed)
        
        # Создание сетки координат
        x = np.arange(self.config.matrix_x) * self.config.spacing_x
        y = np.arange(self.config.matrix_y) * self.config.spacing_y
        z = np.arange(self.config.matrix_z) * self.config.spacing_z
        
        x = x - x.mean()  # Центрирование
        y = y - y.mean()
        z = z - z.mean()
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # 1. Создание эллипсоидального тела
        body_mask = self._create_ellipsoid(
            X, Y, Z,
            self.config.phantom_width_mm / 2,
            self.config.phantom_height_mm / 2,
            self.config.phantom_length_mm / 2
        )
        
        # Фон
        volume = np.zeros_like(X, dtype=np.float32)
        volume[body_mask > 0] = self.config.background_suv
        
        self.mask = body_mask.astype(np.uint8)
        
        # 2. Добавление сфер
        for i, sphere in enumerate(self.config.spheres):
            sphere_mask = self._create_sphere(
                X, Y, Z,
                sphere.center_x,
                sphere.center_y,
                sphere.center_z,
                sphere.diameter_mm / 2
            )
            
            # Сохранение маски сферы для последующего анализа
            self.sphere_masks[i] = sphere_mask.astype(np.uint8)
            
            # Установка активности сферы
            volume[sphere_mask > 0] = sphere.true_suv
        
        # 3. Применение PSF (разрешение системы)
        sigma = self.config.psf_fwhm_mm / (2.355 * self.config.spacing_x)
        volume = gaussian_filter(volume, sigma=sigma)
        
        # 4. Добавление шума
        volume = self._add_noise(volume, body_mask)
        
        # Обнуление вне тела
        volume[body_mask == 0] = 0.0
        
        self.volume = volume
        
        return volume
    
    def _create_ellipsoid(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                         a: float, b: float, c: float) -> np.ndarray:
        """Создание эллипсоида"""
        return ((X/a)**2 + (Y/b)**2 + (Z/c)**2 <= 1).astype(np.float32)
    
    def _create_sphere(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                      cx: float, cy: float, cz: float, r: float) -> np.ndarray:
        """Создание сферы"""
        return (((X-cx)/r)**2 + ((Y-cy)/r)**2 + ((Z-cz)/r)**2 <= 1).astype(np.float32)
    
    def _add_noise(self, volume: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Добавление реалистичного шума
        
        Модель шума:
        1. Пуассоновский шум (статистика счёта)
        2. Гауссовский шум (электронный шум, реконструкция)
        
        См. [1] Laforest et al., JNM 2016
        """
        noisy_volume = volume.copy()
        
        # Масштабирование для Пуассоновской статистики
        scaled = volume * self.config.poisson_scale
        
        # Пуассоновский шум
        poisson_noise = np.random.poisson(scaled) / self.config.poisson_scale
        
        # Гауссовский шум
        gaussian_noise = np.random.normal(0, self.config.noise_level, volume.shape)
        
        # Комбинирование
        noisy_volume = poisson_noise + gaussian_noise * volume
        
        # Сохранение фона в области тела
        noisy_volume[mask == 0] = 0.0
        
        # Отрицательные значения -> 0
        noisy_volume = np.maximum(noisy_volume, 0.0)
        
        return noisy_volume
    
    def save_as_dicom_series(self, output_dir: Path, 
                            patient_id: str = "PHANTOM",
                            study_date: str = "20240101"):
        """
        Сохранение как DICOM-серия
        
        Примечание: Для полноценных DICOM требуется дополнительная информация
        о калибровке. Здесь создаются упрощённые файлы.
        """
        if self.volume is None:
            raise ValueError("Generate phantom first")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        import pydicom
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian
        
        # Получение размеров
        nz, ny, nx = self.volume.shape
        spacing = (self.config.spacing_y, self.config.spacing_x, self.config.spacing_z)
        
        # Создание DICOM для каждого среза
        for z_idx in tqdm(range(nz), desc="Saving DICOM slices"):
            slice_data = self.volume[z_idx, :, :]
            
            # Создание DICOM dataset
            file_meta = FileMetaDataset()
            file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.128'  # PET Image
            file_meta.MediaStorageSOPInstanceUID = f"1.2.3.4.5.{z_idx:04d}"
            file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
            
            ds = FileDataset(
                output_dir / f"PET_{z_idx:04d}.dcm",
                {},
                file_meta=file_meta,
                preamble=b"\0" * 128
            )
            
            # Обязательные теги
            ds.Modality = "PT"
            ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.128'
            ds.SOPInstanceUID = f"1.2.3.4.5.{z_idx:04d}"
            ds.PatientID = patient_id
            ds.StudyDate = study_date
            ds.SeriesDescription = "NEMA Phantom Simulation"
            ds.InstanceNumber = z_idx + 1
            ds.SliceLocation = (z_idx - nz/2) * self.config.spacing_z
            
            # Информация об изображении
            ds.Rows = ny
            ds.Columns = nx
            ds.PixelSpacing = [spacing[0], spacing[1]]
            ds.BitsAllocated = 32
            ds.BitsStored = 32
            ds.HighBit = 31
            ds.PixelRepresentation = 1
            ds.SamplePerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            
            # Калибровочные коэффициенты
            ds.RescaleSlope = 1.0
            ds.RescaleIntercept = 0.0
            ds.Units = "SUV"
            
            # Метаданные ПЭТ
            ds.Radionuclide = "F-18"
            ds.DecayCorrection = "START"
            ds.PatientWeight = 70.0  # kg
            ds.RadionuclideTotalDose = 370e6  # Bq (370 MBq)
            
            # Pixel data
            ds.PixelData = slice_data.astype(np.float32).tobytes()
            
            # Сохранение
            ds.save_as(output_dir / f"PET_{z_idx:04d}.dcm")
        
        print(f"Saved {nz} DICOM files to {output_dir}")
    
    def save_as_nifti(self, output_path: Path):
        """Сохранение в NIfTI формат"""
        if self.volume is None:
            raise ValueError("Generate phantom first")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        sitk_image = sitk.GetImageFromArray(self.volume)
        sitk_image.SetSpacing((
            self.config.spacing_x,
            self.config.spacing_y,
            self.config.spacing_z
        ))
        
        sitk.WriteImage(sitk_image, str(output_path))
        print(f"Saved NIfTI: {output_path}")
    
    def get_sphere_rois(self) -> Dict[int, np.ndarray]:
        """Получение масок сфер для ROI анализа"""
        return self.sphere_masks.copy()
    
    def get_background_roi(self, margin_mm: float = 20) -> np.ndarray:
        """
        Получение ROI фона (вне сфер)
        
        Args:
            margin_mm: Отступ от границ фантома
        """
        if self.mask is None:
            raise ValueError("Generate phantom first")
        
        # Маска тела
        body_mask = self.mask > 0
        
        # Исключение сфер
        sphere_mask = np.zeros_like(body_mask, dtype=bool)
        for sphere_mask_i in self.sphere_masks.values():
            sphere_mask |= (sphere_mask_i > 0)
        
        # Background ROI
        background_roi = body_mask & (~sphere_mask)
        
        # Применение отступа от границ
        if margin_mm > 0:
            from scipy.ndimage import binary_erosion
            iterations = int(margin_mm / min(
                self.config.spacing_x,
                self.config.spacing_y,
                self.config.spacing_z
            ))
            background_roi = binary_erosion(background_roi, iterations=iterations)
        
        return background_roi.astype(np.uint8)


def generate_nema_phantom(output_dir: Optional[Path] = None,
                         matrix_size: int = 192,
                         sphere_ratio: float = 4.0,
                         noise_level: float = 0.1,
                         seed: int = 42) -> Tuple[np.ndarray, Dict]:
    """
    Convenience function для генерации фантома NEMA IEC
    
    Args:
        output_dir: Папка для сохранения DICOM/NIfTI
        matrix_size: Размер матрицы
        sphere_ratio: Отношение активности сфер к фону
        noise_level: Уровень шума
        seed: Random seed
    
    Returns:
        volume: 3D numpy array
        metadata: Dict с информацией о сферах
    """
    config = PhantomConfig(
        matrix_x=matrix_size,
        matrix_y=matrix_size,
        matrix_z=100,
        sphere_activity_ratio=sphere_ratio,
        noise_level=noise_level,
    )
    
    generator = PhantomGenerator(config)
    volume = generator.generate_phantom(seed=seed)
    
    if output_dir:
        output_dir = Path(output_dir)
        generator.save_as_nifti(output_dir / "phantom.nii.gz")
        generator.save_as_dicom_series(output_dir / "dicom")
    
    metadata = {
        'spheres': [
            {
                'diameter_mm': s.diameter_mm,
                'true_suv': s.true_suv,
                'activity_ratio': s.activity_ratio,
            }
            for s in config.spheres
        ],
        'background_suv': config.background_suv,
        'noise_level': config.noise_level,
        'psf_fwhm_mm': config.psf_fwhm_mm,
    }
    
    return volume, metadata


if __name__ == "__main__":
    # Тестирование генератора
    import sys
    
    output_dir = Path("./data/dicom_pet")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating NEMA IEC Body Phantom...")
    volume, metadata = generate_nema_phantom(
        output_dir=output_dir,
        matrix_size=192,
        sphere_ratio=4.0,
        noise_level=0.1,
        seed=42
    )
    
    print(f"Generated volume: {volume.shape}")
    print(f"SUV range: [{volume.min():.3f}, {volume.max():.3f}]")
    print(f"Sphere metadata: {metadata['spheres']}")
