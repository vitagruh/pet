"""
DICOM Loader Module for PET Reconstruction Analysis
Модуль для загрузки и обработки DICOM-данных ПЭТ

Физические основы:
- SUV (Standardized Uptake Value) нормализует активность к введённой дозе и весу пациента
- Формула: SUV = (Activity_voxel [kBq/ml] × PatientWeight [kg]) / InjectedDose [kBq]
- Decay correction учитывает распад радионуклида (обычно F-18, T1/2 = 109.7 мин)
"""

import os
import glob
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk
from scipy.ndimage import zoom
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PETMetadata:
    """Метаданные ПЭТ-исследования для расчёта SUV"""
    patient_weight: float  # kg
    injected_dose: float  # Bq
    injection_time: str  # DICOM time format
    scan_time: str  # DICOM time format
    radionuclide: str  # e.g., "F-18"
    half_life_seconds: float  # seconds
    decay_correction: str  # "START", "ADMIN", etc.
    
    # Расчётные поля
    decay_factor: float = 1.0
    calibration_factor: float = 1.0


@dataclass
class PETVolume:
    """Контейнер для ПЭТ-данных с метаданными"""
    image_array: np.ndarray  # 3D array in SUV units
    original_array: np.ndarray  # 3D array in original units (Bq/ml or counts)
    spacing: Tuple[float, float, float]  # mm
    origin: Tuple[float, float, float]  # mm
    direction: np.ndarray  # 3x3 direction matrix
    metadata: PETMetadata
    dicom_info: Dict  # Raw DICOM info
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.image_array.shape


class DICOMLoader:
    """
    Загрузчик DICOM-серий ПЭТ с автоматическим извлечением метаданных
    
    Поддерживает:
    - Рекурсивный поиск DICOM-файлов
    - Сортировку по SliceLocation или InstanceNumber
    - Извлечение калибровочных коэффициентов
    - Конвертацию в SUV
    """
    
    # DICOM теги для извлечения
    REQUIRED_TAGS = {
        'PatientWeight': (0x0010, 0x1030),
        'RadionuclideTotalDose': (0x0054, 0x0081),
        'RadiopharmaceuticalStartTime': (0x0054, 0x0082),
        'ActualFrameDuration': (0x0054, 0x0090),
        'DecayCorrection': (0x0054, 0x1102),
        'Units': (0x0054, 0x1001),
        'RescaleSlope': (0x0028, 0x1053),
        'RescaleIntercept': (0x0028, 0x1052),
    }
    
    HALF_LIVES = {
        'F-18': 6586.0,  # seconds (109.7 min)
        'C-11': 1224.0,  # seconds (20.4 min)
        'O-15': 122.0,   # seconds (2.03 min)
        'N-13': 598.0,   # seconds (9.97 min)
        'Ga-68': 4086.0, # seconds (68.1 min)
    }
    
    def __init__(self, dicom_path: Union[str, Path], recursive: bool = True):
        """
        Args:
            dicom_path: Путь к папке с DICOM-файлами
            recursive: Рекурсивный поиск в подпапках
        """
        self.dicom_path = Path(dicom_path)
        self.recursive = recursive
        self.dicom_files: List[Path] = []
        self.metadata: Optional[PETMetadata] = None
        
    def find_dicom_files(self) -> List[Path]:
        """Рекурсивный поиск DICOM-файлов"""
        if self.recursive:
            pattern = "**/*.dcm"
        else:
            pattern = "*.dcm"
        
        files = list(self.dicom_path.glob(pattern))
        
        # Фильтрация не-DICOM файлов
        valid_files = []
        for f in tqdm(files, desc="Searching DICOM files"):
            try:
                ds = pydicom.dcmread(f, stop_before_pixels=True)
                # Проверяем Modality: 'PT' (стандарт DICOM) или 'PET'
                if hasattr(ds, 'Modality') and ds.Modality in ['PT', 'PET']:
                    valid_files.append(f)
            except Exception:
                continue
        
        logger.info(f"Found {len(valid_files)} PET DICOM files")
        self.dicom_files = sorted(valid_files)
        return self.dicom_files
    
    def sort_dicom_files(self, files: List[Path]) -> List[Path]:
        """
        Сортировка DICOM-файлов по пространственному положению
        
        Приоритет:
        1. SliceLocation (наиболее надёжный)
        2. ImagePositionPatient
        3. InstanceNumber
        """
        if not files:
            return []
        
        def get_sort_key(filepath: Path) -> float:
            try:
                ds = pydicom.dcmread(filepath, stop_before_pixels=True)
                
                # Приоритет 1: SliceLocation
                if hasattr(ds, 'SliceLocation'):
                    return float(ds.SliceLocation)
                
                # Приоритет 2: ImagePositionPatient (Z-coordinate)
                if hasattr(ds, 'ImagePositionPatient'):
                    return float(ds.ImagePositionPatient[2])
                
                # Приоритет 3: InstanceNumber
                if hasattr(ds, 'InstanceNumber'):
                    return float(ds.InstanceNumber)
                
                return 0.0
            except Exception as e:
                logger.warning(f"Could not read tags from {filepath}: {e}")
                return 0.0
        
        return sorted(files, key=get_sort_key)
    
    def extract_metadata(self, reference_file: Path) -> PETMetadata:
        """
        Извлечение метаданных из DICOM-файла для расчёта SUV
        
        Физика:
        - Активность распадается экспоненциально: A(t) = A0 * exp(-ln(2) * t / T1/2)
        - Decay correction корректирует измеренную активность к времени инъекции
        """
        ds = pydicom.dcmread(reference_file)
        
        # Извлечение обязательных полей
        patient_weight = getattr(ds, 'PatientWeight', 70.0)  # Default 70 kg
        if hasattr(patient_weight, 'value'):
            patient_weight = float(patient_weight.value)
        
        injected_dose = getattr(ds, 'RadionuclideTotalDose', 370e6)  # Default 370 MBq
        if hasattr(injected_dose, 'value'):
            injected_dose = float(injected_dose.value)
        
        injection_time = getattr(ds, 'RadiopharmaceuticalStartTime', '000000')
        scan_time = getattr(ds, 'AcquisitionTime', '000000')
        
        radionuclide = getattr(ds, 'Radionuclide', 'F-18')
        if hasattr(radionuclide, 'value'):
            radionuclide = str(radionuclide.value)
        
        # Определение периода полураспада
        half_life = self.HALF_LIVES.get(radionuclide, 6586.0)
        
        decay_correction = getattr(ds, 'DecayCorrection', 'START')
        if hasattr(decay_correction, 'value'):
            decay_correction = str(decay_correction.value)
        
        # Расчёт фактора распада
        decay_factor = self._calculate_decay_factor(
            injection_time, scan_time, half_life
        )
        
        # Калибровочный коэффициент (Rescale Slope)
        rescale_slope = getattr(ds, 'RescaleSlope', 1.0)
        rescale_intercept = getattr(ds, 'RescaleIntercept', 0.0)
        calibration_factor = float(rescale_slope)
        
        self.metadata = PETMetadata(
            patient_weight=patient_weight,
            injected_dose=injected_dose,
            injection_time=injection_time,
            scan_time=scan_time,
            radionuclide=radionuclide,
            half_life_seconds=half_life,
            decay_correction=decay_correction,
            decay_factor=decay_factor,
            calibration_factor=calibration_factor
        )
        
        logger.info(f"Extracted metadata: Weight={patient_weight}kg, "
                   f"Dose={injected_dose/1e6:.1f}MBq, Radionuclide={radionuclide}")
        
        return self.metadata
    
    def _calculate_decay_factor(self, t1: str, t2: str, half_life: float) -> float:
        """
        Расчёт фактора радиоактивного распада
        
        Args:
            t1: Время инъекции (HHMMSS format)
            t2: Время сканирования (HHMMSS format)
            half_life: Период полураспада в секундах
        
        Returns:
            Decay factor для коррекции активности
        """
        try:
            # Парсинг времени в секунды от полуночи
            def time_to_seconds(time_str: str) -> int:
                time_str = str(time_str).zfill(6)
                h = int(time_str[0:2])
                m = int(time_str[2:4])
                s = int(time_str[4:6])
                return h * 3600 + m * 60 + s
            
            t1_sec = time_to_seconds(t1)
            t2_sec = time_to_seconds(t2)
            
            delta_t = abs(t2_sec - t1_sec)
            
            # Экспоненциальный распад: exp(-ln(2) * t / T1/2)
            decay_factor = np.exp(-np.log(2) * delta_t / half_life)
            
            return decay_factor
        except Exception as e:
            logger.warning(f"Could not calculate decay factor: {e}")
            return 1.0
    
    def load_volume(self) -> PETVolume:
        """
        Загрузка 3D ПЭТ-объёма с конвертацией в SUV
        
        Returns:
            PETVolume объект с данными в SUV и оригинальных единицах
        """
        if not self.dicom_files:
            self.find_dicom_files()
        
        if not self.dicom_files:
            raise FileNotFoundError(f"No DICOM files found in {self.dicom_path}")
        
        # Сортировка файлов
        sorted_files = self.sort_dicom_files(self.dicom_files)
        
        # Извлечение метаданных из первого файла
        if self.metadata is None:
            self.extract_metadata(sorted_files[0])
        
        # Загрузка пикселей
        slices = []
        spacings = []
        
        for filepath in tqdm(sorted_files, desc="Loading slices"):
            ds = pydicom.dcmread(filepath)
            pixel_array = ds.pixel_array.astype(np.float32)
            
            # Применение Rescale Slope/Intercept
            if hasattr(ds, 'RescaleSlope'):
                pixel_array = pixel_array * float(ds.RescaleSlope)
            if hasattr(ds, 'RescaleIntercept'):
                pixel_array = pixel_array + float(ds.RescaleIntercept)
            
            slices.append(pixel_array)
            
            # Извлечение spacing
            if hasattr(ds, 'PixelSpacing'):
                spacing_xy = [float(x) for x in ds.PixelSpacing]
            else:
                spacing_xy = [4.0, 4.0]
            
            spacings.append(spacing_xy)
        
        # Стек в 3D массив
        volume = np.stack(slices, axis=0)
        
        # Определение Z-spacing
        if len(sorted_files) > 1:
            try:
                ds1 = pydicom.dcmread(sorted_files[0], stop_before_pixels=True)
                ds2 = pydicom.dcmread(sorted_files[1], stop_before_pixels=True)
                
                pos1 = getattr(ds1, 'ImagePositionPatient', [0, 0, 0])
                pos2 = getattr(ds2, 'ImagePositionPatient', [0, 0, 0])
                spacing_z = abs(float(pos2[2]) - float(pos1[2]))
            except:
                spacing_z = 3.0
        else:
            spacing_z = 3.0
        
        spacing_xy = spacings[0] if spacings else [4.0, 4.0]
        spacing = (spacing_xy[0], spacing_xy[1], spacing_z)
        
        # Сохранение оригинальной активности
        original_volume = volume.copy()
        
        # Конвертация в SUV
        suv_volume = self._convert_to_suv(volume)
        
        # DICOM info для сохранения
        dicom_info = {
            'PatientID': str(getattr(pydicom.dcmread(sorted_files[0], 
                          stop_before_pixels=True), 'PatientID', 'UNKNOWN')),
            'StudyDate': str(getattr(pydicom.dcmread(sorted_files[0], 
                          stop_before_pixels=True), 'StudyDate', '')),
            'SeriesDescription': str(getattr(pydicom.dcmread(sorted_files[0], 
                                  stop_before_pixels=True), 'SeriesDescription', '')),
        }
        
        pet_volume = PETVolume(
            image_array=suv_volume,
            original_array=original_volume,
            spacing=spacing,
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
            metadata=self.metadata,
            dicom_info=dicom_info
        )
        
        logger.info(f"Loaded volume: {pet_volume.shape}, spacing={spacing}")
        
        return pet_volume
    
    def _convert_to_suv(self, volume: np.ndarray) -> np.ndarray:
        """
        Конвертация активности в SUV
        
        Формула: SUV = (Activity [Bq/ml] × Weight [kg]) / Dose [Bq]
        
        Примечание: Если данные уже в SUV (проверяется по тегу Units), 
        возвращается исходный объём
        """
        if self.metadata is None:
            logger.warning("Metadata not available, returning original volume")
            return volume
        
        # Проверка единиц измерения
        units = getattr(pydicom.dcmread(self.dicom_files[0], 
                                        stop_before_pixels=True), 'Units', '')
        
        if 'SUV' in str(units).upper():
            logger.info("Data already in SUV units")
            return volume
        
        # Конвертация
        weight = self.metadata.patient_weight
        dose = self.metadata.injected_dose * self.metadata.decay_factor
        
        if dose <= 0:
            logger.warning("Invalid dose value, returning original volume")
            return volume
        
        suv_volume = (volume * weight) / dose
        
        logger.info(f"Converted to SUV: weight={weight}kg, dose={dose/1e6:.1f}MBq")
        
        return suv_volume
    
    def save_as_nifti(self, pet_volume: PETVolume, output_path: Union[str, Path]):
        """Сохранение в NIfTI формат для совместимости с другими инструментами"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Конвертация в SimpleITK
        sitk_image = sitk.GetImageFromArray(pet_volume.image_array)
        sitk_image.SetSpacing(pet_volume.spacing)
        sitk_image.SetOrigin(pet_volume.origin)
        
        # Сохранение метаданных
        sitk_image.SetMetaData('patient_weight', str(pet_volume.metadata.patient_weight))
        sitk_image.SetMetaData('injected_dose', str(pet_volume.metadata.injected_dose))
        sitk_image.SetMetaData('radionuclide', pet_volume.metadata.radionuclide)
        
        sitk.WriteImage(sitk_image, str(output_path))
        logger.info(f"Saved NIfTI: {output_path}")
    
    def load_from_nifti(self, nifti_path: Union[str, Path], 
                       metadata_dict: Optional[Dict] = None) -> PETVolume:
        """Загрузка из NIfTI формата"""
        nifti_path = Path(nifti_path)
        
        sitk_image = sitk.ReadImage(str(nifti_path))
        volume = sitk.GetArrayFromImage(sitk_image)
        
        spacing = sitk_image.GetSpacing()
        origin = sitk_image.GetOrigin()
        
        # Метаданные по умолчанию
        if metadata_dict is None:
            metadata_dict = {
                'patient_weight': 70.0,
                'injected_dose': 370e6,
                'radionuclide': 'F-18',
            }
        
        metadata = PETMetadata(
            patient_weight=metadata_dict.get('patient_weight', 70.0),
            injected_dose=metadata_dict.get('injected_dose', 370e6),
            injection_time='000000',
            scan_time='000000',
            radionuclide=metadata_dict.get('radionuclide', 'F-18'),
            half_life_seconds=6586.0,
            decay_correction='START',
        )
        
        return PETVolume(
            image_array=volume,
            original_array=volume.copy(),
            spacing=spacing,
            origin=origin,
            direction=np.eye(3),
            metadata=metadata,
            dicom_info={}
        )


# Convenience function
def load_pet_dicom(dicom_path: Union[str, Path], 
                   recursive: bool = True,
                   save_nifti: Optional[Union[str, Path]] = None) -> PETVolume:
    """
    Удобная функция для загрузки ПЭТ-данных
    
    Args:
        dicom_path: Путь к DICOM-папке
        recursive: Рекурсивный поиск
        save_nifti: Опционально сохранить как NIfTI
    
    Returns:
        PETVolume объект
    """
    loader = DICOMLoader(dicom_path, recursive=recursive)
    volume = loader.load_volume()
    
    if save_nifti:
        loader.save_as_nifti(volume, save_nifti)
    
    return volume


if __name__ == "__main__":
    # Тестирование модуля
    import sys
    
    if len(sys.argv) > 1:
        dicom_path = sys.argv[1]
    else:
        dicom_path = "./data/dicom_pet/"
    
    if os.path.exists(dicom_path):
        volume = load_pet_dicom(dicom_path)
        print(f"Loaded: {volume.shape}, SUV range: [{volume.image_array.min():.3f}, "
              f"{volume.image_array.max():.3f}]")
    else:
        print(f"Directory {dicom_path} does not exist. Create it and add DICOM files.")
