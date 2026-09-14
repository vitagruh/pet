#!/usr/bin/env python
"""
Скрипт для генерации синтетического DICOM фантома NEMA IEC Body
для тестирования пайплайна анализа ПЭТ реконструкции.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Добавляем src в path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def create_simple_dicom_phantom(output_dir: str = "data/dicom_pet"):
    """
    Создаёт простой синтетический DICOM фантом для тестирования.
    Использует только pydicom и numpy.
    """
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Параметры фантома
    matrix_size = 192
    n_slices = 80
    voxel_size = 3.0  # мм
    
    # Создаём 3D объём с "сферами"
    volume = np.zeros((matrix_size, matrix_size, n_slices), dtype=np.float32)
    
    # Центр каждого среза
    center_x, center_y = matrix_size // 2, matrix_size // 2
    
    # Добавляем несколько "сфер" разной интенсивности
    sphere_positions = [
        (center_x + 40, center_y, 5.0),    # Сфера 1: большая активность
        (center_x - 40, center_y, 4.0),    # Сфера 2
        (center_x, center_y + 40, 3.0),    # Сфера 3
        (center_x, center_y - 40, 2.5),    # Сфера 4
        (center_x + 25, center_y + 25, 2.0),  # Сфера 5
        (center_x - 25, center_y - 25, 1.5),  # Сфера 6
    ]
    
    z_centers = [20, 30, 40, 50, 60, 70]
    
    for i, ((sx, sy, intensity), sz) in enumerate(zip(sphere_positions, z_centers)):
        radius = [15, 13, 11, 9, 7, 5][i]  # Разные размеры сфер
        
        for z in range(n_slices):
            for x in range(matrix_size):
                for y in range(matrix_size):
                    dist = np.sqrt((x - sx)**2 + (y - sy)**2 + (z - sz)**2)
                    if dist < radius:
                        volume[x, y, z] += intensity * np.exp(-dist**2 / (2 * radius**2))
    
    # Добавляем фоновую активность
    background = 1.0
    noise = np.random.poisson(background * 100, volume.shape) / 100.0
    volume = volume + noise
    
    print(f"🔬 Создан синтетический фантом: {volume.shape}")
    print(f"📊 Размер вокселя: {voxel_size} мм")
    
    # Метаданные пациента
    patient_id = "NEMA_PHANTOM_001"
    patient_name = "NEMA^IEC^BODY^PHANTOM"
    patient_birth_date = "19000101"
    patient_sex = "O"
    
    study_id = "PET_RECON_STUDY"
    study_description = "NEMA IEC Body Phantom Simulation"
    study_date = "20240101"
    study_time = "120000"
    
    series_description = "PET Reconstruction"
    modality = "PT"  # PET
    
    # Калибровочные данные для SUV
    patient_weight = 70.0  # кг
    injected_dose = 370.0  # MBq (10 mCi)
    injection_time = "110000"
    scan_time = "120000"
    
    # Конвертируем в активность (Bq/ml)
    # Упрощённая калибровка: значение вокселя * scale = Bq/ml
    rescale_slope = 1000.0  # Bq/ml per unit
    rescale_intercept = 0.0
    
    dicom_files = []
    
    for slice_idx in range(n_slices):
        # Создаём DICOM dataset
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.128"  # PET Image Storage
        file_meta.MediaStorageSOPInstanceUID = f"1.2.3.4.5.6.7.8.9.{slice_idx:04d}"
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        
        ds = FileDataset(
            filename_or_obj=f"{output_path}/pet_{slice_idx:04d}.dcm",
            dataset={},
            preamble=b"\0" * 128,
            file_meta=file_meta,
            is_little_endian=True,
            is_implicit_VR=False
        )
        
        # Заполняем обязательные теги
        ds.PatientName = patient_name
        ds.PatientID = patient_id
        ds.PatientBirthDate = patient_birth_date
        ds.PatientSex = patient_sex
        
        ds.StudyInstanceUID = "1.2.3.4.5.6.7.8.9.1"
        ds.StudyID = study_id
        ds.StudyDescription = study_description
        ds.StudyDate = study_date
        ds.StudyTime = study_time
        
        ds.SeriesInstanceUID = "1.2.3.4.5.6.7.8.9.1.1"
        ds.SeriesNumber = 1
        ds.Modality = modality
        ds.SeriesDescription = series_description
        
        ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.128"
        ds.SOPInstanceUID = f"1.2.3.4.5.6.7.8.9.{slice_idx:04d}"
        ds.InstanceNumber = slice_idx + 1
        
        # Позиция и ориентация среза
        ds.ImagePositionPatient = [0.0, 0.0, slice_idx * voxel_size]
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ds.SliceLocation = slice_idx * voxel_size
        ds.SliceThickness = voxel_size
        
        # Размеры пикселя
        ds.PixelSpacing = [voxel_size, voxel_size]
        
        # Данные изображения
        ds.Rows = matrix_size
        ds.Columns = matrix_size
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1  # Signed
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        
        # Конвертируем в int16 для сохранения
        slice_data = volume[:, :, slice_idx].T  # Транспонируем для правильной ориентации
        slice_data_scaled = (slice_data * 100).astype(np.int16)
        ds.PixelData = slice_data_scaled.tobytes()
        
        # PET специфичные теги
        ds.Units = "BQML"  # Bq/ml
        ds.RescaleSlope = rescale_slope
        ds.RescaleIntercept = rescale_intercept
        
        # Калибровочная информация
        ds.DecayCorrection = "ADMIN"
        ds.PatientWeight = patient_weight
        ds.RadionuclideTotalDose = injected_dose * 1e6  # Bq
        ds.RadiopharmaceuticalStartTime = injection_time
        ds.CollectionTime = scan_time
        ds.HalfLife = 6588.0  # F-18 half-life in seconds
        
        # Сохраняем
        output_file = output_path / f"pet_{slice_idx:04d}.dcm"
        ds.save_as(str(output_file))
        dicom_files.append(str(output_file))
    
    print(f"✅ Создано {len(dicom_files)} DICOM файлов в {output_path}/")
    print(f"📁 Первый файл: {dicom_files[0]}")
    print(f"📁 Последний файл: {dicom_files[-1]}")
    
    return dicom_files


if __name__ == "__main__":
    print("=" * 60)
    print("Генерация синтетического DICOM фантома NEMA IEC Body")
    print("=" * 60)
    
    # Определяем рабочую директорию
    script_dir = Path(__file__).parent
    output_dir = script_dir / "data" / "dicom_pet"
    
    print(f"\n📂 Выходная директория: {output_dir}")
    
    # Генерируем фантом
    dicom_files = create_simple_dicom_phantom(str(output_dir))
    
    print("\n" + "=" * 60)
    print("✅ Готово! Фантом создан и готов к анализу.")
    print("=" * 60)
    print(f"\n📊 Количество файлов: {len(dicom_files)}")
    print(f"📁 Путь: {output_dir}")
    print("\nТеперь вы можете запустить Jupyter Notebook:")
    print("   jupyter notebook PET_Reconstruction_Analysis.ipynb")
