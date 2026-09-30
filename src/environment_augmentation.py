import os
import logging
import numpy as np
import librosa
import soundfile as sf
import pandas as pd
from typing import List

logger = logging.getLogger(__name__)

class EnvironmentAugmenter:
    def __init__(self, metadata_path: str):
        self.metadata_path = metadata_path
        self.env_types = ['clean', 'room_reverb', 'background_noise', 'low_quality_mic']
        
    def _add_white_noise(self, y: np.ndarray, snr_db: float = 15.0) -> np.ndarray:
        signal_power = np.mean(y ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(y))
        return y + noise
        
    def _apply_simple_reverb(self, y: np.ndarray) -> np.ndarray:
        # Giả lập reverb đơn giản bằng cách cộng dồn tín hiệu delay
        delay_samples = int(22050 * 0.05) # 50ms delay
        decay = 0.5
        y_reverb = np.copy(y)
        y_reverb[delay_samples:] += y[:-delay_samples] * decay
        return y_reverb / np.max(np.abs(y_reverb)) # Normalize

    def _apply_low_pass_filter(self, y: np.ndarray) -> np.ndarray:
        # Giả lập low-quality mic bằng cách cắt tần số cao
        # Bằng thuật toán rolling mean đơn giản hoặc scipy.signal
        window_size = 5
        y_filtered = np.convolve(y, np.ones(window_size)/window_size, mode='same')
        return y_filtered

    def process_dataset(self, target_non_clean_ratio: float = 0.5):
        """
        Đảm bảo dataset đạt tỉ lệ đa dạng môi trường.
        Tuyệt đối không can thiệp vào các sample thuộc tập 'test' hoặc 'private_test'.
        """
        df = pd.read_csv(self.metadata_path)
        
        # Chỉ lấy các sample thuộc tập train/val để xét augmentation[cite: 1]
        augmentable_indices = df[~df['split'].isin(['test', 'private_test'])].index.tolist()
        
        current_non_clean = len(df[df['environment_type'] != 'clean'])
        total_samples = len(df)
        
        samples_needed = int(total_samples * target_non_clean_ratio) - current_non_clean
        
        if samples_needed <= 0:
            logger.info("Dataset đã đạt đủ tỉ lệ non-clean hợp lệ.")
            return

        logger.info(f"Cần áp dụng augmentation cho {samples_needed} samples thuộc tập train/val.")
        
        # Chọn ngẫu nhiên các sample clean trong tập augmentable để xử lý
        clean_augmentable_indices = df.loc[augmentable_indices]
        clean_augmentable_indices = clean_augmentable_indices[clean_augmentable_indices['environment_type'] == 'clean'].index.tolist()
        
        np.random.shuffle(clean_augmentable_indices)
        selected_indices = clean_augmentable_indices[:samples_needed]
        
        for idx in selected_indices:
            row = df.loc[idx]
            file_path = row['file_path']
            
            try:
                y, sr = librosa.load(file_path, sr=None)
                
                # Chọn ngẫu nhiên 1 trong 3 hiệu ứng môi trường
                env_choice = np.random.choice(['room_reverb', 'background_noise', 'low_quality_mic'])
                
                if env_choice == 'room_reverb':
                    y_aug = self._apply_simple_reverb(y)
                elif env_choice == 'background_noise':
                    y_aug = self._add_white_noise(y, snr_db=np.random.uniform(10, 20))
                else:
                    y_aug = self._apply_low_pass_filter(y)
                    
                # Ghi đè file hoặc lưu thành file mới (tùy chiến lược đường dẫn)
                sf.write(file_path, y_aug, sr)
                
                # Cập nhật metadata
                df.at[idx, 'environment_type'] = env_choice
                
            except Exception as e:
                logger.error(f"Lỗi augmentation file {file_path}: {e}")
                
        # Lưu lại metadata đã cập nhật
        df.to_csv(self.metadata_path, index=False)
        logger.info("Hoàn tất quy trình Environment Augmentation.")