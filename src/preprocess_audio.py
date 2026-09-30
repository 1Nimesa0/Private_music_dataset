import librosa
import soundfile as sf
import os
import numpy as np

class AudioPreprocessor:
    def __init__(self, target_sr=22050, duration=10, segments_per_song=2):
        self.sr = target_sr
        self.duration = duration
        self.segments = segments_per_song
        self.samples_per_segment = self.sr * self.duration

    def process_and_segment(self, input_path: str, output_dir: str, song_id: str) -> list:
        """
        Đọc file, convert mono, resample, cắt ra n đoạn.
        KHÔNG DÙNG aggressive noise reduction để giữ Environment Target Domain.
        """
        # sr=None giữ nguyên sample rate ban đầu, librosa tự động convert sang Mono
        y, orig_sr = librosa.load(input_path, sr=None, mono=True) 
        
        # Resample về chuẩn 22050Hz
        if orig_sr != self.sr:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=self.sr)
            
        total_samples = len(y)
        min_required_samples = self.samples_per_segment * self.segments
        
        if total_samples < min_required_samples:
            raise ValueError(f"Audio quá ngắn. Yêu cầu {min_required_samples} samples, có {total_samples}")

        # Chọn điểm cắt (ví dụ đoạn 1 từ 20%, đoạn 2 từ 60% để âm thanh đa dạng)
        start_points = [
            int(total_samples * 0.2), 
            int(total_samples * 0.6)
        ]
        
        generated_files = []
        for i, start_idx in enumerate(start_points):
            end_idx = start_idx + self.samples_per_segment
            segment = y[start_idx:end_idx]
            
            # Khử đỉnh (Peak Normalization nhẹ) nếu clipping, không thay đổi noise floor
            max_amp = np.max(np.abs(segment))
            if max_amp > 1.0:
                segment = segment / max_amp
                
            out_filename = f"{song_id}_seg{i+1}.wav"
            out_path = os.path.join(output_dir, out_filename)
            sf.write(out_path, segment, self.sr, subtype='PCM_16')
            
            generated_files.append({
                'segment_id': f"{song_id}_seg{i+1}",
                'file_path': out_path,
                'start_time': start_idx / self.sr,
                'end_time': end_idx / self.sr
            })
            
        return generated_files