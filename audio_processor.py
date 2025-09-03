import librosa
import numpy as np
import soundfile as sf
import os

class AudioProcessor:
    """音频处理工具，用于提取音频特征"""
    def __init__(self, sample_rate=16000, n_mfcc=128, n_fft=2048, hop_length=512):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        
    def load_audio(self, audio_path):
        """加载音频文件并转换为指定采样率"""
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            return y
        except Exception as e:
            raise ValueError(f"无法加载音频文件 {audio_path}: {str(e)}")
    
    def extract_mfcc(self, audio):
        """提取MFCC特征"""
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        # 转置为 (时间步, 特征维度)
        return mfcc.T
    
    def extract_mel_spectrogram(self, audio):
        """提取梅尔频谱特征"""
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        # 转换为分贝并转置
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return mel_db.T
    
    def extract_chroma(self, audio):
        """提取色度特征"""
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        return chroma.T
    
    def extract_features(self, audio_path, feature_type="mfcc"):
        """提取指定类型的音频特征"""
        audio = self.load_audio(audio_path)
        
        if feature_type == "mfcc":
            return self.extract_mfcc(audio)
        elif feature_type == "mel":
            return self.extract_mel_spectrogram(audio)
        elif feature_type == "chroma":
            return self.extract_chroma(audio)
        else:
            raise ValueError(f"不支持的特征类型: {feature_type}")
    
    def resample_audio(self, audio_path, output_path, target_sr=16000):
        """重采样音频到目标采样率"""
        y, sr = librosa.load(audio_path, sr=None)
        y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sf.write(output_path, y_resampled, target_sr)
        return output_path
    
    def match_length(self, audio_features, target_length):
        """调整音频特征长度以匹配目标长度"""
        current_length = audio_features.shape[0]
        
        if current_length < target_length:
            # 填充
            pad_length = target_length - current_length
            return np.pad(audio_features, ((0, pad_length), (0, 0)), mode='edge')
        elif current_length > target_length:
            # 截断
            return audio_features[:target_length]
        return audio_features
    
    def save_features(self, features, output_path):
        """保存音频特征到文件"""
        np.save(output_path, features)
        
    def load_features(self, feature_path):
        """从文件加载音频特征"""
        return np.load(feature_path)
    