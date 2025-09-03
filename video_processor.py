import cv2
import numpy as np
import dlib
import os
from PIL import Image
import torch
import torchvision.transforms as transforms

class VideoProcessor:
    """视频处理工具，用于提取视频帧和面部特征"""
    def __init__(self, image_size=128, face_detector_path=None):
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        
        # 初始化面部检测器和特征点预测器
        self.face_detector = dlib.get_frontal_face_detector()
        
        if face_detector_path is None:
            # 默认使用68点特征预测器
            self.landmark_predictor = dlib.shape_predictor(
                "pretrained/shape_predictor_68_face_landmarks.dat"
            )
        else:
            self.landmark_predictor = dlib.shape_predictor(face_detector_path)
        
        # 面部特征点索引（只关注嘴巴周围）
        self.mouth_landmarks_indices = list(range(48, 68))  # 嘴巴周围的20个点

    def load_video(self, video_path, max_frames=None):
        """加载视频并提取帧"""
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 转换为RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                
                if max_frames is not None and len(frames) >= max_frames:
                    break
            
            cap.release()
            return np.array(frames), fps
        
        except Exception as e:
            raise ValueError(f"无法加载视频文件 {video_path}: {str(e)}")

    def detect_face(self, frame):
        """检测面部并返回边界框"""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = self.face_detector(gray)
        
        if len(faces) == 0:
            return None  # 未检测到人脸
        
        # 返回最大的人脸
        return max(faces, key=lambda rect: rect.width() * rect.height())

    def extract_landmarks(self, frame, face_rect):
        """提取面部特征点"""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        shape = self.landmark_predictor(gray, face_rect)
        
        # 提取所有68个点的坐标
        landmarks = []
        for i in range(68):
            x = shape.part(i).x
            y = shape.part(i).y
            landmarks.append((x, y))
        
        return np.array(landmarks)

    def normalize_landmarks(self, landmarks, face_rect):
        """标准化面部特征点（相对于面部边界框）"""
        # 获取面部边界框
        x, y, w, h = (
            face_rect.left(),
            face_rect.top(),
            face_rect.width(),
            face_rect.height()
        )
        
        # 标准化到[0, 1]范围
        normalized = []
        for (px, py) in landmarks:
            nx = (px - x) / w
            ny = (py - y) / h
            normalized.append((nx, ny))
        
        return np.array(normalized)

    def extract_face_features(self, frames):
        """从视频帧序列中提取面部特征点"""
        features = []
        
        for frame in frames:
            # 检测人脸
            face_rect = self.detect_face(frame)
            if face_rect is None:
                # 如果未检测到人脸，使用上一帧的特征或默认值
                if features:
                    features.append(features[-1])
                else:
                    # 默认特征点（零矩阵）
                    features.append(np.zeros((68, 2)))
                continue
            
            # 提取并标准化特征点
            landmarks = self.extract_landmarks(frame, face_rect)
            normalized_landmarks = self.normalize_landmarks(landmarks, face_rect)
            
            # 展平特征点并添加到列表
            features.append(normalized_landmarks.flatten())
        
        return np.array(features)

    def crop_face(self, frame, face_rect, expand=0.2):
        """裁剪面部区域"""
        x, y, w, h = (
            face_rect.left(),
            face_rect.top(),
            face_rect.width(),
            face_rect.height()
        )
        
        # 扩展边界框
        expand_w = int(w * expand)
        expand_h = int(h * expand)
        x = max(0, x - expand_w)
        y = max(0, y - expand_h)
        w = min(frame.shape[1] - x, w + 2 * expand_w)
        h = min(frame.shape[0] - y, h + 2 * expand_h)
        
        # 裁剪并调整大小
        face = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (self.image_size, self.image_size))
        
        return face_resized

    def preprocess_frames(self, frames):
        """预处理视频帧（裁剪面部并转换为张量）"""
        processed = []
        
        for frame in frames:
            # 检测人脸并裁剪
            face_rect = self.detect_face(frame)
            if face_rect is not None:
                face = self.crop_face(frame, face_rect)
            else:
                # 如果未检测到人脸，使用整个帧调整大小
                face = cv2.resize(frame, (self.image_size, self.image_size))
            
            # 转换为张量
            face_tensor = self.transform(Image.fromarray(face))
            processed.append(face_tensor)
        
        return torch.stack(processed)

    def save_video(self, frames, output_path, fps=25, audio_path=None):
        """保存视频帧为视频文件，可选添加音频"""
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 转换为BGR格式（OpenCV要求）
        frames_bgr = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]
        
        # 设置编码器和创建VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width = frames_bgr[0].shape[:2]
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # 写入帧
        for frame in frames_bgr:
            out.write(frame)
        
        out.release()
        
        # 如果提供了音频，合并音视频
        if audio_path is not None and os.path.exists(audio_path):
            try:
                import moviepy.editor as mp
                
                # 临时文件
                temp_video = output_path.replace(".mp4", "_temp.mp4")
                os.rename(output_path, temp_video)
                
                # 合并音视频
                video = mp.VideoFileClip(temp_video)
                audio = mp.AudioFileClip(audio_path)
                
                # 确保音频长度与视频匹配
                if audio.duration > video.duration:
                    audio = audio.subclip(0, video.duration)
                else:
                    video = video.subclip(0, audio.duration)
                
                final_video = video.set_audio(audio)
                final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")
                
                # 清理临时文件
                video.close()
                audio.close()
                final_video.close()
                os.remove(temp_video)
                
            except Exception as e:
                print(f"合并音视频时出错: {str(e)}，将保存无音频视频")
                os.rename(temp_video, output_path)
        
        return output_path
    