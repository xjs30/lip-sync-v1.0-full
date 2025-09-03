import argparse
import os
import yaml
import torch
import gradio as gr
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import uvicorn
import tempfile
from datetime import datetime

# 导入内部模块
from models.latent_diffusion import LatentDiffusionModel
from utils.audio_processor import AudioProcessor
from utils.video_processor import VideoProcessor
from train.trainer import Trainer

# 全局配置和模型
config = None
model = None
audio_processor = None
video_processor = None
device = None

def load_resources(config_path="configs/model_config.yaml"):
    """加载配置和模型资源"""
    global config, model, audio_processor, video_processor, device
    
    # 加载配置
    config = yaml.safe_load(open(config_path))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 初始化处理器
    audio_processor = AudioProcessor(
        sample_rate=config['data']['audio_sample_rate'],
        n_mfcc=config['data']['n_mfcc']
    )
    
    video_processor = VideoProcessor(
        image_size=config['model']['image_size']
    )
    
    # 初始化并加载模型
    model = LatentDiffusionModel(config).to(device)
    
    # 加载预训练权重
    if os.path.exists(config['model']['pretrained_path']):
        checkpoint = torch.load(config['model']['pretrained_path'], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"Loaded pretrained model from {config['model']['pretrained_path']}")
    else:
        print("Warning: No pretrained model found. Using untrained model.")

def generate_lip_sync(video_path, audio_path, progress=gr.Progress()):
    """生成唇形同步视频"""
    global model, audio_processor, video_processor, device, config
    
    try:
        # 处理输入
        progress(0, desc="处理输入视频和音频")
        video_frames, fps = video_processor.load_video(video_path)
        audio_features = audio_processor.extract_features(audio_path)
        
        # 提取面部特征
        progress(0.2, desc="提取面部特征")
        face_features = video_processor.extract_face_features(video_frames)
        
        # 调整长度匹配
        min_length = min(len(video_frames), len(audio_features))
        video_frames = video_frames[:min_length]
        audio_features = audio_features[:min_length]
        face_features = face_features[:min_length]
        
        # 转换为张量并添加批次维度
        progress(0.4, desc="准备模型输入")
        audio_tensor = torch.FloatTensor(audio_features).unsqueeze(0).to(device)
        face_tensor = torch.FloatTensor(face_features).unsqueeze(0).to(device)
        
        # 生成唇形同步视频
        progress(0.6, desc="生成唇形同步视频")
        with torch.no_grad():
            generated_frames = model.sample(
                audio_tensor, 
                face_tensor,
                num_steps=config['inference']['num_steps'],
                guidance_scale=config['inference']['guidance_scale']
            )
        
        # 处理输出
        progress(0.8, desc="保存输出视频")
        generated_frames = generated_frames.squeeze(0).permute(0, 2, 3, 1).cpu().numpy()
        generated_frames = (generated_frames * 255).astype('uint8')
        
        # 合并音频和视频
        output_path = os.path.join(
            "outputs", 
            f"lip_sync_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        )
        os.makedirs("outputs", exist_ok=True)
        
        video_processor.save_video(
            generated_frames, 
            output_path, 
            fps=fps,
            audio_path=audio_path
        )
        
        progress(1.0, desc="完成")
        return output_path
        
    except Exception as e:
        print(f"Error generating lip sync: {str(e)}")
        raise gr.Error(f"生成失败: {str(e)}")

def create_web_interface():
    """创建Gradio Web界面"""
    with gr.Blocks(title="唇同步大模型V1.0", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 唇同步大模型V1.0")
        gr.Markdown("上传视频和音频，生成唇形同步的视频效果")
        
        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="输入视频（含人脸）")
                audio_input = gr.Audio(label="输入音频", type="filepath")
                generate_btn = gr.Button("生成唇同步视频", variant="primary")
            
            with gr.Column(scale=1):
                output_video = gr.Video(label="输出唇同步视频")
        
        generate_btn.click(
            fn=generate_lip_sync,
            inputs=[video_input, audio_input],
            outputs=output_video
        )
        
        gr.Examples(
            examples=[
                ["examples/speaker.mp4", "examples/voice.wav"],
                ["examples/presenter.mp4", "examples/speech.wav"]
            ],
            inputs=[video_input, audio_input],
            outputs=output_video,
            fn=generate_lip_sync
        )
    
    return demo

def create_api_service():
    """创建FastAPI服务"""
    app = FastAPI(title="唇同步大模型API")
    
    @app.post("/generate", response_class=FileResponse)
    async def api_generate(
        video_file: UploadFile = File(...),
        audio_file: UploadFile = File(...)
    ):
        # 保存上传文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as vf:
            vf.write(await video_file.read())
            video_path = vf.name
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as af:
            af.write(await audio_file.read())
            audio_path = af.name
        
        # 生成唇同步视频
        output_path = generate_lip_sync(video_path, audio_path)
        
        # 清理临时文件
        os.unlink(video_path)
        os.unlink(audio_path)
        
        return output_path
    
    return app

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="唇同步大模型V1.0")
    parser.add_argument("--mode", type=str, default="web", choices=["web", "api", "train"], 
                      help="运行模式: web(网页界面), api(接口服务), train(模型训练)")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml", 
                      help="配置文件路径")
    parser.add_argument("--port", type=int, default=7860, 
                      help="服务端口")
    
    args = parser.parse_args()
    
    # 加载资源
    load_resources(args.config)
    
    if args.mode == "web":
        # 启动Web界面
        demo = create_web_interface()
        demo.launch(server_port=args.port, share=False)
        
    elif args.mode == "api":
        # 启动API服务
        app = create_api_service()
        uvicorn.run(app, host="0.0.0.0", port=args.port)
        
    elif args.mode == "train":
        # 启动模型训练
        trainer = Trainer(args.config)
        trainer.train()

if __name__ == "__main__":
    main()
    