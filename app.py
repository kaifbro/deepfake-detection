import gradio as gr
import torch
import torch.nn as nn
import timm
import librosa
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import tempfile
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Image Model
image_model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=2)
image_model.load_state_dict(torch.load("models/deepfake_image_model.pth", map_location=device))
image_model = image_model.to(device)
image_model.eval()

# Load Audio Model
audio_model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
audio_model = nn.DataParallel(audio_model)
audio_model.load_state_dict(torch.load("models/deepfake_audio_model.pth", map_location=device))
audio_model = audio_model.to(device)
audio_model.eval()

# Transforms
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

audio_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(img):
    img_tensor = image_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = image_model(img_tensor)
        probs = torch.softmax(output, dim=1)
        real_prob = probs[0][0].item() * 100
        fake_prob = probs[0][1].item() * 100
    return real_prob, fake_prob

def predict_audio_file(audio_path):
    y, sr = librosa.load(audio_path, sr=16000, duration=5.0)
    if len(y) < sr * 5:
        y = np.pad(y, (0, sr * 5 - len(y)))
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = ((mel_db - mel_db.min()) / (mel_db.max() - mel_db.min()) * 255).astype(np.uint8)
    mel_img = Image.fromarray(mel_norm).convert("RGB").resize((224, 224))
    img_tensor = audio_transform(mel_img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = audio_model(img_tensor)
        probs = torch.softmax(output, dim=1)
        real_prob = probs[0][0].item() * 100
        fake_prob = probs[0][1].item() * 100
    return real_prob, fake_prob

def extract_frames(video_path, num_frames=30):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
    cap.release()
    return frames

def detect_image(image):
    real_prob, fake_prob = predict_image(image)
    verdict = "✅ REAL" if real_prob > fake_prob else "❌ FAKE"
    return f"{verdict}\n\nReal: {real_prob:.2f}%\nFake: {fake_prob:.2f}%"

def detect_video(video_path):
    frames = extract_frames(video_path)
    real_probs, fake_probs = [], []
    for frame in frames:
        real_prob, fake_prob = predict_image(frame)
        real_probs.append(real_prob)
        fake_probs.append(fake_prob)
    avg_real = np.mean(real_probs)
    avg_fake = np.mean(fake_probs)
    verdict = "✅ REAL" if avg_real > avg_fake else "❌ FAKE"
    return f"{verdict}\n\nReal: {avg_real:.2f}%\nFake: {avg_fake:.2f}%"

def detect_audio(audio_path):
    real_prob, fake_prob = predict_audio_file(audio_path)
    verdict = "✅ REAL" if real_prob > fake_prob else "❌ FAKE"
    return f"{verdict}\n\nReal: {real_prob:.2f}%\nFake: {fake_prob:.2f}%"

# Gradio Interface
with gr.Blocks(title="Deepfake Detector") as demo:
    gr.Markdown("# 🔍 Deepfake Detection System")
    gr.Markdown("Detect deepfakes in images, videos, and audio files.")
    
    with gr.Tab("🖼️ Image"):
        img_input = gr.Image(type="pil", label="Upload Image")
        img_output = gr.Textbox(label="Result")
        img_btn = gr.Button("Detect")
        img_btn.click(detect_image, inputs=img_input, outputs=img_output)
    
    with gr.Tab("🎥 Video"):
        vid_input = gr.Video(label="Upload Video")
        vid_output = gr.Textbox(label="Result")
        vid_btn = gr.Button("Detect")
        vid_btn.click(detect_video, inputs=vid_input, outputs=vid_output)
    
    with gr.Tab("🎵 Audio"):
        aud_input = gr.Audio(type="filepath", label="Upload Audio")
        aud_output = gr.Textbox(label="Result")
        aud_btn = gr.Button("Detect")
        aud_btn.click(detect_audio, inputs=aud_input, outputs=aud_output)

demo.launch()