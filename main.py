from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
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

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

# ─── Load Image Model ───
image_model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=2)
image_model.load_state_dict(torch.load("models/deepfake_image_model.pth", map_location=device))
image_model = image_model.to(device)
image_model.eval()
print("Image model loaded!")

# ─── Load Audio Model ───
audio_model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
audio_model = nn.DataParallel(audio_model)
audio_model.load_state_dict(torch.load("models/deepfake_audio_model.pth", map_location=device))
audio_model = audio_model.to(device)
audio_model.eval()
print("Audio model loaded!")

# ─── Transforms ───
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

audio_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ─── Helper Functions ───
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

# ─── Routes ───
@app.get("/")
def root():
    return {"status": "Deepfake Detection API is running"}

@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    contents = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
        f.write(contents)
        tmp_path = f.name
    
    img = Image.open(tmp_path).convert("RGB")
    real_prob, fake_prob = predict_image(img)
    os.unlink(tmp_path)
    
    return {
        "real": round(real_prob, 2),
        "fake": round(fake_prob, 2),
        "verdict": "REAL" if real_prob > fake_prob else "FAKE"
    }

@app.post("/detect/video")
async def detect_video(file: UploadFile = File(...)):
    contents = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
        f.write(contents)
        tmp_path = f.name
    
    frames = extract_frames(tmp_path)
    os.unlink(tmp_path)
    
    real_probs, fake_probs = [], []
    for frame in frames:
        real_prob, fake_prob = predict_image(frame)
        real_probs.append(real_prob)
        fake_probs.append(fake_prob)
    
    avg_real = np.mean(real_probs)
    avg_fake = np.mean(fake_probs)
    
    return {
        "real": round(avg_real, 2),
        "fake": round(avg_fake, 2),
        "verdict": "REAL" if avg_real > avg_fake else "FAKE"
    }

@app.post("/detect/audio")
async def detect_audio(file: UploadFile = File(...)):
    contents = await file.read()
    suffix = ".wav" if file.filename.endswith(".wav") else ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(contents)
        tmp_path = f.name
    
    real_prob, fake_prob = predict_audio_file(tmp_path)
    os.unlink(tmp_path)
    
    return {
        "real": round(real_prob, 2),
        "fake": round(fake_prob, 2),
        "verdict": "REAL" if real_prob > fake_prob else "FAKE"
    }