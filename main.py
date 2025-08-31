import torch
import torchaudio
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
import uvicorn
import os
from tempfile import NamedTemporaryFile

# === FastAPI App ===
app = FastAPI(title="Emotion Detection API")

# === Environment Settings ===
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_CACHE"] = "./models"

# Buat folder cache untuk model
os.makedirs("./models", exist_ok=True)

MODEL_NAME = "superb/wav2vec2-base-superb-er"
model = None
feature_extractor = None

# === Label Mapping ===
label_mapping = {
    "ang": "Marah", "angry": "Marah",
    "hap": "Senang", "happy": "Senang",
    "sad": "Sedih",
    "neu": "Netral", "neutral": "Netral",
    "dis": "Jijik", "disgusted": "Jijik",
    "fea": "Takut", "fearful": "Takut",
    "sur": "Terkejut", "surprised": "Terkejut"
}

# === Load Model on Startup ===
@app.on_event("startup")
def load_model():
    global model, feature_extractor
    torch.set_num_threads(1)  # Hemat CPU thread
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)

# === Predict Endpoint ===
@app.post("/predict/")
async def predict_emotion(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    # Validasi format file
    if not file.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="File harus berformat .wav")

    # Simpan file sementara dengan nama unik
    with NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(await file.read())
        temp_path = temp_file.name

    # Hapus file di background setelah selesai
    if background_tasks:
        background_tasks.add_task(os.remove, temp_path)

    # Proses audio
    speech_array, sampling_rate = torchaudio.load(temp_path)

    # Resample ke 16kHz jika perlu
    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
        speech_array = resampler(speech_array)

    # Preprocessing untuk model
    inputs = feature_extractor(speech_array.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")

    # Prediksi
    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_id = torch.argmax(logits).item()
        predicted_label = model.config.id2label[predicted_id]

    # Konversi ke Bahasa Indonesia
    emotion_id = label_mapping.get(predicted_label, predicted_label)

    return {"emotion": emotion_id}

# === Run Server (for local dev) ===
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
