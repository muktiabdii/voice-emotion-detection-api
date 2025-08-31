import torch
import torchaudio
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import os

# Inisialisasi FastAPI
app = FastAPI(title="Emotion Detection API")

# Load model & processor sekali saat startup
MODEL_NAME = "superb/wav2vec2-base-superb-er"
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)

# Mapping label ke bahasa Indonesia
label_mapping = {
    "ang": "Marah",
    "angry": "Marah",
    "hap": "Senang",
    "happy": "Senang",
    "sad": "Sedih",
    "neu": "Netral",
    "neutral": "Netral",
    "dis": "Jijik",
    "disgusted": "Jijik",
    "fea": "Takut",
    "fearful": "Takut",
    "sur": "Terkejut",
    "surprised": "Terkejut"
}

@app.post("/predict/")
async def predict_emotion(file: UploadFile = File(...)):
    # Validasi ekstensi file
    if not file.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="File harus berformat .wav")

    # Simpan file sementara
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())

    try:
        # Load audio
        speech_array, sampling_rate = torchaudio.load(temp_path)

        # Resample ke 16kHz jika perlu
        if sampling_rate != 16000:
            resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
            speech_array = resampler(speech_array)

        # Preprocess untuk model
        inputs = feature_extractor(speech_array.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")

        # Prediksi
        with torch.no_grad():
            logits = model(**inputs).logits
            predicted_id = torch.argmax(logits).item()
            predicted_label = model.config.id2label[predicted_id]

        # Konversi ke bahasa Indonesia
        emotion_id = label_mapping.get(predicted_label, predicted_label)

        return {
            "emotion": emotion_id
        }

    finally:
        # Hapus file sementara
        if os.path.exists(temp_path):
            os.remove(temp_path)

# Untuk development local
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
