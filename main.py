import torch
import torchaudio
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import os

app = FastAPI(title="Emotion Detection API")

MODEL_NAME = "superb/wav2vec2-base-superb-er"
model = None
feature_extractor = None

# Mapping label
label_mapping = {
    "ang": "Marah", "angry": "Marah",
    "hap": "Senang", "happy": "Senang",
    "sad": "Sedih",
    "neu": "Netral", "neutral": "Netral",
    "dis": "Jijik", "disgusted": "Jijik",
    "fea": "Takut", "fearful": "Takut",
    "sur": "Terkejut", "surprised": "Terkejut"
}

@app.on_event("startup")
def load_model():
    global model, feature_extractor
    torch.set_num_threads(1)  # hemat CPU
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)

@app.post("/predict/")
async def predict_emotion(file: UploadFile = File(...)):
    if not file.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="File harus berformat .wav")

    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())

    try:
        speech_array, sampling_rate = torchaudio.load(temp_path)
        if sampling_rate != 16000:
            resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
            speech_array = resampler(speech_array)

        inputs = feature_extractor(speech_array.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")

        with torch.no_grad():
            logits = model(**inputs).logits
            predicted_id = torch.argmax(logits).item()
            predicted_label = model.config.id2label[predicted_id]

        emotion_id = label_mapping.get(predicted_label, predicted_label)
        return {"emotion": emotion_id}

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
