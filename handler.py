import os
import io
import base64
import requests
import soundfile as sf
import numpy as np
import torch
import runpod

# Load F5-TTS
from f5_tts.api import F5TTS
print("Loading F5-TTS...")
f5tts = F5TTS()

# Load Meta MMS-TTS (Bengali)
from transformers import VitsModel, AutoTokenizer
print("Loading Meta MMS Bengali TTS...")
mms_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-ben")
mms_model = VitsModel.from_pretrained("facebook/mms-tts-ben")

device = "cuda" if torch.cuda.is_available() else "cpu"
mms_model.to(device)

def handler(job):
    job_input = job.get("input", {})
    text = job_input.get("gen_text", "").strip()
    mode = job_input.get("engine_mode", "mms_bengali")
    voice_type = job_input.get("mms_voice", "Default Clear")
    
    if not text:
        return {"error": "Missing 'gen_text'"}

    # --- MODE 1: NATIVE BENGALI (Meta MMS) ---
    if mode == "mms_bengali":
        try:
            inputs = mms_tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Voice style variation adjustments
            noise_scale = 0.667
            noise_scale_w = 0.8
            length_scale = 1.0

            if voice_type == "Deep Story Male":
                noise_scale = 0.5
                length_scale = 1.15
            elif voice_type == "Radio News Fast":
                noise_scale = 0.4
                length_scale = 0.88
            elif voice_type == "Soft Narration":
                noise_scale = 0.8
                length_scale = 1.05

            with torch.no_grad():
                output = mms_model(
                    **inputs,
                    noise_scale=noise_scale,
                    noise_scale_w=noise_scale_w,
                    length_scale=length_scale
                ).waveform
            
            wav = output.squeeze().cpu().numpy()
            sr = mms_model.config.sampling_rate

            buffer = io.BytesIO()
            sf.write(buffer, wav, sr, format='WAV')
            buffer.seek(0)
            return {"status": "success", "audio_base64": base64.b64encode(buffer.read()).decode('utf-8')}
        except Exception as e:
            return {"error": f"MMS Bengali Error: {str(e)}"}

    # --- MODE 2: CUSTOM VOICE CLONE (F5-TTS) ---
    else:
        ref_audio = job_input.get("ref_audio", "")
        ref_text = job_input.get("ref_text", "")
        ref_path = "/tmp/ref_voice.wav"

        if ref_audio:
            if len(ref_audio) > 300:
                with open(ref_path, "wb") as f:
                    f.write(base64.b64decode(ref_audio))
            elif ref_audio.startswith("http"):
                res = requests.get(ref_audio)
                with open(ref_path, "wb") as f:
                    f.write(res.content)
        else:
            return {"error": "F5-TTS requires reference audio for cloning."}

        try:
            wav, sr, _ = f5tts.infer(
                ref_file=ref_path,
                ref_text=ref_text,
                gen_text=text
            )
            buffer = io.BytesIO()
            sf.write(buffer, wav, sr, format='WAV')
            buffer.seek(0)
            return {"status": "success", "audio_base64": base64.b64encode(buffer.read()).decode('utf-8')}
        except Exception as e:
            return {"error": f"F5-TTS Clone Error: {str(e)}"}

runpod.serverless.start({"handler": handler})
