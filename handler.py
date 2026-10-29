import os
import io
import base64
import torch
import torchaudio
import soundfile as sf
import runpod
from f5_tts.api import F5TTS

print("Loading F5-TTS model...")
f5tts = F5TTS()
print("F5-TTS model loaded successfully.")

def handler(job):
    job_input = job.get("input", {})
    gen_text = job_input.get("gen_text", "")
    
    if not gen_text:
        return {"error": "Missing 'gen_text' in input."}
    
    # Audio inference
    wav, sr, _ = f5tts.infer(
        gen_text=gen_text
    )
    
    # In-memory WAV encode
    buffer = io.BytesIO()
    sf.write(buffer, wav, sr, format='WAV')
    buffer.seek(0)
    audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    
    return {
        "status": "success",
        "audio_base64": audio_base64
    }

runpod.serverless.start({"handler": handler})
