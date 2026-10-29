import os
import io
import base64
import requests
import soundfile as sf
import numpy as np
import runpod
from f5_tts.api import F5TTS

print("Loading F5-TTS model...")
f5tts = F5TTS()
print("F5-TTS model loaded successfully.")

# Create a clean fallback reference audio in-memory/temp to avoid missing file errors
DEFAULT_REF_AUDIO = "/tmp/default_ref.wav"
DEFAULT_REF_TEXT = "Hello, this is default reference audio."

if not os.path.exists(DEFAULT_REF_AUDIO):
    # 2 seconds of silent/subtle wave so F5-TTS always has a valid WAV file
    sample_rate = 24000
    silence = np.zeros(sample_rate * 2, dtype=np.float32)
    sf.write(DEFAULT_REF_AUDIO, silence, sample_rate, format='WAV')

def handler(job):
    job_input = job.get("input", {})
    gen_text = job_input.get("gen_text", "")

    if not gen_text:
        return {"error": "Missing 'gen_text' in input."}

    ref_audio = job_input.get("ref_audio", "")
    ref_text = job_input.get("ref_text", "")

    ref_audio_path = "/tmp/custom_ref.wav"

    # User provided reference audio (URL, Base64, or file path)
    if ref_audio:
        try:
            if ref_audio.startswith("http://") or ref_audio.startswith("https://"):
                res = requests.get(ref_audio)
                with open(ref_audio_path, "wb") as f:
                    f.write(res.content)
            elif len(ref_audio) > 300:
                with open(ref_audio_path, "wb") as f:
                    f.write(base64.b64decode(ref_audio))
            else:
                ref_audio_path = ref_audio
        except Exception as e:
            return {"error": f"Failed to load ref_audio: {str(e)}"}
    else:
        ref_audio_path = DEFAULT_REF_AUDIO
        ref_text = ref_text or DEFAULT_REF_TEXT

    # Audio inference
    try:
        wav, sr, _ = f5tts.infer(
            ref_file=ref_audio_path,
            ref_text=ref_text,
            gen_text=gen_text
        )
    except Exception as e:
        return {"error": str(e)}

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
