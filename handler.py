import os
import io
import base64
import requests
import soundfile as sf
import urllib.request
import runpod
from f5_tts.api import F5TTS

print("Loading F5-TTS model...")
f5tts = F5TTS()
print("F5-TTS model loaded successfully.")

DEFAULT_REF_AUDIO = "/tmp/default_ref.wav"
DEFAULT_REF_TEXT = "Some call me nature, others call me mother nature."

if not os.path.exists(DEFAULT_REF_AUDIO):
    sample_url = "https://raw.githubusercontent.com/SWivid/F5-TTS/main/src/f5_tts/infer/examples/basic/basic_ref_en.wav"
    try:
        urllib.request.urlretrieve(sample_url, DEFAULT_REF_AUDIO)
    except Exception as e:
        print(f"Failed to fetch sample: {e}")

def handler(job):
    job_input = job.get("input", {})
    gen_text = job_input.get("gen_text", "").strip()

    if not gen_text:
        return {"error": "Missing 'gen_text' in input."}

    ref_audio = job_input.get("ref_audio", "")
    ref_text = job_input.get("ref_text", "").strip()

    ref_audio_path = "/tmp/custom_ref.wav"

    # User-provided reference voice handling
    if ref_audio:
        try:
            if ref_audio.startswith("http://") or ref_audio.startswith("https://"):
                res = requests.get(ref_audio, timeout=30)
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

    # Inference with cross-lingual support
    try:
        wav, sr, _ = f5tts.infer(
            ref_file=ref_audio_path,
            ref_text=ref_text,
            gen_text=gen_text,
            speed=1.0
        )
    except Exception as e:
        return {"error": str(e)}

    buffer = io.BytesIO()
    sf.write(buffer, wav, sr, format='WAV')
    buffer.seek(0)
    audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    return {
        "status": "success",
        "audio_base64": audio_base64
    }

runpod.serverless.start({"handler": handler})
