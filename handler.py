import os
import io
import base64
import requests
import soundfile as sf
import runpod
from f5_tts.api import F5TTS

print("Loading F5-TTS model...")
f5tts = F5TTS()
print("F5-TTS model loaded successfully.")

# Default sample voice reference from F5-TTS package
DEFAULT_REF_AUDIO = os.path.join(os.path.dirname(f5tts.__file__ if hasattr(f5tts, '__file__') else "/usr/local/lib/python3.10/site-packages/f5_tts"), "infer/examples/basic/basic_ref_en.wav")
DEFAULT_REF_TEXT = "Some call me nature, others call me mother nature."

def handler(job):
    job_input = job.get("input", {})
    gen_text = job_input.get("gen_text", "")

    if not gen_text:
        return {"error": "Missing 'gen_text' in input."}

    ref_audio = job_input.get("ref_audio", "")
    ref_text = job_input.get("ref_text", "")

    ref_audio_path = "/tmp/ref_audio.wav"

    # If user provided a reference audio (URL or base64)
    if ref_audio:
        if ref_audio.startswith("http://") or ref_audio.startswith("https://"):
            res = requests.get(ref_audio)
            with open(ref_audio_path, "wb") as f:
                f.write(res.content)
        elif len(ref_audio) > 300: # Base64 string
            with open(ref_audio_path, "wb") as f:
                f.write(base64.b64decode(ref_audio))
        else:
            ref_audio_path = ref_audio
    else:
        # Use default reference if none provided
        if os.path.exists(DEFAULT_REF_AUDIO):
            ref_audio_path = DEFAULT_REF_AUDIO
            ref_text = ref_text or DEFAULT_REF_TEXT
        else:
            ref_audio_path = ""

    # Audio inference with reference parameters
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
