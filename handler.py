import os
import torch
import torchaudio
import soundfile as sf
import runpod
import base64
import tempfile
from f5_tts.api import F5TTS

print("Loading F5-TTS model...")
f5tts = F5TTS(model_type="F5-TTS", device="cuda" if torch.cuda.is_available() else "cpu")
print("Model loaded successfully!")

def handler(job):
    job_input = job.get("input", {})
    gen_text = job_input.get("gen_text", "")
    ref_audio_b64 = job_input.get("ref_audio", None)
    ref_text = job_input.get("ref_text", "")
    speed = float(job_input.get("speed", 1.0))
    remove_silence = job_input.get("remove_silence", False)

    if not gen_text:
        return {"error": "Missing gen_text parameter."}

    ref_audio_path = None
    try:
        if ref_audio_b64:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_ref:
                temp_ref.write(base64.b64decode(ref_audio_b64))
                ref_audio_path = temp_ref.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_out:
            output_path = temp_out.name

        wav, sr, _ = f5tts.infer(
            ref_file=ref_audio_path if ref_audio_path else "",
            ref_text=ref_text,
            gen_text=gen_text,
            speed=speed,
            remove_silence=remove_silence
        )

        sf.write(output_path, wav, sr)

        with open(output_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")

        if ref_audio_path and os.path.exists(ref_audio_path):
            os.remove(ref_audio_path)
        if os.path.exists(output_path):
            os.remove(output_path)

        return {"audio_base64": audio_base64, "sampling_rate": sr}

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
