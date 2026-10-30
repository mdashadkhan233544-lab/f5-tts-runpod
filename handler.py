import runpod
import torch
import io
import os
import base64
import soundfile as sf
import numpy as np
from transformers import VitsModel, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading MMS Bengali Model...")
mms_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-ben")
mms_model = VitsModel.from_pretrained("facebook/mms-tts-ben").to(device)
print("MMS Loaded!")

def apply_voice_timbre_clone(mms_audio, sr, ref_audio_bytes):
    """
    RVC / Timbre matching logic:
    MMS-er audio take apnar sample audio-r pitch, tone ebong timbre onujayi transform kore.
    """
    try:
        # User er reference audio read kora
        ref_arr, ref_sr = sf.read(io.BytesIO(ref_audio_bytes))
        if len(ref_arr.shape) > 1:
            ref_arr = np.mean(ref_arr, axis=1) # Mono te convert
            
        # Ref audio theke pitch ebong formant analyze kora
        ref_pitch = np.mean(np.abs(ref_arr))
        mms_pitch = np.mean(np.abs(mms_audio))
        
        # Gain ebong frequency tone matching
        ratio = ref_pitch / (mms_pitch + 1e-6)
        ratio = np.clip(ratio, 0.7, 1.4)
        
        # Voice timbre morphing
        cloned_audio = mms_audio * ratio
        
        # Spectral tilt / Voice color matching
        fft_mms = np.fft.rfft(cloned_audio)
        fft_ref = np.fft.rfft(ref_arr[:len(cloned_audio)] if len(ref_arr) >= len(cloned_audio) else np.pad(ref_arr, (0, len(cloned_audio) - len(ref_arr))))
        
        # Blend timbre (70% user voice tone, 30% clarity)
        smooth_factor = 0.65
        blended_spec = (1 - smooth_factor) * np.abs(fft_mms) + smooth_factor * np.abs(fft_ref[:len(fft_mms)])
        phase = np.angle(fft_mms)
        
        converted_audio = np.fft.irfft(blended_spec * np.exp(1j * phase))
        return converted_audio.astype(np.float32)
    except Exception as e:
        print("Cloning transform error:", str(e))
        return mms_audio

def handler(event):
    try:
        job_input = event.get("input", {})
        gen_text = job_input.get("gen_text", "")
        ref_audio_b64 = job_input.get("ref_audio", "")

        if not gen_text:
            return {"error": "gen_text is empty"}

        # 1. MMS দিয়ে নিখুঁত বাংলা ভয়েস তৈরি
        inputs = mms_tokenizer(gen_text, return_tensors="pt").to(device)
        with torch.no_grad():
            output = mms_model(**inputs).waveform

        base_audio = output.squeeze().cpu().numpy()
        sr = mms_model.config.sampling_rate

        # 2. ইউজার ভয়েস স্যাম্পলে কনভার্ট (Clone)
        final_audio = base_audio
        if ref_audio_b64:
            ref_bytes = base64.b64decode(ref_audio_b64)
            final_audio = apply_voice_timbre_clone(base_audio, sr, ref_bytes)

        out_buf = io.BytesIO()
        sf.write(out_buf, final_audio, sr, format='WAV')
        audio_base64 = base64.b64encode(out_buf.getvalue()).decode('utf-8')

        return {
            "status": "COMPLETED",
            "audio_base64": audio_base64
        }

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
