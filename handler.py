import runpod
import torch
import io
import base64
import soundfile as sf
import numpy as np
from transformers import VitsModel, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading MMS Bengali Base Model...")
mms_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-ben")
mms_model = VitsModel.from_pretrained("facebook/mms-tts-ben").to(device)
print("MMS Loaded Successfully!")

def handler(event):
    try:
        job_input = event.get("input", {})
        gen_text = job_input.get("gen_text", "")
        ref_audio_b64 = job_input.get("ref_audio", "")

        if not gen_text:
            return {"error": "gen_text is empty"}

        # 1. MMS দিয়ে নিখুঁত বাংলা অডিও তৈরি
        inputs = mms_tokenizer(gen_text, return_tensors="pt").to(device)
        with torch.no_grad():
            output = mms_model(**inputs).waveform

        base_audio = output.squeeze().cpu().numpy()
        sr = mms_model.config.sampling_rate

        # 2. ইউজার রেফারেন্স অডিও থাকলে ভয়েস ম্যাচিং / ক্লোনিং প্রসেস
        final_audio = base_audio
        if ref_audio_b64:
            try:
                # Reference audio load kore pitch & timbre match korano
                ref_bytes = base64.b64decode(ref_audio_b64)
                ref_arr, ref_sr = sf.read(io.BytesIO(ref_bytes))
                
                # Pitch adjustment (যদি টোন আলাদা হয়)
                # এখানে RVC ইনফারেন্স কল হবে
                final_audio = base_audio  
            except Exception as e:
                print("Voice match error:", str(e))

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
