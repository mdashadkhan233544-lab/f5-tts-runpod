import os
import sys
import runpod
import torch
import soundfile as sf
import base64
import subprocess

# ১. মিসিং লাইব্রেরি ইনস্টলেশন
try:
    import gdown
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
    import gdown

try:
    from transformers import VitsModel, AutoTokenizer
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers", "soundfile"])
    from transformers import VitsModel, AutoTokenizer

# ২. মডেল রাখার ডিরেক্টরি সেটআপ
MODELS_DIR = "/app/models"
os.makedirs(MODELS_DIR, exist_ok=True)

PTH_PATH = os.path.join(MODELS_DIR, "asad_male.pth")
INDEX_PATH = os.path.join(MODELS_DIR, "asad_male.index")

# ৩. ড্রাইভ থেকে Asad মডেল অটো-ডাউনলোড
def download_voice_models():
    if not os.path.exists(PTH_PATH) or os.path.getsize(PTH_PATH) == 0:
        print("Downloading asad_male.pth from Google Drive...")
        gdown.download(id="1wCNWpfYQOwetefTBqDwBvLxNLSRaHM9Z", output=PTH_PATH, quiet=False)

    if not os.path.exists(INDEX_PATH) or os.path.getsize(INDEX_PATH) == 0:
        print("Downloading asad_male.index from Google Drive...")
        gdown.download(id="1Fj4fFNuYRPMzJm1AjQzH_kIGTQGf4QHR", output=INDEX_PATH, quiet=False)

download_voice_models()

# ৪. Meta MMS-TTS লোড করা
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading MMS-TTS model on {device}...")
tts_model_name = "facebook/mms-tts-ben"
tokenizer = AutoTokenizer.from_pretrained(tts_model_name)
tts_model = VitsModel.from_pretrained(tts_model_name).to(device)

def generate_base_audio(text: str) -> str:
    """বাংলা টেক্সট থেকে বেস অডিও তৈরি"""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        output = tts_model(**inputs).waveform
    
    base_audio_path = "/tmp/base_tts.wav"
    sf.write(base_audio_path, output.cpu().numpy().squeeze(), tts_model.config.sampling_rate)
    return base_audio_path

def convert_voice_with_rvc(input_audio_path: str) -> str:
    """RVC ভয়েস কনভার্সন"""
    output_audio_path = "/tmp/final_output.wav"
    
    # rvc-python রানটাইম লোড
    try:
        from rvc_python.infer import RVCInference
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-build-isolation", "rvc-python"])
        from rvc_python.infer import RVCInference

    try:
        rvc = RVCInference(device=device)
        rvc.load_model(PTH_PATH)
        rvc.infer_file(
            input_path=input_audio_path,
            output_path=output_audio_path,
            index_path=INDEX_PATH if os.path.exists(INDEX_PATH) else None,
            f0_up_key=0,
            f0_method="rmvpe",
            index_rate=0.75,
            protect=0.33
        )
        if os.path.exists(output_audio_path):
            return output_audio_path
    except Exception as e:
        print(f"RVC Error: {str(e)}")
        
    return input_audio_path

def handler(event):
    input_data = event.get("input", {})
    text = input_data.get("gen_text") or input_data.get("text", "")
    engine_mode = input_data.get("engine_mode", "mms_rvc")
    
    if not text:
        return {"error": "টেক্সট পাওয়া যায়নি।"}
    
    # ধাপ ১: বাংলা TTS অডিও তৈরি
    base_audio = generate_base_audio(text)
    
    # ধাপ ২: আপনার প্রশিক্ষিত আসল কণ্ঠে রূপান্তর
    if "rvc" in engine_mode.lower():
        final_audio = convert_voice_with_rvc(base_audio)
    else:
        final_audio = base_audio
    
    # ধাপ ৩: Base64 এনকোড করে ক্লায়েন্টে পাঠানো
    with open(final_audio, "rb") as f:
        audio_base64 = base64.b64encode(f.read()).decode("utf-8")
        
    return {
        "status": "success",
        "audio_base64": audio_base64
    }

runpod.serverless.start({"handler": handler})
