from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import tempfile
from dotenv import load_dotenv
from services.detection import process_traffic_video, process_traffic_image
from services.advisor import get_traffic_advice

load_dotenv()

app = FastAPI(title="TrafficSense API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

@app.get("/")
def read_root():
    return {"message": "TrafficSense API is running!"}

@app.post("/api/v1/analyze")
async def analyze_traffic(
    file: UploadFile = File(...),
    line_y: int = Form(None)
):
    suffix = os.path.splitext(file.filename or "")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_path = temp_file.name
        shutil.copyfileobj(file.file, temp_file)
        
    try:
        content_type = file.content_type
        
        if content_type.startswith('video/'):
            hasil_analisis = process_traffic_video(temp_path, line_y)
        elif content_type.startswith('image/'):
            hasil_analisis = process_traffic_image(temp_path)
        else:
            raise Exception("Format file tidak didukung. Harap unggah Gambar (JPG/PNG) atau Video (MP4).")

        status_jalan = hasil_analisis["kemacetan"]["status"]
        total_kendaraan = hasil_analisis["total_kendaraan"]
        
        pesan_ai = get_traffic_advice(total_kendaraan, status_jalan, GEMINI_API_KEY)
        
        hasil_analisis["pesan_ai_advisor"] = pesan_ai

        os.remove(temp_path)
        return {
            "status": "success",
            "data": hasil_analisis
        }
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return {"status": "error", "message": str(e)}
