try:
    from google import genai
except ImportError:
    genai = None

def get_fallback_advice(total_kendaraan, status):
    """Fungsi ini berjalan jika Gemini API gagal atau kuota habis."""
    if status == "MACET":
        return f"Waspada! Lalu lintas terpantau MACET dengan {total_kendaraan} kendaraan. Sangat disarankan mencari jalur alternatif untuk menghindari penumpukan."
    elif status == "PADAT":
        return f"Lalu lintas terlihat PADAT dengan {total_kendaraan} kendaraan. Harap berkendara dengan sabar dan jaga jarak aman."
    else:
        return f"Kondisi lalu lintas terpantau LANCAR dengan {total_kendaraan} kendaraan. Perjalanan Anda diprediksi lancar."

def get_traffic_advice(total_kendaraan, status_kemacetan, api_key):
    if genai is None or not api_key or api_key == "ISI_API_KEY_KAMU_DISINI":
        return get_fallback_advice(total_kendaraan, status_kemacetan)

    try:
        client = genai.Client(api_key=api_key)
        
        prompt = f"Total {total_kendaraan} kendaraan, status {status_kemacetan}. Berikan pesan mengenai kondisi lalu lintas saat ini secara singkat dan profesional."
        
        response = client.models.generate_content(
            model="gemini-1.5-flash", 
            contents=prompt,
        )
        return response.text.strip()
    
    except Exception as e:
        return get_fallback_advice(total_kendaraan, status_kemacetan)
