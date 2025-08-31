import requests
import os
from datetime import datetime, timezone, timedelta

# --- KONFIGURASI ---
PASARAN_FILES = {
    'hongkongpools': 'keluaran hongkongpools.txt',
    'hongkong': 'keluaran hongkong lotto.txt',
    'sydneypools': 'keluaran sydneypools.txt',
    'sydney': 'keluaran sydney lotto.txt',
    'singapore': 'keluaran singapura.txt',
    'bullseye': 'keluaran bullseye.txt',
    'moroccoquatro18': 'keluaran morocco quatro 18.txt',
    'moroccoquatro21': 'keluaran morocco quatro 21.txt',
    'moroccoquatro00': 'keluaran morocco quatro 00.txt',
}

# ===== MODIFIKASI FINAL: Menggunakan "API Tersembunyi" dari Angkanet =====
ANGKANET_API_URL = "https://www.angkanet.org/api/livedraw"

# Kamus untuk mencocokkan nama pasaran kita dengan nama di API Angkanet
ANGKANET_API_NAMES = {
    'hongkongpools': 'Hongkong Pools',
    'hongkong': 'Hongkong Pools',
    'sydneypools': 'Sydney Pools',
    'sydney': 'Sydney Pools',
    'singapore': 'Singapore',
    'bullseye': 'Bullseye',
    'moroccoquatro21': 'Morocco Quatro 21.00 Wib',
    'moroccoquatro18': 'Morocco Quatro 18.00 Wib',
    'moroccoquatro00': 'Morocco Quatro 00.00 Wib',
}

WEB_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Referer': 'https://www.angkanet.org/'
}

# Cache untuk menyimpan hasil API sementara agar tidak meminta berulang kali
api_data_cache = None

def get_latest_result(pasaran):
    """Mengambil hasil terbaru dari API tersembunyi Angkanet."""
    global api_data_cache
    
    pasaran_lower = pasaran.lower()
    if pasaran_lower not in ANGKANET_API_NAMES:
        print(f"Pasaran '{pasaran}' tidak dikonfigurasi untuk API Angkanet. Dilewati.")
        return None

    market_name_to_find = ANGKANET_API_NAMES[pasaran_lower]
    
    try:
        # Jika cache kosong, panggil API. Jika sudah ada isinya, gunakan cache.
        if api_data_cache is None:
            print(f"Cache kosong, mengambil data baru dari {ANGKANET_API_URL}")
            response = requests.get(ANGKANET_API_URL, headers=WEB_HEADERS, timeout=20)
            response.raise_for_status()
            api_data_cache = response.json()
        else:
            print("Menggunakan data dari cache.")

        # Mencari data di dalam hasil JSON dari API
        for market_data in api_data_cache:
            if market_name_to_find.lower() in market_data.get('market', '').lower():
                result = market_data.get('result', '').strip()
                if len(result) == 4 and result.isdigit():
                    print(f"Sukses mendapatkan hasil untuk {market_name_to_find}: {result}")
                    return result
        
        print(f"Tidak dapat menemukan data untuk '{market_name_to_find}' di dalam respons API.")
        return None

    except requests.exceptions.RequestException as e:
        print(f"Error saat mengakses API Angkanet untuk {pasaran}: {e}")
    except Exception as e:
        print(f"Terjadi error tak terduga saat memproses data API untuk {pasaran}: {e}")
        
    return None

# V V V V V (TIDAK ADA PERUBAHAN PADA FUNGSI DI BAWAH INI) V V V V V
def update_file(filename, new_result):
    if not os.path.exists(filename):
        print(f"File {filename} tidak ditemukan. Membuat file baru.")
        existing_results = set()
    else:
        with open(filename, 'r', encoding='utf-8') as f:
            existing_results = set(line.strip() for line in f if line.strip())

    if new_result not in existing_results:
        print(f"Hasil baru ditemukan untuk {filename}: {new_result}. Menambahkan ke file.")
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(f"\n{new_result}")
        return True
    else:
        print(f"Hasil {new_result} sudah ada di {filename}. Tidak ada perubahan.")
        return False

def main():
    wib = timezone(timedelta(hours=7))
    print(f"--- Memulai proses pembaruan pada {datetime.now(wib).strftime('%Y-%m-%d %H:%M:%S WIB')} ---")
    
    any_file_updated = False
    for pasaran, filename in PASARAN_FILES.items():
        print(f"\nMemproses pasaran: {pasaran.capitalize()}")
        latest_result = get_latest_result(pasaran)
        
        if latest_result:
            if update_file(filename, latest_result):
                any_file_updated = True
        else:
            print(f"Tidak dapat mengambil hasil terbaru untuk {pasaran}.")
            
    print("\n--- Proses pembaruan selesai. ---")
    if not any_file_updated:
        print("Tidak ada file yang diperbarui. Keluar.")
        exit(0)

if __name__ == "__main__":
    main()
