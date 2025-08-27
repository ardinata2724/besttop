import requests
import os
from datetime import datetime, timezone, timedelta

# --- KONFIGURASI ---
# Kamus (dictionary) yang memetakan nama pasaran di API ke nama file Anda
PASARAN_FILES = {
    'hongkongpools': 'keluaran hongkongpools.txt',
    'hongkong': 'keluaran hongkong lotto.txt',
    'sydneypools': 'keluaran sydneypools.txt',
    'sydney': 'keluaran sydney lotto.txt',
    'bullseye': 'keluaran bullseye.txt',
    'singapore': 'keluaran singapura.txt',
    # Konfigurasi untuk pasaran Maroko sudah benar di sini
    'moroccoquatro18': 'keluaran morocco quatro 18.txt',
    'moroccoquatro21': 'keluaran morocco quatro 21.txt',
    'moroccoquatro00': 'keluaran morocco quatro 00.txt',
}

# Header otorisasi untuk API
HEADERS = {
    "Authorization": "Bearer 6705327a2c9a9135f2c8fbad19f09b46"
}

def get_latest_result(pasaran):
    """Mengambil satu hasil terbaru dari API untuk pasaran tertentu."""
    # Menggunakan putaran=1 dan urut=desc untuk efisiensi
    url = f"https://wysiwygscan.com/api?pasaran={pasaran.lower()}&hari=harian&putaran=1&format=json&urut=desc"
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()  # Cek jika ada error HTTP
        data = response.json()
        
        if data.get("data") and len(data["data"]) > 0:
            latest_entry = data["data"][0]
            result = str(latest_entry.get("result", "")).strip()
            if len(result) == 4 and result.isdigit():
                return result
    except requests.exceptions.RequestException as e:
        print(f"Error saat mengambil data untuk {pasaran}: {e}")
    return None

def update_file(filename, new_result):
    """Membaca file, memeriksa duplikat, dan menambahkan hasil baru jika belum ada."""
    if not os.path.exists(filename):
        print(f"File {filename} tidak ditemukan. Membuat file baru.")
        existing_results = set()
    else:
        with open(filename, 'r', encoding='utf-8') as f:
            # Membaca semua baris dan menghapus spasi/newline yang tidak perlu
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
    """Fungsi utama untuk menjalankan proses pembaruan."""
    # Set zona waktu ke Waktu Indonesia Barat (WIB)
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
    # Baris ini penting untuk langkah selanjutnya di GitHub Actions
    if not any_file_updated:
        print("Tidak ada file yang diperbarui. Keluar.")
        exit(0)


if __name__ == "__main__":
    main()
