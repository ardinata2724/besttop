import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone, timedelta

# --- KONFIGURASI FINAL ---
# Mapping nama file ke ID pasaran di sumber data yang baru
PASARAN_MAPPING = {
    'keluaran hongkongpools.txt': 'hk',
    'keluaran hongkong lotto.txt': 'hk',
    'keluaran sydneypools.txt': 'sdy',
    'keluaran sydney lotto.txt': 'sdy',
    'keluaran singapura.txt': 'sgp',
    'keluaran bullseye.txt': 'bl',
}

# URL dari sumber data yang baru dan lebih stabil
DATA_URL = "https://www.paitopaman.com/paito/"

def get_latest_result(pasaran_id):
    try:
        target_url = DATA_URL + pasaran_id
        print(f"Mengunjungi URL: {target_url}")
        
        # Gunakan header untuk menyamar sebagai browser biasa
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36'
        }
        
        response = requests.get(target_url, headers=headers, timeout=30)
        response.raise_for_status() # Cek jika ada error HTTP

        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Cari tabel dengan class 'table' dan ambil baris pertama di tbody
        first_row = soup.select_one("table.table tbody tr:first-child")
        
        if not first_row:
            print("Gagal menemukan baris pertama di tabel hasil.")
            return None

        # Ambil semua sel (td) di baris tersebut
        cells = first_row.find_all("td")
        
        # Hasil biasanya ada di sel terakhir
        if cells and len(cells) > 1:
            result = cells[-1].get_text(strip=True)
            if len(result) == 4 and result.isdigit():
                print(f"Sukses mendapatkan hasil untuk {pasaran_id.upper()}: {result}")
                return result
            else:
                print(f"Format hasil tidak valid: '{result}'")
                return None
        else:
            print("Tidak ditemukan sel yang cukup di baris pertama.")
            return None

    except Exception as e:
        print(f"Terjadi error saat memproses {pasaran_id.upper()}: {e}")
        return None

def update_file(filename, new_result):
    if not os.path.exists(filename):
        existing_results = set()
    else:
        with open(filename, 'r', encoding='utf-8') as f:
            existing_results = set(line.strip() for line in f if line.strip())
            
    if new_result and new_result not in existing_results:
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(f"\n{new_result}")
        print(f"HASIL BARU DITAMBAHKAN: {new_result} -> {filename}")
        return True
    elif new_result:
        print(f"Hasil {new_result} sudah ada di {filename}. Tidak ada perubahan.")
    return False

def main():
    wib = timezone(timedelta(hours=7))
    print(f"--- Memulai proses pembaruan pada {datetime.now(wib).strftime('%Y-%m-%d %H:%M:%S WIB')} ---")
    
    any_file_updated = False
    for filename, pasaran_id in PASARAN_MAPPING.items():
        print(f"\nMemproses file: {filename}")
        latest_result = get_latest_result(pasaran_id)
        if latest_result:
            if update_file(filename, latest_result): 
                any_file_updated = True
        else: 
            print(f"Tidak dapat mengambil hasil terbaru untuk {pasaran_id.upper()}.")
            
    print("\n--- Proses pembaruan selesai. ---")
    if not any_file_updated:
        print("PERINGATAN: Tidak ada satu pun file yang diperbarui. Proses akan ditandai sebagai gagal.")
        exit(1)
    else:
        print("Pembaruan berhasil. Setidaknya satu file telah diubah.")

if __name__ == "__main__":
    main()
