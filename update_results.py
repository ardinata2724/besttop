import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone, timedelta

# --- KONFIGURASI FINAL UNTUK SUMBER DATA BARU ---
PASARAN_MAPPING = {
    'keluaran hongkongpools.txt': 'hongkong',
    'keluaran hongkong lotto.txt': 'hongkong',
    'keluaran sydneypools.txt': 'sydney',
    'keluaran sydney lotto.txt': 'sydney',
    'keluaran singapura.txt': 'sgp',
    'keluaran bullseye.txt': 'bullseye',
}

DATA_URL = "https://togelmaster.org/paito/"

def get_latest_result(pasaran_path):
    try:
        target_url = DATA_URL + pasaran_path
        print(f"Mengunjungi URL: {target_url}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36'
        }
        
        response = requests.get(target_url, headers=headers, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        
        # [PERBAIKAN] Menggunakan selector yang lebih spesifik untuk tabel data di togelmaster.org
        # Mencari tabel di dalam div dengan class 'paito-month-group'
        data_table = soup.select_one("div.paito-month-group table.wla-daily-datatable")

        if not data_table:
            print("Gagal menemukan tabel data utama.")
            return None
        
        # Ambil baris pertama di dalam tbody tabel tersebut
        first_row = data_table.select_one("tbody tr:first-child")
        
        if not first_row:
            print("Gagal menemukan baris pertama di dalam tabel hasil.")
            return None

        # Ambil sel kedua (kolom 'Result') dari baris tersebut
        result_cell = first_row.select_one("td:nth-child(2)")
        
        if result_cell:
            result = result_cell.get_text(strip=True)
            if len(result) >= 4 and result.isdigit():
                result = result[-4:]
                print(f"Sukses mendapatkan hasil untuk {pasaran_path.upper()}: {result}")
                return result
            else:
                print(f"Format hasil tidak valid: '{result}'")
                return None
        else:
            print("Tidak ditemukan sel hasil di baris pertama.")
            return None

    except Exception as e:
        print(f"Terjadi error saat memproses {pasaran_path.upper()}: {e}")
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
    for filename, pasaran_path in PASARAN_MAPPING.items():
        print(f"\nMemproses file: {filename}")
        latest_result = get_latest_result(pasaran_path)
        if latest_result:
            if update_file(filename, latest_result): 
                any_file_updated = True
        else: 
            print(f"Tidak dapat mengambil hasil terbaru untuk {pasaran_path.upper()}.") [cite: 1]
            
    print("\n--- Proses pembaruan selesai. ---")
    if not any_file_updated:
        print("PERINGATAN: Tidak ada satu pun file yang diperbarui. Proses akan ditandai sebagai gagal.")
        exit(1)
    else:
        print("Pembaruan berhasil. Setidaknya satu file telah diubah.")

if __name__ == "__main__":
    main()
