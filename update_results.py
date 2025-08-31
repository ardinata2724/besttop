import requests
import os
from bs4 import BeautifulSoup
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

# ===== PERBAIKAN FINAL DI SINI =====
# URL target utama tempat semua data berada
ANGKANET_URL = "https://www.angkanet.org/" # diganti dari .tv menjadi .org

# Kamus untuk mencocokkan nama pasaran kita dengan nama yang ada di Angkanet
ANGKANET_MARKET_NAMES = {
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
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def get_latest_result(pasaran):
    """Mengambil hasil terbaru dari Angkanet dengan logika pencarian yang andal."""
    pasaran_lower = pasaran.lower()
    if pasaran_lower not in ANGKANET_MARKET_NAMES:
        print(f"Pasaran '{pasaran}' tidak dikonfigurasi untuk Angkanet. Dilewati.")
        return None

    market_name_to_find = ANGKANET_MARKET_NAMES[pasaran_lower]
    print(f"Mencari '{market_name_to_find}' di {ANGKANET_URL}")
    
    try:
        response = requests.get(ANGKANET_URL, headers=WEB_HEADERS, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        tables = soup.find_all('table')
        if not tables:
            print("Tidak ada tabel ditemukan di halaman.")
            return None

        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all('td')
                if len(cells) > 2:
                    market_cell_tag = cells[0].find('a')
                    if market_cell_tag:
                        current_market_name = market_cell_tag.text.strip()
                        if market_name_to_find.lower() in current_market_name.lower():
                            result_cell = cells[2]
                            result_numbers = result_cell.find_all('b')
                            result_str = "".join([num.text.strip() for num in result_numbers])
                            
                            if len(result_str) == 4 and result_str.isdigit():
                                print(f"Sukses mendapatkan hasil untuk {market_name_to_find}: {result_str}")
                                return result_str
            
        print(f"Tidak dapat menemukan baris yang cocok untuk '{market_name_to_find}'")
        return None

    except requests.exceptions.RequestException as e:
        print(f"Error saat mengakses URL {ANGKANET_URL}: {e}")
    except Exception as e:
        print(f"Terjadi error tak terduga saat memproses halaman: {e}")
    
    return None

def update_file(filename, new_result):
    """Membaca file, memeriksa duplikat, dan menambahkan hasil baru jika belum ada."""
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
    """Fungsi utama untuk menjalankan proses pembaruan."""
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
