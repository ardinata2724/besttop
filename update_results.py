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

# ===== MODIFIKASI: Menambahkan URL khusus untuk setiap jenis halaman =====
TARGET_URLS = {
    # Pasaran di Halaman Utama
    'hongkongpools': 'https://angkanet.tv/',
    'hongkong': 'https://angkanet.tv/',
    'sydneypools': 'https://angkanet.tv/',
    'sydney': 'https://angkanet.tv/',
    'singapore': 'https://angkanet.tv/',
    'bullseye': 'https://angkanet.tv/',
    
    # Pasaran di Halaman "Rumus Harian" (URL ini adalah contoh, mungkin perlu penyesuaian)
    'moroccoquatro21': 'https://angkanet.tv/rumus-lengkap/?pasaran=morocco-quatro-21-00-wib',
    'moroccoquatro18': 'https://angkanet.tv/rumus-lengkap/?pasaran=morocco-quatro-18-00-wib',
    'moroccoquatro00': 'https://angkanet.tv/rumus-lengkap/?pasaran=morocco-quatro-00-00-wib',
}

# Kamus untuk mencocokkan nama pasaran kita dengan nama yang ada di tabel Halaman Utama Angkanet
ANGKANET_MAIN_PAGE_NAMES = {
    'hongkongpools': 'Hongkong Pools',
    'hongkong': 'Hongkong Pools',
    'sydneypools': 'Sydney Pools',
    'sydney': 'Sydney Pools',
    'singapore': 'Singapore',
    'bullseye': 'Bullseye',
}

# Header browser standar
WEB_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def _scrape_main_page_table(soup, market_name):
    """Fungsi untuk mengambil data dari tabel di halaman utama Angkanet."""
    table = soup.find('table')
    if not table: return None
    rows = table.find('tbody').find_all('tr')
    for row in rows:
        cells = row.find_all('td')
        if len(cells) > 2 and market_name.lower() in cells[0].text.strip().lower():
            result = ''.join(filter(str.isdigit, cells[2].text))
            return result if len(result) == 4 and result.isdigit() else None
    return None

def _scrape_rumus_page(soup):
    """Fungsi untuk mengambil data dari halaman 'Rumus Harian' Angkanet."""
    # Mencari tabel hasil, biasanya yang pertama di halaman rumus
    result_table = soup.find('table', class_='table-hover')
    if not result_table: return None
    # Hasil terbaru biasanya ada di baris pertama dari body tabel
    first_row = result_table.find('tbody').find('tr')
    if not first_row: return None
    # Di halaman rumus, hasil ada di kolom kedua (index 1)
    cells = first_row.find_all('td')
    if len(cells) > 1:
        result = cells[1].text.strip()
        return result if len(result) == 4 and result.isdigit() else None
    return None

def get_latest_result(pasaran):
    """Mengambil hasil terbaru dengan metode web scraping yang sesuai."""
    pasaran_lower = pasaran.lower()
    if pasaran_lower not in TARGET_URLS:
        print(f"Tidak ada URL target untuk pasaran: {pasaran}. Dilewati.")
        return None

    url = TARGET_URLS[pasaran_lower]
    print(f"Mencoba mengambil data dari URL: {url}")
    
    try:
        response = requests.get(url, headers=WEB_HEADERS, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        result = None
        # Memutuskan fungsi mana yang akan digunakan berdasarkan pasaran
        if pasaran_lower in ANGKANET_MAIN_PAGE_NAMES:
            market_name_to_find = ANGKANET_MAIN_PAGE_NAMES[pasaran_lower]
            result = _scrape_main_page_table(soup, market_name_to_find)
        else: # Asumsikan ini adalah halaman rumus (untuk Maroko)
            result = _scrape_rumus_page(soup)

        if result:
            print(f"Sukses mendapatkan hasil untuk {pasaran}: {result}")
            return result
        else:
            print(f"Gagal menemukan hasil yang valid untuk {pasaran} di halaman tersebut.")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Error saat mengakses URL untuk {pasaran}: {e}")
    except Exception as e:
        print(f"Error saat memproses halaman web untuk {pasaran}: {e}")
        
    return None

# V V V V V (TIDAK ADA PERUBAHAN PADA FUNGSI DI BAWAH INI) V V V V V
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
