import os
from bs4 import BeautifulSoup
from datetime import datetime, timezone, timedelta
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re

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

# Kamus untuk mencocokkan nama internal kita dengan nama lengkap di website
# ===== NAMA PASARAN MAROKO SUDAH DIPERBAIKI DI SINI =====
ANGKANET_MARKET_NAMES = {
    'hongkongpools': 'Hongkong Pools',
    'hongkong': 'Hongkong Lotto',
    'sydneypools': 'Sydneypools',
    'sydney': 'Sydney Lotto',
    'singapore': 'SGP | Singapore',
    'bullseye': 'Bullseye',
    'moroccoquatro21': 'Morocco Quatro 21:00 Wib',
    'moroccoquatro18': 'Morocco Quatro 18:00 Wib',
    'moroccoquatro00': 'Morocco Quatro 23:59 Wib', # Diperbaiki dari 00:00 menjadi 23:59
}

# --- PETA URL UNTUK SETIAP PASARAN (SEKARANG LENGKAP) ---
ANGKANET_BASE_URL = "http://159.223.64.48"
TARGET_URLS = {
    'hongkongpools': ANGKANET_BASE_URL + '/',
    'hongkong': ANGKANET_BASE_URL + '/',
    'sydneypools': ANGKANET_BASE_URL + '/',
    'sydney': ANGKANET_BASE_URL + '/',
    'singapore': ANGKANET_BASE_URL + '/',
    'bullseye': ANGKANET_BASE_URL + '/',
    'moroccoquatro21': ANGKANET_BASE_URL + '/rumus-lengkap/?pasaran=morocco-quatro-21-00-wib',
    'moroccoquatro18': ANGKANET_BASE_URL + '/rumus-lengkap/?pasaran=morocco-quatro-18-00-wib',
    'moroccoquatro00': ANGKANET_BASE_URL + '/rumus-lengkap/?pasaran=morocco-quatro-00-00-wib',
}

driver = None
page_soups = {} # Cache untuk menyimpan HTML dari setiap halaman yang dikunjungi

def setup_driver():
    global driver
    if driver is None:
        try:
            print("Menyiapkan driver Selenium...")
            options = webdriver.ChromeOptions()
            options.add_argument("--headless"); options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage"); options.add_argument("--window-size=1920,1080")
            driver = webdriver.Chrome(options=options)
            print("Driver Selenium siap.")
        except Exception as e:
            print(f"Error saat menyiapkan driver: {e}")

def get_page_soup(url):
    """Mengunjungi URL dan mengembalikan objek soup, menggunakan cache jika memungkinkan."""
    global page_soups
    if url in page_soups:
        print(f"Menggunakan HTML dari cache untuk {url}")
        return page_soups[url]
    
    if driver is None: return None
    try:
        print(f"Mengunjungi URL: {url}")
        driver.get(url)
        wait = WebDriverWait(driver, 30)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "table")))
        print("Halaman dan tabel berhasil dimuat.")
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        page_soups[url] = soup # Simpan ke cache
        return soup
    except Exception as e:
        print(f"Gagal memuat atau menemukan tabel di {url}. Error: {e}")
        return None

def parse_result_from_soup(soup, market_name_to_find):
    """Satu fungsi parsing yang andal untuk semua halaman."""
    try:
        tables = soup.find_all('table')
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all('td')
                if len(cells) > 2:
                    market_text = cells[0].text.strip()
                    if market_name_to_find.lower() in market_text.lower():
                        result_cell = cells[2]
                        # Coba baca dari gambar dulu
                        image_tags = result_cell.find_all('img')
                        if image_tags:
                            result_str = "".join(re.findall(r'N(\d)\.gif', img.get('src', '')))
                        else: # Jika tidak ada gambar, coba baca dari span
                            span_tags = result_cell.find_all('span', class_='rescir')
                            result_str = "".join([span.text.strip() for span in span_tags])
                        
                        if len(result_str) == 4 and result_str.isdigit():
                            print(f"Sukses mendapatkan hasil untuk {market_name_to_find}: {result_str}")
                            return result_str
        print(f"Tidak dapat menemukan baris yang cocok untuk '{market_name_to_find}'.")
    except Exception as e:
        print(f"Error saat parsing tabel: {e}")
    return None

def get_latest_result(pasaran):
    pasaran_lower = pasaran.lower()
    url = TARGET_URLS.get(pasaran_lower)
    market_name = ANGKANET_MARKET_NAMES.get(pasaran_lower)
    if not url or not market_name:
        print(f"Konfigurasi untuk '{pasaran}' tidak ditemukan. Dilewati.")
        return None
        
    soup = get_page_soup(url)
    if soup:
        return parse_result_from_soup(soup, market_name)
    return None

def update_file(filename, new_result):
    if not os.path.exists(filename): existing_results = set()
    else:
        with open(filename, 'r', encoding='utf-8') as f:
            existing_results = set(line.strip() for line in f if line.strip())
    if new_result not in existing_results:
        with open(filename, 'a', encoding='utf-8') as f: f.write(f"\n{new_result}")
        print(f"HASIL BARU DITAMBAHKAN: {new_result} -> {filename}")
        return True
    else:
        print(f"Hasil {new_result} sudah ada di {filename}. Tidak ada perubahan.")
        return False

def main():
    wib = timezone(timedelta(hours=7))
    print(f"--- Memulai proses pembaruan pada {datetime.now(wib).strftime('%Y-%m-%d %H:%M:%S WIB')} ---")
    setup_driver()
    any_file_updated = False
    if driver:
        for pasaran, filename in PASARAN_FILES.items():
            print(f"\nMemproses pasaran: {pasaran.capitalize()}")
            latest_result = get_latest_result(pasaran)
            if latest_result:
                if update_file(filename, latest_result): any_file_updated = True
            else: print(f"Tidak dapat mengambil hasil terbaru untuk {pasaran}.")
        driver.quit()
    print("\n--- Proses pembaruan selesai. ---")
    if not any_file_updated:
        print("Tidak ada file yang diperbarui. Keluar.")
        exit(0)

if __name__ == "__main__":
    main()
