import os
import time
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
    'moroccoquatro 18:00': 'keluaran morocco quatro 18:00.txt',
    'moroccoquatro 21:00': 'keluaran morocco quatro 21:00.txt',
    'moroccoquatro 23:59': 'keluaran morocco quatro 23:59.txt',
}

# Kamus untuk mencocokkan nama pasaran kita dengan nama di website
ANGKANET_MARKET_NAMES = {
    'hongkongpools': 'Hongkong Pools',
    'hongkong': 'Hongkong Lotto',
    'sydneypools': 'Sydneypools',
    'sydney': 'Sydney Lotto',
    'singapore': 'SGP | Singapore',
    'bullseye': 'Bullseye',
    'moroccoquatro 18:00': 'Morocco Quatro 18:00 Wib',
    'moroccoquatro 21:00': 'Morocco Quatro 21:00 Wib',
    'moroccoquatro 23:59': 'Morocco Quatro 23:59 Wib',
}

# --- PETA URL UNTUK SETIAP PASARAN ---
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

def setup_driver():
    global driver
    if driver is None:
        try:
            print("Menyiapkan driver Selenium...")
            options = webdriver.ChromeOptions()
            options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--window-size=1920,1080")
            driver = webdriver.Chrome(options=options)
            print("Driver Selenium siap.")
        except Exception as e:
            print(f"Error saat menyiapkan driver: {e}")

def _scrape_main_page(soup, market_name):
    """Logika untuk membaca tabel di halaman utama."""
    table = soup.find('table', id='myTable')
    if not table: return None
    rows = table.find('tbody').find_all('tr')
    for row in rows:
        cells = row.find_all('td')
        if len(cells) > 2:
            button = cells[0].find('button')
            if button and market_name.lower() in button.text.strip().lower():
                span_tags = cells[2].find_all('span', class_='rescir')
                result_str = "".join([span.text.strip() for span in span_tags])
                if len(result_str) == 4 and result_str.isdigit():
                    print(f"Sukses mendapatkan hasil (halaman utama) untuk {market_name}: {result_str}")
                    return result_str
    return None

def _scrape_rumus_page(soup):
    """Logika untuk membaca tabel di halaman Rumus Harian."""
    table = soup.find('table', class_='table-hover')
    if not table: return None
    first_row = table.find('tbody').find('tr')
    if not first_row: return None
    cells = first_row.find_all('td')
    if len(cells) > 1:
        result = cells[1].text.strip()
        if len(result) == 4 and result.isdigit():
            print(f"Sukses mendapatkan hasil (halaman rumus): {result}")
            return result
    return None

def get_latest_result(pasaran):
    if driver is None: return None
    pasaran_lower = pasaran.lower()
    if pasaran_lower not in TARGET_URLS:
        print(f"Pasaran '{pasaran}' tidak memiliki URL target. Dilewati.")
        return None

    target_url = TARGET_URLS[pasaran_lower]
    market_name_to_find = ANGKANET_MARKET_NAMES.get(pasaran_lower)
    
    try:
        print(f"Mengunjungi URL: {target_url}")
        driver.get(target_url)
        wait = WebDriverWait(driver, 30)
        
        # Menentukan logika mana yang akan dipakai berdasarkan URL
        if "/rumus-lengkap/" in target_url:
            wait.until(EC.presence_of_element_located((By.CLASS_NAME, "table-hover")))
            print("Halaman Rumus terdeteksi. Membaca tabel hasil...")
            return _scrape_rumus_page(BeautifulSoup(driver.page_source, 'html.parser'))
        else:
            wait.until(EC.presence_of_element_located((By.ID, "myTable")))
            print("Halaman Utama terdeteksi. Mencari pasaran di tabel...")
            return _scrape_main_page(BeautifulSoup(driver.page_source, 'html.parser'), market_name_to_find)

    except Exception as e:
        print(f"Terjadi error tak terduga saat memproses {pasaran}: {e}")
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
