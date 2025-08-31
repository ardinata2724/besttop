import requests
import os
import time
from bs4 import BeautifulSoup
from datetime import datetime, timezone, timedelta
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

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

# ===== PERBAIKAN FINAL: Menggunakan ALAMAT WEBSITE YANG BENAR =====
ANGKANET_URL = "https://angkanet.top/"

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

driver = None

def setup_driver():
    """Menyiapkan driver browser Selenium."""
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
            driver = None

def get_latest_result(pasaran):
    """Mengambil hasil terbaru menggunakan Selenium."""
    if driver is None: return None
    pasaran_lower = pasaran.lower()
    if pasaran_lower not in ANGKANET_MARKET_NAMES: return None
    market_name_to_find = ANGKANET_MARKET_NAMES[pasaran_lower]
    
    try:
        if driver.current_url != ANGKANET_URL:
            print(f"Mengunjungi URL: {ANGKANET_URL}")
            driver.get(ANGKANET_URL)
        
        wait = WebDriverWait(driver, 30)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "tbody")))
        
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        
        rows = soup.find_all('tr')
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

    except TimeoutException:
        print(f"Gagal menemukan tabel dalam 30 detik. Mengambil screenshot...")
        driver.save_screenshot('debug_screenshot.png')
        print("Screenshot 'debug_screenshot.png' berhasil diambil.")
    except Exception as e:
        print(f"Terjadi error tak terduga saat proses Selenium: {e}")
        driver.save_screenshot('debug_screenshot.png')
        print("Screenshot 'debug_screenshot.png' berhasil diambil karena error.")
    return None

def update_file(filename, new_result):
    if not os.path.exists(filename):
        existing_results = set()
    else:
        with open(filename, 'r', encoding='utf-8') as f:
            existing_results = set(line.strip() for line in f if line.strip())
    if new_result not in existing_results:
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(f"\n{new_result}")
        return True
    else:
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
