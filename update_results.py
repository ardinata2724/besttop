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

# ===== PERBAIKAN FINAL: Menggunakan ALAMAT IP YANG BENAR =====
ANGKANET_BASE_URL = "http://159.223.64.48/"

# Kamus untuk membuat URL halaman "Rumus Lengkap" untuk setiap pasaran
ANGKANET_RUMUS_PARAMS = {
    'hongkongpools': 'rumus-lengkap/?pasaran=hongkong-pools',
    'hongkong': 'rumus-lengkap/?pasaran=hongkong-pools',
    'sydneypools': 'rumus-lengkap/?pasaran=sydney-pools',
    'sydney': 'rumus-lengkap/?pasaran=sydney-pools',
    'singapore': 'rumus-lengkap/?pasaran=singapore-pools',
    'bullseye': 'rumus-lengkap/?pasaran=bullseye',
    'moroccoquatro21': 'rumus-lengkap/?pasaran=morocco-quatro-21-00-wib',
    'moroccoquatro18': 'rumus-lengkap/?pasaran=morocco-quatro-18-00-wib',
    'moroccoquatro00': 'rumus-lengkap/?pasaran=morocco-quatro-00-00-wib',
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
    """Mengambil hasil terbaru dari halaman Rumus Lengkap Angkanet."""
    if driver is None: return None
    pasaran_lower = pasaran.lower()
    if pasaran_lower not in ANGKANET_RUMUS_PARAMS:
        print(f"Pasaran '{pasaran}' tidak dikonfigurasi. Dilewati.")
        return None

    # Membuat URL lengkap untuk halaman rumus pasaran yang dituju
    target_url = ANGKANET_BASE_URL + ANGKANET_RUMUS_PARAMS[pasaran_lower]
    
    try:
        print(f"Mengunjungi URL: {target_url}")
        driver.get(target_url)
        
        # Menunggu tabel hasil (dengan class table-hover) muncul, maksimal 30 detik
        wait = WebDriverWait(driver, 30)
        result_table = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "table-hover")))
        
        # Mengambil HTML setelah JavaScript berjalan
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        
        # Menemukan hasil di baris pertama tabel
        first_row = soup.find('table', class_='table-hover').find('tbody').find('tr')
        if not first_row:
            print("Tidak bisa menemukan baris pertama di tabel hasil.")
            return None

        # Di halaman rumus, hasil ada di kolom kedua (index 1)
        cells = first_row.find_all('td')
        if len(cells) > 1:
            result = cells[1].text.strip()
            if len(result) == 4 and result.isdigit():
                print(f"Sukses mendapatkan hasil untuk {pasaran}: {result}")
                return result
        
        print(f"Tidak dapat menemukan format hasil yang benar di baris pertama.")
        return None

    except TimeoutException:
        print(f"Gagal menemukan tabel hasil dalam 30 detik. Halaman mungkin lambat atau berubah.")
    except Exception as e:
        print(f"Terjadi error tak terduga saat proses Selenium: {e}")
    
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
