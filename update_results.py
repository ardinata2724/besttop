import os
from bs4 import BeautifulSoup
from datetime import datetime, timezone, timedelta
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# --- KONFIGURASI ---
PASARAN_FILES = {
    'hongkongpools': 'keluaran hongkongpools.txt',
    'hongkong lotto': 'keluaran hongkong lotto.txt',
    'sydneypools': 'keluaran sydneypools.txt',
    'sydney lotto': 'keluaran sydney lotto.txt',
    'singapore': 'keluaran singapura.txt',
    'bullseye': 'keluaran bullseye.txt',
    'moroccoquatro18': 'keluaran morocco quatro 18:00.txt',
    'moroccoquatro21': 'keluaran morocco quatro 21:00.txt',
    'moroccoquatro00': 'keluaran morocco quatro 23:59.txt',
}

ANGKANET_URL = "http://159.223.64.48/"
ANGKANET_MARKET_NAMES = {
    'hongkongpools': 'Hongkong Pools',
    'hongkong lotto': 'Hongkong Lotto',
    'sydneypools': 'Sydneypools',
    'sydney lotto': 'Sydney Lotto',
    'singapore': 'SGP | Singapore',
    'bullseye': 'Bullseye',
    'moroccoquatro 21:00': 'Morocco Quatro 21:00 Wib',
    'moroccoquatro 18:00': 'Morocco Quatro 18:00 Wib',
    'moroccoquatro 23:59': 'Morocco Quatro 23:59 Wib',
}

driver = None
page_soup = None

def setup_driver_and_get_soup():
    """Menyiapkan driver, mengunjungi URL, dan mengembalikan objek soup."""
    global driver, page_soup
    if driver is None:
        try:
            print("Menyiapkan driver Selenium...")
            options = webdriver.ChromeOptions()
            options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--window-size=1920,1080")
            driver = webdriver.Chrome(options=options)
            print(f"Driver Selenium siap. Mengunjungi {ANGKANET_URL}...")
            
            driver.get(ANGKANET_URL)
            wait = WebDriverWait(driver, 30)
            wait.until(EC.presence_of_element_located((By.ID, "myTable")))
            print("Tabel utama ditemukan.")
            
            html = driver.page_source
            page_soup = BeautifulSoup(html, 'html.parser')

        except Exception as e:
            print(f"Error fatal saat menyiapkan Selenium atau mengambil halaman: {e}")
            if driver:
                driver.quit()
            driver = None

def get_latest_result(pasaran):
    """Mengambil hasil terbaru dari HTML yang sudah diambil."""
    if page_soup is None: return None
    
    pasaran_lower = pasaran.lower()
    if pasaran_lower not in ANGKANET_MARKET_NAMES: return None
    market_name_to_find = ANGKANET_MARKET_NAMES[pasaran_lower]
    
    try:
        table = page_soup.find('table', id='myTable')
        rows = table.find('tbody').find_all('tr')

        for row in rows:
            cells = row.find_all('td')
            if len(cells) > 2:
                # Mengambil teks dari tombol dropdown di sel pertama
                button = cells[0].find('button')
                if button:
                    current_market_name = button.text.strip()
                    if market_name_to_find.lower() in current_market_name.lower():
                        result_cell = cells[2]
                        span_tags = result_cell.find_all('span', class_='rescir')
                        result_str = "".join([span.text.strip() for span in span_tags])
                        
                        if len(result_str) == 4 and result_str.isdigit():
                            print(f"Sukses mendapatkan hasil untuk {market_name_to_find}: {result_str}")
                            return result_str
        
        print(f"Tidak dapat menemukan baris yang cocok untuk '{market_name_to_find}'.")
        return None

    except Exception as e:
        print(f"Terjadi error saat parsing tabel: {e}")
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
    
    setup_driver_and_get_soup()
    
    any_file_updated = False
    if page_soup and driver:
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
