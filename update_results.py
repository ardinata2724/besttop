import os
import time
from bs4 import BeautifulSoup
from datetime import datetime, timezone, timedelta
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# --- KONFIGURASI ---
PASARAN_FILES = {
    'hongkongpools': 'keluaran hongkongpools.txt', 'hongkong': 'keluaran hongkong lotto.txt',
    'sydneypools': 'keluaran sydneypools.txt', 'sydney': 'keluaran sydney lotto.txt',
    'singapore': 'keluaran singapura.txt', 'bullseye': 'keluaran bullseye.txt',
    'moroccoquatro18': 'keluaran morocco quatro 18:00 wib.txt',
    'moroccoquatro21': 'keluaran morocco quatro 21:00 wib.txt',
    'moroccoquatro23.59': 'keluaran morocco quatro 23:59 wib.txt',
}
ANGKANET_URL = "http://159.223.64.48/rumus-lengkap/"
ANGKANET_DROPDOWN_VALUES = {
    'hongkongpools': 'hongkong-pools', 'hongkong': 'hongkong-pools',
    'sydneypools': 'sydney-pools', 'sydney': 'sydney-pools',
    'singapore': 'singapore-pools', 'bullseye': 'bullseye',
    'moroccoquatro21': 'morocco-quatro-21-00-wib',
    'moroccoquatro18': 'morocco-quatro-18-00-wib',
    'moroccoquatro23.59': 'morocco-quatro-00-00-wib',
}

driver = None

def setup_driver():
    global driver
    if driver is None:
        try:
            print("Menyiapkan driver undetected-chromedriver...")
            options = uc.ChromeOptions()
            options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            driver = uc.Chrome(options=options)
            print("Driver undetected-chromedriver siap.")
        except Exception as e:
            print(f"Error saat menyiapkan driver: {e}")

def get_latest_result(pasaran):
    if driver is None: return None
    pasaran_lower = pasaran.lower()
    if pasaran_lower not in ANGKANET_DROPDOWN_VALUES: return None

    try:
        print(f"Mengunjungi URL: {ANGKANET_URL}")
        driver.get(ANGKANET_URL)
        wait = WebDriverWait(driver, 30)
        
        print("Memberi jeda 5 detik agar halaman stabil...")
        time.sleep(5)
        
        dropdown_value = ANGKANET_DROPDOWN_VALUES[pasaran_lower]
        print(f"Memilih '{dropdown_value}' dari dropdown...")
        select_element = wait.until(EC.visibility_of_element_located((By.NAME, "pasaran")))
        Select(select_element).select_by_value(dropdown_value)
        print("Dropdown berhasil dipilih.")
        
        go_button = wait.until(EC.element_to_be_clickable((By.NAME, "patah")))
        driver.execute_script("arguments[0].click();", go_button)
        print("Tombol 'Go' berhasil ditekan.")

        print("Menunggu tabel hasil...")
        result_table = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "table-hover")))
        
        html = result_table.get_attribute('outerHTML')
        soup = BeautifulSoup(html, 'html.parser')
        
        first_row = soup.find('tbody').find('tr')
        if not first_row: return None

        cells = first_row.find_all('td')
        if len(cells) > 1:
            result = cells[1].text.strip()
            if len(result) == 4 and result.isdigit():
                print(f"Sukses mendapatkan hasil untuk {pasaran}: {result}")
                return result
        return None
    except Exception as e:
        print(f"Terjadi error saat memproses {pasaran}: {e}")
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
    wib = timezone(timedelta(hours=7)); print(f"--- Memulai proses pembaruan pada {datetime.now(wib).strftime('%Y-%m-%d %H:%M:%S WIB')} ---")
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
        print("Tidak ada file yang diperbarui. Keluar."); exit(0)

if __name__ == "__main__":
    main()
