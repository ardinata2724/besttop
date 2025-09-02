import os
import time
from bs4 import BeautifulSoup
from datetime import datetime, timezone, timedelta
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium_stealth import stealth

# --- KONFIGURASI (Sudah Benar) ---
NEW_URL = "https://server.scanangka.fun/keluaranharian"
PASARAN_FILES = {
    'hongkongpools': 'keluaran hongkongpools.txt', 'hongkong': 'keluaran hongkong lotto.txt',
    'sydneypools': 'keluaran sydneypools.txt', 'sydney': 'keluaran sydney lotto.txt',
    'singapore': 'keluaran singapura.txt', 'bullseye': 'keluaran bullseye.txt',
}
NEW_DROPDOWN_VALUES = {
    'hongkongpools': 'HONGKONG', 'hongkong': 'HONGKONG', 'sydneypools': 'SYDNEY',
    'sydney': 'SYDNEY', 'singapore': 'SINGAPORE', 'bullseye': 'BULLSEYE',
}

driver = None

def setup_driver():
    global driver
    if driver is None:
        try:
            print("Menyiapkan driver undetected-chromedriver dengan opsi tambahan...")
            options = uc.ChromeOptions()
            options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            options.add_argument("--window-size=1920,1200")
            options.add_argument("--ignore-certificate-errors")
            options.add_argument("--disable-extensions")
            options.add_argument("--start-maximized")
            options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36")
            driver = uc.Chrome(options=options)
            print("Mengaktifkan mode stealth untuk melewati Cloudflare...")
            stealth(driver, languages=["en-US", "en"], vendor="Google Inc.", platform="Win32", webgl_vendor="Intel Inc.", renderer="Intel Iris OpenGL Engine", fix_hairline=True)
            print("Mode stealth aktif.")
        except Exception as e:
            print(f"Error saat menyiapkan driver: {e}")

def get_latest_result(pasaran):
    if driver is None: return None
    pasaran_lower = pasaran.lower()
    if pasaran_lower not in NEW_DROPDOWN_VALUES:
        print(f"Pasaran '{pasaran}' tidak ada di dalam mapping.")
        return None

    try:
        print(f"Mengunjungi URL: {NEW_URL}")
        driver.get(NEW_URL)
        wait = WebDriverWait(driver, 60)

        print("Memberi jeda 10 detik agar halaman dimuat sepenuhnya...")
        time.sleep(10)

        print("Mencari dan membuka dropdown pasaran...")
        dropdown_box = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "span.select2-selection__rendered")))
        dropdown_box.click()
        
        pasaran_target_text = NEW_DROPDOWN_VALUES[pasaran_lower]
        print(f"Mencari dan mengklik opsi '{pasaran_target_text}' dari daftar...")
        pasaran_option = wait.until(EC.element_to_be_clickable((By.XPATH, f"//li[text()='{pasaran_target_text}']")))
        pasaran_option.click()

        print("Beralih ke dalam frame data...")
        wait.until(EC.frame_to_be_available_and_switch_to_it((By.TAG_NAME, "iframe")))
        
        # [PENYEMPURNAAN FINAL] Tambahkan jeda singkat di sini
        print("Memberi jeda 3 detik agar tabel di dalam frame stabil...")
        time.sleep(3)

        print("Menunggu tabel hasil untuk dimuat...")
        result_table = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table.table.table-bordered")))
        
        print("Mengambil angka keluaran dari tabel...")
        first_row_result = result_table.find_element(By.CSS_SELECTOR, "tbody tr:first-child td:nth-child(2)")
        result = first_row_result.text.strip()
        
        driver.switch_to.default_content()
        
        if len(result) == 4 and result.isdigit():
            print(f"Sukses mendapatkan hasil untuk {pasaran}: {result}")
            return result
        else:
            print(f"Format hasil tidak valid untuk {pasaran}: '{result}'")
            return None

    except Exception as e:
        print(f"Terjadi error saat memproses {pasaran}: {e}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_file = f"error_screenshot_{pasaran}_{timestamp}.png"
        driver.save_screenshot(screenshot_file)
        print(f"DEBUG: Screenshot disimpan sebagai '{screenshot_file}'")
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
        print(f"HASIL BARU DITAMBAHKAN: {new_result} -> {filename}")
        return True
    else:
        print(f"Hasil {new_result} sudah ada di {filename}. Tidak ada perubahan.")
        return False

def main():
    wib = timezone(timedelta(hours=7))
    print(f"--- Memulai proses pembaruan pada {datetime.now(wib).strftime('%Y-%m-%d %H:%M%S WIB')} ---")
    setup_driver()
    any_file_updated = False
    if driver:
        for pasaran, filename in PASARAN_FILES.items():
            print(f"\nMemproses pasaran: {pasaran.capitalize()}")
            latest_result = get_latest_result(pasaran)
            if latest_result:
                if update_file(filename, latest_result): 
                    any_file_updated = True
            else: 
                print(f"Tidak dapat mengambil hasil terbaru untuk {pasaran}.")
        driver.quit()
        
    print("\n--- Proses pembaruan selesai. ---")
    if not any_file_updated:
        print("PERINGATAN: Tidak ada satu pun file yang diperbarui. Proses akan ditandai sebagai gagal.")
        exit(1)
    else:
        print("Pembaruan berhasil. Setidaknya satu file telah diubah.")

if __name__ == "__main__":
    main()
