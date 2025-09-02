import os
import time
from bs4 import BeautifulSoup
from datetime import datetime, timezone, timedelta
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# --- KONFIGURASI FINAL ---
PASARAN_MAPPING = {
    'keluaran hongkongpools.txt': 'hongkong',
    'keluaran hongkong lotto.txt': 'hongkong',
    'keluaran sydneypools.txt': 'sydney',
    'keluaran sydney lotto.txt': 'sydney',
    'keluaran singapura.txt': 'sgp',
    'keluaran bullseye.txt': 'bullseye',
}

DATA_URL = "https://togelmaster.org/paito/"
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
            options.add_argument("--disable-gpu")
            options.add_argument("--window-size=1920,1200")
            options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36")
            driver = uc.Chrome(options=options)
            print("Driver siap.")
        except Exception as e:
            print(f"Error saat menyiapkan driver: {e}")

def get_latest_result(pasaran_path):
    if driver is None:
        return None
    try:
        target_url = DATA_URL + pasaran_path
        print(f"Mengunjungi URL: {target_url}")
        driver.get(target_url)
        
        wait = WebDriverWait(driver, 30)
        
        # Tunggu sampai tabel datanya muncul (dibuat oleh JavaScript)
        print("Menunggu tabel data untuk dimuat oleh JavaScript...")
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table.wla-daily-datatable")))
        
        print("Tabel ditemukan, mengambil kode HTML...")
        html_content = driver.page_source
        soup = BeautifulSoup(html_content, 'html.parser')
        
        first_row = soup.select_one("table.wla-daily-datatable tbody tr:first-child")
        
        if not first_row:
            print("Gagal menemukan baris pertama di dalam tabel hasil.")
            return None

        result_cell = first_row.select_one("td:nth-child(2)")
        
        if result_cell:
            result = result_cell.get_text(strip=True)
            if len(result) >= 4 and result.isdigit():
                result = result[-4:]
                print(f"Sukses mendapatkan hasil untuk {pasaran_path.upper()}: {result}")
                return result
        
        print("Tidak ditemukan sel hasil yang valid.")
        return None

    except Exception as e:
        print(f"Terjadi error saat memproses {pasaran_path.upper()}: {e}")
        return None

def update_file(filename, new_result):
    if not os.path.exists(filename): existing_results = set()
    else:
        with open(filename, 'r', encoding='utf-8') as f:
            existing_results = set(line.strip() for line in f if line.strip())
            
    if new_result and new_result not in existing_results:
        with open(filename, 'a', encoding='utf-8') as f: f.write(f"\n{new_result}")
        print(f"HASIL BARU DITAMBAHKAN: {new_result} -> {filename}")
        return True
    elif new_result:
        print(f"Hasil {new_result} sudah ada di {filename}. Tidak ada perubahan.")
    return False

def main():
    wib = timezone(timedelta(hours=7))
    print(f"--- Memulai proses pembaruan pada {datetime.now(wib).strftime('%Y-%m-%d %H:%M:%S WIB')} ---")
    
    setup_driver()
    any_file_updated = False

    if driver:
        for filename, pasaran_path in PASARAN_MAPPING.items():
            print(f"\nMemproses file: {filename}")
            latest_result = get_latest_result(pasaran_path)
            if latest_result:
                if update_file(filename, latest_result): 
                    any_file_updated = True
            else: 
                print(f"Tidak dapat mengambil hasil terbaru untuk {pasaran_path.upper()}.")
        driver.quit()
    
    print("\n--- Proses pembaruan selesai. ---")
    if not any_file_updated:
        print("PERINGATAN: Tidak ada satu pun file yang diperbarui. Proses akan ditandai sebagai gagal.")
        exit(1)
    else:
        print("Pembaruan berhasil. Setidaknya satu file telah diubah.")

if __name__ == "__main__":
    main()
