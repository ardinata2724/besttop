import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# --- KONFIGURASI ---
ANGKANET_URL = "http://159.223.64.48/"

def capture_html():
    """Menyiapkan driver, mengunjungi URL, dan menyimpan sumber halaman HTML."""
    driver = None
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
        
        # Menunggu elemen tabel (tbody) muncul, maksimal 30 detik
        wait = WebDriverWait(driver, 30)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "tbody")))
        print("Elemen tabel terdeteksi. Menunggu 5 detik tambahan agar semua data ter-load...")
        time.sleep(5) # Memberi waktu ekstra agar semua javascript selesai
        
        # Menyimpan isi halaman ke file
        html_content = driver.page_source
        with open("debug_page_source.html", "w", encoding="utf-8") as f:
            f.write(html_content)
        
        print("Sukses! Isi halaman web telah disimpan ke file 'debug_page_source.html'.")

    except Exception as e:
        print(f"Terjadi error saat mencoba menyimpan halaman: {e}")
        # Jika ada error, tetap coba simpan apa yang sudah ada
        if driver:
            html_content = driver.page_source
            with open("debug_page_source.html", "w", encoding="utf-8") as f:
                f.write(f"<h1>Error Terjadi, namun ini isi HTML saat error:</h1>\n<p>{e}</p>\n{html_content}")
            print("Berhasil menyimpan sebagian HTML saat terjadi error.")
            
    finally:
        if driver:
            driver.quit()
            print("Driver Selenium ditutup.")

if __name__ == "__main__":
    capture_html()
