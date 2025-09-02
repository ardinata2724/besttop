def get_latest_result(pasaran):
    if driver is None: return None
    pasaran_lower = pasaran.lower()
    if pasaran_lower not in ANGKANET_DROPDOWN_VALUES: return None

    try:
        print(f"Mengunjungi URL: {ANGKANET_URL}")
        driver.get(ANGKANET_URL)
        wait = WebDriverWait(driver, 60)
        
        try:
            cookie_button = WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.XPATH, "//*[contains(text(), 'Setuju') or contains(text(), 'Accept')]")))
            cookie_button.click()
            print("Cookie banner ditemukan dan ditutup.")
        except TimeoutException:
            print("Tidak ada cookie banner yang ditemukan, melanjutkan proses.")

        dropdown_value = ANGKANET_DROPDOWN_VALUES[pasaran_lower]
        print(f"Memilih '{dropdown_value}' dari dropdown menggunakan JavaScript...")

        # [PERBAIKAN UTAMA] Menggunakan JavaScript untuk memilih nilai dropdown
        # Ini lebih andal daripada metode Select() jika ada JavaScript yang kompleks di halaman
        try:
            select_element = wait.until(EC.presence_of_element_located((By.NAME, "pasaran")))
            driver.execute_script(f"arguments[0].value = '{dropdown_value}'; arguments[0].dispatchEvent(new Event('change'));", select_element)
            print("Dropdown berhasil dipilih via JavaScript.")
        except TimeoutException:
            print("Gagal menemukan elemen dropdown 'pasaran' bahkan setelah menunggu.")
            raise # Lemparkan kembali error untuk ditangkap oleh blok except utama

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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_file = f"error_screenshot_{pasaran}_{timestamp}.png"
        html_file = f"error_page_source_{pasaran}_{timestamp}.html"
        
        # Simpan file debug
        driver.save_screenshot(screenshot_file)
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(driver.page_source)
        
        print(f"DEBUG: Screenshot disimpan sebagai '{screenshot_file}'")
        print(f"DEBUG: Kode sumber halaman disimpan sebagai '{html_file}'")
    return None
