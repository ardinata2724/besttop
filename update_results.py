import os
from datetime import datetime, timezone, timedelta
from requests_html import HTMLSession

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

ANGKANET_URL = "http://159.223.64.48/"
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

session = None
page_html = None

def get_latest_result(pasaran):
    """Mengambil hasil terbaru menggunakan requests-html yang bisa menjalankan JavaScript."""
    global session, page_html
    
    pasaran_lower = pasaran.lower()
    if pasaran_lower not in ANGKANET_MARKET_NAMES:
        print(f"Pasaran '{pasaran}' tidak dikonfigurasi. Dilewati.")
        return None

    market_name_to_find = ANGKANET_MARKET_NAMES[pasaran_lower]
    
    try:
        # Hanya fetching halaman jika belum ada di cache
        if page_html is None:
            if session is None:
                session = HTMLSession()
            
            print(f"Mengunjungi dan me-render JavaScript di URL: {ANGKANET_URL}")
            r = session.get(ANGKANET_URL, timeout=30)
            
            # Menunggu JavaScript untuk memuat konten, tunggu maksimal 20 detik
            r.html.render(sleep=15, timeout=40)
            page_html = r.html
            print("Render JavaScript selesai.")

        # Mencari semua baris tabel
        rows = page_html.find('tr')
        print(f"Mencari '{market_name_to_find}' diantara {len(rows)} baris...")

        for row in rows:
            # Mencari nama market di dalam teks baris
            if market_name_to_find.lower() in row.text.lower():
                # Menemukan sel-sel (kolom) di dalam baris yang cocok
                cells = row.find('td')
                if len(cells) > 2:
                    # Mengambil isi dari sel ketiga (indeks 2)
                    result_cell_html = cells[2].html
                    # Mengambil angka dari gambar (misal: <img src="/.../N4.gif">)
                    result_numbers = re.findall(r'N(\d)\.gif', result_cell_html)
                    result_str = "".join(result_numbers)

                    if len(result_str) == 4 and result_str.isdigit():
                        print(f"Sukses mendapatkan hasil untuk {market_name_to_find}: {result_str}")
                        return result_str

        print(f"Tidak dapat menemukan baris yang cocok untuk '{market_name_to_find}'.")
        return None

    except Exception as e:
        print(f"Terjadi error: {e}")
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
    import re
    wib = timezone(timedelta(hours=7))
    print(f"--- Memulai proses pembaruan pada {datetime.now(wib).strftime('%Y-%m-%d %H:%M:%S WIB')} ---")
    
    any_file_updated = False
    for pasaran, filename in PASARAN_FILES.items():
        print(f"\nMemproses pasaran: {pasaran.capitalize()}")
        latest_result = get_latest_result(pasaran)
        
        if latest_result:
            if update_file(filename, latest_result):
                any_file_updated = True
        else:
            print(f"Tidak dapat mengambil hasil terbaru untuk {pasaran}.")
            
    print("\n--- Proses pembaruan selesai. ---")
    if not any_file_updated:
        print("Tidak ada file yang diperbarui. Keluar.")
        exit(0)

if __name__ == "__main__":
    main()
