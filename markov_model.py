import random
from collections import defaultdict, Counter
import pandas as pd

def _ensure_unique_top_n(top_list, n=6):
    """Memastikan daftar berisi N digit unik, dengan menambahkan angka acak jika perlu."""
    # Mengambil N elemen pertama dan memastikan keunikannya
    unique_list = list(dict.fromkeys(top_list))[:n]
    
    if len(unique_list) >= n:
        return unique_list

    # Jika kurang dari N, tambahkan angka unik secara acak
    all_digits = list(range(10))
    random.shuffle(all_digits)
    
    unique_set = set(unique_list)
    
    for digit in all_digits:
        if len(unique_set) >= n:
            break
        if digit not in unique_set:
            unique_set.add(digit)
            
    return list(unique_set)

# MARKOV ORDER-1
def build_transition_matrix(data):
    matrix = [defaultdict(lambda: defaultdict(int)) for _ in range(3)]
    for number in data:
        digits = f"{int(number):04d}"
        for i in range(3):
            matrix[i][digits[i]][digits[i+1]] += 1
    return matrix

def top6_markov(df, top_n=6):
    data = df["angka"].astype(str).tolist()
    matrix = build_transition_matrix(data)

    freq_ribuan = Counter([int(x[0]) for x in data])
    transisi = [{k: dict(v) for k, v in matrix[i].items()} for i in range(3)]
    kombinasi = Counter(data).most_common(10)

    hasil = []

    # Ribuan (digit ke-1)
    top_pos1 = [k for k, _ in freq_ribuan.most_common(top_n)]
    hasil.append(_ensure_unique_top_n(top_pos1, n=top_n))

    # Ratusan, Puluhan, Satuan
    for i in range(3):
        kandidat = []
        for prev in matrix[i]:
            kandidat.extend(matrix[i][prev].keys())
        kandidat_sorted = Counter(kandidat).most_common()
        top = [int(k) for k, _ in kandidat_sorted]
        hasil.append(_ensure_unique_top_n(top, n=top_n))

    info = {
        "frekuensi_ribuan": dict(freq_ribuan),
        "transisi": transisi,
        "kombinasi_populer": kombinasi
    }

    return [hasil[0], hasil[1], hasil[2], hasil[3]], info

# MARKOV ORDER-2
def build_transition_matrix_order2(data):
    # (Fungsi ini tidak perlu diubah)
    matrix = [{} for _ in range(2)]
    for number in data:
        digits = f"{int(number):04d}"
        key1 = digits[0] + digits[1]
        key2 = digits[1] + digits[2]
        if key1 not in matrix[0]:
            matrix[0][key1] = defaultdict(int)
        if key2 not in matrix[1]:
            matrix[1][key2] = defaultdict(int)
        matrix[0][key1][digits[2]] += 1
        matrix[1][key2][digits[3]] += 1
    return matrix

def top6_markov_order2(df, top_n=6):
    data = df["angka"].astype(str).tolist()
    matrix = build_transition_matrix_order2(data)

    pairs = [x[:2] for x in data]
    top_pairs = Counter(pairs).most_common(top_n)
    
    if not top_pairs:
        return [_ensure_unique_top_n([], n=top_n) for _ in range(4)]
        
    d1, d2 = top_pairs[0][0][0], top_pairs[0][0][1]

    top_d1 = list(set([int(p[0][0]) for p in top_pairs]))
    top_d2 = list(set([int(p[0][1]) for p in top_pairs]))

    hasil = [_ensure_unique_top_n(top_d1, n=top_n), _ensure_unique_top_n(top_d2, n=top_n)]

    key1 = d1 + d2
    dist3 = matrix[0].get(key1, {})
    top_d3_sorted = sorted(dist3.items(), key=lambda x: -x[1])
    top_d3 = [int(k) for k, _ in top_d3_sorted]
    hasil.append(_ensure_unique_top_n(top_d3, n=top_n))

    key2 = d2 + str(hasil[2][0]) if hasil[2] else d2 + '0'
    dist4 = matrix[1].get(key2, {})
    top_d4_sorted = sorted(dist4.items(), key=lambda x: -x[1])
    top_d4 = [int(k) for k, _ in top_d4_sorted]
    hasil.append(_ensure_unique_top_n(top_d4, n=top_n))

    return hasil

# HYBRID
def top6_markov_hybrid(df, top_n=6):
    hasil1, _ = top6_markov(df, top_n=top_n)
    hasil2 = top6_markov_order2(df, top_n=top_n)

    hasil = []
    for i in range(4):
        gabung = list(dict.fromkeys(hasil1[i] + hasil2[i]))
        freq = Counter(gabung)
        top_kandidat = [k for k, _ in freq.most_common()]
        hasil.append(_ensure_unique_top_n(top_kandidat, n=top_n))

    return hasil
