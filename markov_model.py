import random
from collections import defaultdict, Counter
import pandas as pd

# MARKOV ORDER-1
def build_transition_matrix(data):
    matrix = [defaultdict(lambda: defaultdict(int)) for _ in range(3)]
    for number in data:
        digits = f"{int(number):04d}"
        for i in range(3):
            matrix[i][digits[i]][digits[i+1]] += 1
    return matrix

def top6_markov(df):
    data = df["angka"].astype(str).tolist()
    matrix = build_transition_matrix(data)

    freq_ribuan = Counter([int(x[0]) for x in data])
    transisi = [{k: dict(v) for k, v in matrix[i].items()} for i in range(3)]
    kombinasi = Counter(data).most_common(10)

    hasil = []

    # Ribuan (digit ke-1)
    top6_pos1 = [k for k, _ in freq_ribuan.most_common(6)]
    while len(top6_pos1) < 6:
        top6_pos1.append(random.randint(0, 9))
    hasil.append(top6_pos1)  # index 0 → ribuan

    # Ratusan (matrix[0]), Puluhan (matrix[1]), Satuan (matrix[2])
    for i in range(3):
        kandidat = []
        for prev in matrix[i]:
            kandidat.extend(matrix[i][prev].keys())
        kandidat = Counter(kandidat).most_common()
        top6 = [int(k) for k, _ in kandidat[:6]]
        while len(top6) < 6:
            top6.append(random.randint(0, 9))
        hasil.append(top6)

    info = {
        "frekuensi_ribuan": dict(freq_ribuan),
        "transisi": transisi,
        "kombinasi_populer": kombinasi
    }

    print("Ribuan:", hasil[0], flush=True)
    print("Ratusan:", hasil[1], flush=True)
    print("Puluhan:", hasil[2], flush=True)
    print("Satuan:", hasil[3], flush=True)

    return [hasil[0], hasil[1], hasil[2], hasil[3]], info

# MARKOV ORDER-2
def build_transition_matrix_order2(data):
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

def top6_markov_order2(df):
    data = df["angka"].astype(str).tolist()
    matrix = build_transition_matrix_order2(data)

    pairs = [x[:2] for x in data]
    top_pairs = Counter(pairs).most_common(6)
    d1, d2 = top_pairs[0][0][0], top_pairs[0][0][1]

    top6_d1 = list(set([int(p[0][0]) for p in top_pairs]))
    top6_d2 = list(set([int(p[0][1]) for p in top_pairs]))
    while len(top6_d1) < 6:
        top6_d1.append(random.randint(0, 9))
    while len(top6_d2) < 6:
        top6_d2.append(random.randint(0, 9))

    hasil = [top6_d1, top6_d2]

    key1 = d1 + d2
    dist3 = matrix[0].get(key1, {})
    top6_d3 = sorted(dist3.items(), key=lambda x: -x[1])
    top6_d3 = [int(k) for k, _ in top6_d3[:6]]
    while len(top6_d3) < 6:
        top6_d3.append(random.randint(0, 9))
    hasil.append(top6_d3)

    key2 = d2 + str(top6_d3[0])
    dist4 = matrix[1].get(key2, {})
    top6_d4 = sorted(dist4.items(), key=lambda x: -x[1])
    top6_d4 = [int(k) for k, _ in top6_d4[:6]]
    while len(top6_d4) < 6:
        top6_d4.append(random.randint(0, 9))
    hasil.append(top6_d4)

    return hasil

# HYBRID
def top6_markov_hybrid(df):
    hasil1, _ = top6_markov(df)
    hasil2 = top6_markov_order2(df)

    hasil = []
    for i in range(4):
        gabung = hasil1[i] + hasil2[i]
        freq = Counter(gabung)
        top6 = [k for k, _ in freq.most_common(6)]
        while len(top6) < 6:
            top6.append(random.randint(0, 9))
        hasil.append(top6)

    return hasil
