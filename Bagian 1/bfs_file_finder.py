import os
from collections import deque

def bfs_file_search(start_path, target_filename):
    """
    Melakukan pencarian berkas menggunakan BFS dari direktori awal yang diberikan.

    Args:
        start_path (str): Jalur direktori awal untuk memulai pencarian.
        target_filename (str): Nama berkas yang ingin dicari (case-sensitive).

    Returns:
        str: Jalur lengkap berkas jika ditemukan, None jika tidak ditemukan.
    """
    queue = deque()
    visited = set()

    # Pastikan jalur awal adalah direktori yang valid
    if not os.path.isdir(start_path):
        print(f"Error: Jalur awal '{start_path}' bukan direktori yang valid atau tidak ada.")
        return None

    queue.append(start_path)
    visited.add(start_path)

    while queue:
        current_path = queue.popleft()

        # Periksa apakah ini file yang dicari
        if os.path.isfile(current_path) and os.path.basename(current_path) == target_filename:
            return current_path

        # Jika ini direktori, jelajahi isinya
        if os.path.isdir(current_path):
            try:
                for item_name in os.listdir(current_path):
                    item_path = os.path.join(current_path, item_name)
                    if item_path not in visited:
                        visited.add(item_path)
                        queue.append(item_path)
            except PermissionError:
                print(f"Izin ditolak untuk mengakses: {current_path} (Lewati)")
            except Exception as e:
                print(f"Terjadi kesalahan saat membaca {current_path}: {e} (Lewati)")
    return None

# --- Bagian Pencarian File ---
if __name__ == "__main__":
    # Direktori awal berupa D:\Kuliah
    starting_directory = r"D:\Kuliah"

    print("\n--- Pencarian Berkas BFS ---")
    print(f"Pencarian akan dimulai dari direktori: {starting_directory}")

    # Meminta input nama file dari pengguna
    base_file_name = input("Masukkan nama file yang ingin dicari (tanpa ekstensi, cth: 'laporan_akhir'): ")

    # Meminta input tipe file (ekstensi) dari pengguna
    file_extension = input("Masukkan tipe/ekstensi file (cth: 'pdf', 'docx', 'txt'): ")

    # Menggabungkan nama file dan ekstensi untuk mendapatkan nama file target lengkap
    if not file_extension.startswith('.'):
        full_target_filename = f"{base_file_name}.{file_extension}"
    else:
        full_target_filename = f"{base_file_name}{file_extension}"

    print(f"\nMencari '{full_target_filename}' di dalam '{starting_directory}'...")

    found_path = bfs_file_search(starting_directory, full_target_filename)

    if found_path:
        print(f"\nBerkas ditemukan di: {found_path}")
    else:
        print(f"\nBerkas '{full_target_filename}' tidak ditemukan di '{starting_directory}' atau subdirektorinya.")

    print("\n--- Pencarian Selesai ---")