import requests


PTAU_FILES = {
    8: 'https://storage.googleapis.com/zkevm/ptau/powersOfTau28_hez_final_08.ptau',
    9: 'https://storage.googleapis.com/zkevm/ptau/powersOfTau28_hez_final_09.ptau',
    10: 'https://storage.googleapis.com/zkevm/ptau/powersOfTau28_hez_final_10.ptau',
    11: 'https://storage.googleapis.com/zkevm/ptau/powersOfTau28_hez_final_11.ptau',
    12: 'https://storage.googleapis.com/zkevm/ptau/powersOfTau28_hez_final_12.ptau',
    13: 'https://storage.googleapis.com/zkevm/ptau/powersOfTau28_hez_final_13.ptau',
    14: 'https://storage.googleapis.com/zkevm/ptau/powersOfTau28_hez_final_14.ptau',
    15: 'https://storage.googleapis.com/zkevm/ptau/powersOfTau28_hez_final_15.ptau',
    16: 'https://storage.googleapis.com/zkevm/ptau/powersOfTau28_hez_final_16.ptau',
    17: 'https://storage.googleapis.com/zkevm/ptau/powersOfTau28_hez_final_17.ptau',
    18: 'https://storage.googleapis.com/zkevm/ptau/powersOfTau28_hez_final_18.ptau',
    19: 'https://storage.googleapis.com/zkevm/ptau/powersOfTau28_hez_final_19.ptau',
    20: 'https://storage.googleapis.com/zkevm/ptau/powersOfTau28_hez_final_20.ptau',
    21: 'https://storage.googleapis.com/zkevm/ptau/powersOfTau28_hez_final_21.ptau',
    22: 'https://storage.googleapis.com/zkevm/ptau/powersOfTau28_hez_final_22.ptau',
    23: 'https://storage.googleapis.com/zkevm/ptau/powersOfTau28_hez_final_23.ptau',
    24: 'https://storage.googleapis.com/zkevm/ptau/powersOfTau28_hez_final_24.ptau',
}


def get_remote_file_size(url):
    try:
        response = requests.head(url, allow_redirects=True)

        if response.status_code == 200:
            size_bytes = int(response.headers.get('Content-Length', 0))
            return size_bytes
        else:
            print(f"Failed to access: {response.status_code}")
            return None
    except Exception as e:
        print(f"Connection error: {e}")
        return None


def ptau_powers(constraints: int):
    for i in range(31, 0, -1):
        if (constraints >> i) & 1:
            return i+1
    return constraints & 1


def get_ptau_size_gb(constraints: int):
    powers = ptau_powers(constraints)
    bytes_size = get_remote_file_size(PTAU_FILES[powers])
    return bytes_size / 1_000_000_000.0
