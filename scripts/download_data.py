"""Download the published dataset snapshot without replacing an existing original."""
from pathlib import Path
import hashlib
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://video.udacity-data.com/topher/2025/September/68c92f00_noshowappointments-kagglev2-may-2016/noshowappointments-kagglev2-may-2016.csv'
SHA256 = '9132d3e7d0246617df9041d3764f20ad6f08e7b0d9f0997fa254fc5e52eda27d'
DEST = ROOT / 'data/original/noshowappointments-kagglev2-may-2016.csv'

def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if not DEST.exists():
        with urllib.request.urlopen(URL, timeout=90) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != SHA256:
            raise ValueError('Source snapshot changed: review provenance before accepting new data.')
        with DEST.open('xb') as handle:
            handle.write(payload)
    assert hashlib.sha256(DEST.read_bytes()).hexdigest() == SHA256, 'Original checksum mismatch'
    print('Original snapshot present and SHA-256 verified; no data values printed.')

if __name__ == '__main__':
    main()
