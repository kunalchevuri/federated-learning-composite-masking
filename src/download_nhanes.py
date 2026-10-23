"""
Download the raw NHANES files used by build_labels.py and build_features.py.

  python src/download_nhanes.py          2015-2018 (cycles I, J): everything the paper uses
  python src/download_nhanes.py 2011     also 2011-2014 (cycles G, H), for build_labels.py 2011

Files go to data/nhanes_raw/{COMPONENT}_{cycle}.xpt; files already present are skipped.
NHANES data are public: https://wwwn.cdc.gov/nchs/nhanes/
"""
import os
import sys
import urllib.request

RAW = "data/nhanes_raw/"
URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{name}_{cycle}.xpt"
YEARS = {'G': 2011, 'H': 2013, 'I': 2015, 'J': 2017}
# labels and base inputs (build_labels.py) + enriched features (build_features.py)
COMPONENTS = ['DEMO', 'BPX', 'BMX', 'DPQ', 'DR1TOT', 'HSQ', 'PAQ', 'SLQ',
              'GHB', 'TCHOL', 'HDL', 'SMQ', 'HIQ', 'HUQ', 'DIQ', 'BPQ', 'MCQ', 'RXQ_RX']
LABEL_COMPONENTS = ['DEMO', 'BPX', 'BMX', 'DPQ', 'DR1TOT', 'HSQ', 'PAQ', 'SLQ']


def main():
    os.makedirs(RAW, exist_ok=True)
    jobs = [(c, cyc) for cyc in 'IJ' for c in COMPONENTS]
    if len(sys.argv) > 1 and sys.argv[1] == '2011':
        jobs += [(c, cyc) for cyc in 'GH' for c in LABEL_COMPONENTS]
    for name, cyc in jobs:
        path = f"{RAW}{name}_{cyc}.xpt"
        if os.path.exists(path):
            continue
        url = URL.format(year=YEARS[cyc], name=name, cycle=cyc)
        print(f"downloading {url}", flush=True)
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as r, open(path + '.part', 'wb') as f:
            f.write(r.read())
        os.replace(path + '.part', path)
    print(f"{len(jobs)} files in {RAW}")


if __name__ == '__main__':
    main()
