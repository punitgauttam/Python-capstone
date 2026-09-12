"""
Downloads the three credit-risk datasets used in the Track C project:
  1. German Credit (UCI)      - no login needed
  2. Give Me Some Credit      - needs Kaggle account + API key
  3. Lending Club             - needs Kaggle account + API key

Before running this script:
  - pip install kaggle pandas requests
  - Set up your Kaggle API key (kaggle.json) - see the step-by-step guide
    provided alongside this script.
  - Visit kaggle.com/c/GiveMeSomeCredit and click "Join Competition" once
    in your browser (the API will refuse the download otherwise).

Run with:  python download_datasets.py
"""

import os
import zipfile
import requests

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


def download_german_credit():
    """German Credit dataset from UCI - direct download, no login required."""
    print("\n[1/3] Downloading German Credit dataset from UCI...")
    folder = os.path.join(DATA_DIR, "german_credit")
    os.makedirs(folder, exist_ok=True)

    url = "https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip"
    zip_path = os.path.join(folder, "german_credit.zip")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            f.write(response.content)

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(folder)

        os.remove(zip_path)
        print(f"    Done. Files saved in: {folder}/")
    except Exception as e:
        print(f"    FAILED: {e}")
        print("    Try downloading manually from:")
        print("    https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data")


def download_kaggle_dataset(competition_or_dataset, folder_name, is_competition=False):
    """Uses the official kaggle package (requires kaggle.json to be set up)."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("    FAILED: 'kaggle' package not installed. Run: pip install kaggle")
        return
    except OSError as e:
        print(f"    FAILED: Kaggle credentials not found. {e}")
        print("    Make sure kaggle.json is in the right folder (see setup steps).")
        return

    folder = os.path.join(DATA_DIR, folder_name)
    os.makedirs(folder, exist_ok=True)

    try:
        api = KaggleApi()
        api.authenticate()

        if is_competition:
            api.competition_download_files(competition_or_dataset, path=folder)
        else:
            api.dataset_download_files(competition_or_dataset, path=folder, unzip=False)

        # Unzip whatever was downloaded
        for fname in os.listdir(folder):
            if fname.endswith(".zip"):
                zip_path = os.path.join(folder, fname)
                with zipfile.ZipFile(zip_path, "r") as z:
                    z.extractall(folder)
                os.remove(zip_path)

        print(f"    Done. Files saved in: {folder}/")
    except Exception as e:
        print(f"    FAILED: {e}")
        if is_competition:
            print(f"    Make sure you've clicked 'Join Competition' at:")
            print(f"    https://www.kaggle.com/c/{competition_or_dataset}")
        else:
            print(f"    Try downloading manually from:")
            print(f"    https://www.kaggle.com/datasets/{competition_or_dataset}")


def download_give_me_some_credit():
    print("\n[2/3] Downloading Give Me Some Credit from Kaggle...")
    download_kaggle_dataset("GiveMeSomeCredit", "give_me_some_credit", is_competition=True)


def download_lending_club():
    print("\n[3/3] Downloading Lending Club data from Kaggle...")
    print("    Note: this dataset is large (multiple GB). This may take a while.")
    download_kaggle_dataset("wordsforthewise/lending-club", "lending_club", is_competition=False)


if __name__ == "__main__":
    download_german_credit()
    download_give_me_some_credit()
    download_lending_club()
    print("\nAll downloads attempted. Check the 'data' folder for results.")
    print("If any step said FAILED, follow the manual link it printed instead.")
