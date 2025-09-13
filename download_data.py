import os
import requests
from tqdm import tqdm

# A dictionary of URLs to publicly available text files.
# These are from Project Gutenberg and other public repositories.
# We'll use these to create a small, diverse dataset for RAG.
TEXT_FILES = {
    "War_of_the_Worlds.txt": "https://www.gutenberg.org/cache/epub/36/pg36.txt",
    "Frankenstein.txt": "https://www.gutenberg.org/files/84/84-0.txt",
    "Pride_and_Prejudice.txt": "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
    "The_Time_Machine.txt": "https://www.gutenberg.org/files/35/35-0.txt",
    "The_Adventures_of_Sherlock_Holmes.txt": "https://www.gutenberg.org/cache/epub/1661/pg1661.txt"
}

def download_file(url, filepath):
    """
    Downloads a file from a URL and saves it to a specified path,
    showing a progress bar.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes

        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc=f"Downloading {os.path.basename(filepath)}")

        with open(filepath, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        print(f"Successfully downloaded {os.path.basename(filepath)}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")

def main():
    """
    Main function to create a data directory and download the text files.
    """
    # Create a directory to store the downloaded files.
    # This keeps our project organized.
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: '{data_dir}'")
    
    # Download each file in our list.
    for filename, url in TEXT_FILES.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            print(f"'{filename}' already exists. Skipping download.")
            continue
        download_file(url, filepath)

if __name__ == "__main__":
    main()
