import requests
from bs4 import BeautifulSoup
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

def get_image_urls(query, num_images=10):
    """Scrape image URLs for a given query from Bing."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    search_url = f"https://www.bing.com/images/search?q={query.replace(' ', '+')}&form=QBLH"
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    image_elements = soup.find_all('a', {'class': 'iusc'}, limit=num_images)
    image_urls = []
    for element in image_elements:
        # Extract image metadata
        try:
            m = element['m']
            start = m.find('"murl":"') + 8
            end = m.find('","turl')
            image_url = m[start:end]
            image_urls.append(image_url)
        except KeyError:
            pass
    return image_urls

def download_images(image_urls, folder_name, prefix="image"):
    """Download images from a list of URLs and save them to a folder."""
    os.makedirs(folder_name, exist_ok=True)  # Create folder if it doesn't exist

    for i, url in enumerate(image_urls):
        try:
            response = requests.get(url, stream=True, timeout=10)
            response.raise_for_status()  # Check if the request was successful
            
            # Define the file name and path
            file_name = f"{prefix}_{i + 1}.jpg"
            file_path = os.path.join(folder_name, file_name)
            
            # Write the content to a file
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            
            print(f"✅ Image {i + 1} downloaded: {file_path}")
        except Exception as e:
            print(f"❌ Failed to download image {i + 1}: {e}")

# Example usage
competitors = ['rouiba boite de jus', 'toudja boite de jus', 'ifri boite de jus', 'tazedj boite de jus']

if __name__ == '__main__':
    for competitor in competitors:
        print(f"Fetching images for: {competitor}")
        
        # Fetch image URLs
        image_urls = get_image_urls(competitor, num_images=50)
        
        # Download images
        if image_urls:
            download_images(image_urls, folder_name=competitor.replace(" ", "_"), prefix=competitor.split()[0])
        else:
            print(f"❌ No images found for {competitor}")
