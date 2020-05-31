import requests
import shutil
from tqdm import tqdm
import os
from pathlib import Path


def download_image(url, file_name, dict_path, verbose=True):
    # Open the url image, set stream to True, this will return the stream content.
    try:
        resp = requests.get(url, stream=True)

        # Open a local file with wb ( write binary ) permission.
        file_path = os.path.join(dict_path, file_name)

        local_file = open(file_path, 'wb')

        # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
        resp.raw.decode_content = True

        # Copy the response stream raw data to local image file.
        shutil.copyfileobj(resp.raw, local_file)

        # Remove the image url response object.
        del resp
        
        if verbose:
            print('Downloaded image from {}'.format(url))

    except requests.exceptions.RequestException as e:
        print('Failed to get image from {}'.format(url))



def main():
    # 'Plant-flora-plant-life.txt', 
    #         'Fungus.txt', 
    #         'Person-individual-someone-somebody-mortal-soul.txt'
    files = ['Artifact-artefact.txt', 
            'Geological-formation-formation.txt', 
            'Animal-animate-being-beast-brute-creature-fauna.txt']

    for file_name in files:
        file_path = Path('/home/stanislaw/Downloads/', file_name)
        dict_path = Path('./dataset', file_name.split('.txt')[0])
        dict_path.mkdir(parents=True, exist_ok=True)
        with file_path.open('r') as f:
            lines = f.readlines()
            for url in tqdm(lines):
                image_name = url.lower().split('/')[-1].split('.jpg')[0] + '.jpg'
                download_image(url, image_name, dict_path=dict_path, verbose=False)


if __name__ == '__main__':
    main()