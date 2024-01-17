import json
import requests
import os
save_folder = "images"

base_url: str = "http://172.24.108.140:18080"
event_url: str = f"{base_url}/api/cctv/get"
for i in range(0,100):
    params = {
        "soCmts": "ID1758",
        "event_type": "HUMAN",
        "page_idx": i, 
        "page_size": 10
    }
    response = requests.get(url=event_url, params=params)
    result_get: dict = response.json()
    event_list = result_get['results']

    for event in event_list:
        soCmt_str = event['soCmt']
        image_path = event['image']
        image_crop_path = event['crop_image']
        metadata_str = event['metadata']
        metadata_dict = json.loads(metadata_str)
        event_type = event['event_type']
        save_path = os.path.join(save_folder, soCmt_str)
        os.makedirs(save_path, exist_ok=True)

        if event_type == "ANPR":
            if image_crop_path is not None:
                url = f"{image_crop_path}"
                print(f"Downloading {url}")
                response = requests.get(url)

                if response.status_code == 200:
                    image_crop_name = os.path.join(save_path, os.path.basename(image_crop_path))
                    with open(image_crop_name, 'wb') as f:
                        f.write(response.content)
            
            # if image_path is not None:
            #     url = f"{image_path}"
            #     response = requests.get(url)
            #     if response.status_code == 200:
            #         image_name = os.path.join(save_path, os.path.basename(image_path))
            #         with open(image_name, 'wb') as f:
            #             f.write(response.content)
        elif event_type == "HUMAN":
            if image_crop_path is not None:
                url = f"{image_crop_path}"
                print(f"Downloading {url}")
                response = requests.get(url)
                if response.status_code == 200:
                    image_crop_name = os.path.join(save_path, os.path.basename(image_crop_path))
                    with open(image_crop_name, 'wb') as f:
                        f.write(response.content)
            
            # if image_path is not None:
            #     url = f"{image_path}"
            #     print(f"Downloading {url}")
            #     response = requests.get(url)
            #     if response.status_code == 200:
            #         image_name = os.path.join(save_path, os.path.basename(image_path))
            #         with open(image_name, 'wb') as f:
            #             f.write(response.content)


            