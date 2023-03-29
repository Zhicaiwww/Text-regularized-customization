import os
import requests

# create directory for images
if not os.path.exists('laion-400m-dog'):
    os.makedirs('laion-400m-dog')

# download images
count = 0
for i in range(1, 1001):
    url = f'https://storage.googleapis.com/laion-400m/dogs/{i}.jpg'
    response = requests.get(url)
    if response.status_code == 200:
        with open(f'laion-400m-dog/{i}.jpg', 'wb') as f:
            f.write(response.content)
        count += 1
        if count == 600:
            break