import json

import requests

# Request created from CURL: https://curlconverter.com/
headers = {
    'accept': 'application/json',
    # requests won't add a boundary if this header is set when you pass files=
    # 'Content-Type': 'multipart/form-data',
}

params = {
    'threshold': '0.9',
    'classes_of_interest': [
        '1',
        '3',
    ],
}

files = [
    ('files', ('test_image', open('tests/test_image.jpg', 'rb'), 'image/jpeg')),
    ('files', ('test_image2', open('tests/test_image2.jpg', 'rb'), 'image/jpeg')),
]

try:
    response = requests.post('http://localhost/object-detection/', 
                             params=params, headers=headers, files=files)
    response.raise_for_status()
except requests.exceptions.HTTPError as err:
    print(err)
    exit()

detection_results = json.loads(response.text)

try:
    assert detection_results[0]['test_image']['labels'] == ['person', 'car']
    assert detection_results[1]['test_image2']['labels'] == ['person', 'car', 
                                                             'car', 'car']
    print("Detection results correct. TESTS OK")
except AssertionError:
    print("Detection results incorrect. TESTS KO")
