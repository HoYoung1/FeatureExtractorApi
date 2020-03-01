import base64
import json
import os

import requests

import config_test_reid_mock_api as ENV


# sys.path.append(os.path.abspath(os.path.dirname(__file__)))


def test_reid_mock_api():
    for file in ENV.USER_FILE_LIST:
        print('test_reid_image : file - {}'.format(file))
        with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'resource', file), 'rb') as f:
            image_read = f.read()
            image_64_encode = base64.encodebytes(image_read)
            image_64_encode = image_64_encode.decode("UTF-8")
            print('image encode type : {}'.format(image_64_encode))

        headers = {
            'content-type': 'application/json'
        }
        body = {
            'image': image_64_encode
        }
        response = requests.get(ENV.REST_API_URL, data=json.dumps(body), headers=headers)
        print('response : status_code - ', response.status_code)
        print('response text - : ', response.text)

        data = json.loads(response.text)
        feature = data['feature']
        print('first 5 elements in vector=> {}'.format(feature[:5]))
        assert (response.status_code == 200)


test_reid_mock_api()
