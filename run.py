import base64
import json
import os
import cv2
import numpy as np
from flask import Flask, request
from lib.reidentifier import create_box_encoder

app = Flask(__name__)


@app.route('/feature', methods=['GET'])
def feature():
    if request.method == 'GET':
        req_data = request.get_json()

        image_64 = req_data['image']
        image_64 = image_64.encode("UTF-8")
        image_64 = base64.decodebytes(image_64)
        image_64 = cv2.imdecode(np.frombuffer(image_64, np.uint8), -1)
        print('image_64 type : {}'.format(type(image_64)))

        height, width, _ = image_64.shape
        print('height : {}'.format(height))
        print('width : {}'.format(width))

        features = encoder(image_64, [(0, 0, width, height)])
        feature = features[0].tolist()
        print('first 5 elements in vector=> {}'.format(feature[:5]))
        print('feature len => {}'.format(len(feature)))

        item = {'feature': feature}
        return json.dumps(item)


if __name__ == '__main__':
    print('Flask Start')

    current_dir_path = os.path.dirname(os.path.abspath(__file__))
    print('current path : {}'.format(current_dir_path))
    model_path = current_dir_path + '/model/mars-small128.pb'

    encoder = create_box_encoder(model_path, batch_size=32)  # feature Extractor

    app.run(
        port=3000,
        debug=True,
        host="0.0.0.0")
    print("Reidentifier Running")
