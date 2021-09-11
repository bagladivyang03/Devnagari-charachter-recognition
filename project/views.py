from django.shortcuts import render
from PIL import Image
from django.views.decorators import csrf
# from skimage.transform import resize
# from scipy.misc.pilutil import imread,imresize
import numpy as np
import re
import sys
import os
print(os.path.abspath("./model"))
sys.path.append(os.path.abspath("./models"))
from .utils import *
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
global model, graph

import base64
OUTPUT = os.path.join(os.path.dirname(__file__), 'output.png')
from PIL import Image
from io import BytesIO
def getI420FromBase64(codec):
    base64_data = re.sub('^data:image/.+;base64,', '', codec)
    # print(base64_data, "image data")
    byte_data = base64.b64decode(base64_data)
    image_data = BytesIO(byte_data)
    img = Image.open(image_data)
    img.save(OUTPUT)


def convertImage(imgData):
    getI420FromBase64(imgData)

@csrf_exempt
def predict(request):
    imgData = request.POST.get('img')
    # print(imgData, "imagedata")
    convertImage(imgData)
    # print(imgData)
    x = Image.open(OUTPUT)
    # print(type(x))
    x = np.invert(x)
    # print(type(x))
    # x =  x.resize((32,32),Image.BICUBIC)
    x = np.array(Image.fromarray(x).resize((32,32)))
    # print(type(x))
    x = x.reshape(2, 32, 32, 2)
    model, graph = init()
    print(model, "model")
    print(graph, "graph")
    with graph.as_default():
        out = model.predict(x)
        print(out)
        print(np.argmax(out, axis=1))
        response = np.array_str(np.argmax(out, axis=1))
        return JsonResponse({"output": response})

    # with graph.as_default():
    #     json_file = open('models/model.json','r')
        # loaded_model_json = json_file.read()
        # json_file.close()
        # loaded_model = model_from_json(loaded_model_json)
        # #load weights into new model
        # loaded_model.load_weights("models/model.h5")
        # print("Loaded Model from disk")
        # #compile and evaluate loaded model
        # loaded_model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        # # perform the prediction
        # out = loaded_model.predict(img)
        # print(out)
        # print(class_names[np.argmax(out)])
        # # convert the response to a string
        # response = class_names[np.argmax(out)]
        # return str(response)


def index(request):
    return render(request, 'index.html', {})
