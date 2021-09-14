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
    convertImage(imgData)
    x = Image.open(OUTPUT)
    print(x.size)
    print(x.size)
    x = x.convert('L')
    x = x.resize((32,32))
    x.save(OUTPUT)
    x = np.array(x)
    x = x.reshape(1,32,32,1)
    print(len(x),"length of array")
    model, graph = init()
    print(model, "model")
    print(graph, "graph")
    out = model.predict(x)
    # print(out)
    # print(np.argmax(out, axis=1))
    response = np.array(np.argmax(out, axis=1))
    print(response , "response")
    return JsonResponse({"output": str(response[0]) })


def index(request):
    return render(request, 'index.html', { "imagestr" : "static/hindi_characters/1.png"})
