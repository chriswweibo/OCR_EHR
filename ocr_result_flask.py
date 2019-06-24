#-*- coding:utf-8 -*-

import os
import sys
import ocr
import time
import numpy as np
from PIL import Image
from glob import glob
reload(sys)
sys.setdefaultencoding( "utf-8" )
from flask import Flask, request
app=Flask(__name__)
import tensorflow as tf
graph = tf.get_default_graph()
# images are contained ih the image_input fold
@app.route('/',methods=['GET','POST'])
def ocr_result():
    #arg='1555667289524.jpg'
    arg=request.args.get('id')
    image = np.array(Image.open('/'.join(['./image_input',arg])).convert('RGB'))
    t = time.time()
    with graph.as_default():
        result, image_framed = ocr.model(image)
    output_file = '/'.join(['./image_result', arg+'.txt'])
    # Image.fromarray(image_framed).save(output_file)
    print("Mission complete, it took {:.3f}s".format(time.time() - t))
    print("\nRecognition Result:\n")
    txt=''
    for key in result:
        txt=txt+result[key][1]
    print(txt)
    with open(output_file,'wb') as f:
        f.write(txt)
        f.close()
    return txt

   
