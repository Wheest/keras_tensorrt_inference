# Keras to TensorRT inference

[Install the latest TensorFlow for the Jetson platform](https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/).

Install other dependencies, such as `pip3 install keras Pillow pycuda pillow`

Ensure all commands are run with Python3.

Run the script `$ python3 export_keras_mobilenetv2.py`.  It creates an `.h5` model from a pretrained model, exports to TensorFlow `.pb`, and exports to the UFF format.

Note that in input and output nodes' names are printed.  If you use a different model, this will need to be changed for your TensorRT inference script.

For this example, `$ python3 run_mobilenetv2.py`, which loads the model from UFF and runs inference on a random image in the folder.

However, it appears that the output is incorrect, when compared to the original Keras model.

You can see the Keras model predictions with:

```
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import os
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2 as Net

model = Net(weights='imagenet')

img_path = '/home/gecl/keras_tensorrt_inference/mobilenetv2/tabby_tiger_cat.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])
```

See this TensorRT thread on the issue:
https://github.com/NVIDIA/TensorRT/issues/375
