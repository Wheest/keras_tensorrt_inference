# Keras to TensorRT inference

[Install the latest TensorFlow for the Jetson platform](https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/).

Install other dependencies, such as `pip3 install keras Pillow pycuda pillow`

Ensure all commands are run with Python3.

Run the script `$ python3 export_keras_mobilenetv2.py`.  It creates an `.h5` model from a pretrained model, exports to TensorFlow `.pb`, and exports to the UFF format.

Note that in input and output nodes' names are printed.  If you use a different model, this will need to be changed for your TensorRT inference script.

For this example, `$ python3 run_mobilenetv2.py`, which loads the model from UFF and runs inference on a random image in the folder.

However, it appears that the output is incorrect, when compared to the original Keras model.

See this TensorRT thread on the issue:
https://github.com/NVIDIA/TensorRT/issues/375
