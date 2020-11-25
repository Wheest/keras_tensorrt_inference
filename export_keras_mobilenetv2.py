import os
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2 as Net

# save Keras model to `.h5` format
model = Net(weights='imagenet')

os.makedirs('./model', exist_ok=True)
# Save the h5 file to path specified.
model.save("./model/model.h5")

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.framework import graph_io
from tensorflow.keras.models import load_model

# Clear any previous session.
tf.keras.backend.clear_session()

# Load `.h5` model, and export to TensorFlow `.pb` model
# Print input and output node names, to be used in TensorRT script
save_pb_dir = './model'
model_fname = './model/model.h5'
def freeze_graph(graph, session, output, save_pb_dir='.', save_pb_name='frozen_model.pb', save_pb_as_text=False):
    with graph.as_default():
        graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
        graphdef_frozen = tf.graph_util.convert_variables_to_constants(session, graphdef_inf, output)
        graph_io.write_graph(graphdef_frozen, save_pb_dir, save_pb_name, as_text=save_pb_as_text)
        return graphdef_frozen

# This line must be executed before loading Keras model.
tf.keras.backend.set_learning_phase(0)

model = load_model(model_fname)

session = tf.keras.backend.get_session()

input_names = [t.op.name for t in model.inputs]
output_names = [t.op.name for t in model.outputs]

frozen_graph = freeze_graph(session.graph, session, [out.op.name for out in model.outputs], save_pb_dir=save_pb_dir)

import uff
# convert to uff model
output_fname = 'mobilenetv2/mobilenetv2.uff'
uff = uff.from_tensorflow(frozen_graph, output_filename=output_fname)

# Prints input and output nodes names, take notes of them.
print(f'Finished, exported UFF model to {output_fname}')
print(f'In your ModelData class for inference, set INPUT_NAME to {input_names[0]}')
print(f'and OUTPUT_NAME to `{output_names}`')
