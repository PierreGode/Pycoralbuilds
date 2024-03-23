import numpy as np
from PIL import Image
from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.utils.edgetpu import make_interpreter

# Initialize the TPU interpreter
interpreter = make_interpreter('mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite')
interpreter.allocate_tensors()

# Load and prepare your test image
image = Image.open('parrot.jpg').convert('RGB').resize(common.input_size(interpreter), Image.ANTIALIAS)
input_tensor = np.asarray(image).flatten()

# Perform inference
common.set_input(interpreter, input_tensor)
interpreter.invoke()

# Get the result and print the classification
classes = classify.get_classes(interpreter, top_k=1)
for c in classes:
    print(f'Class: {c.id}, Score: {c.score}')
