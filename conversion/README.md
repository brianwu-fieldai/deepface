# Conversion Module
This module was added to convert TensorFlow models to facilitate the conversion of existing TensorFlow models part of the DeepFace library into ONNX and PyTorch formats.

## Dependencies
If some of the models here were previously built using TensorFlow, then TensorFlow will be required in order to convert the model to ONNX (in order to ensure that the HDF5 binary can be properly exported to the SavedModel directory format). This means that TensorFlow must be enabled as a dependency if you wish to do model conversion from an existing TensorFlow HDF5 binary. To ensure that this is installed, run `pip install -r requirements_additional.txt` from root in your virtual environment.

## Usage
First, ensure that the PyTorch model weights are downloaded locally. You will first need to convert it to TensorFlow's SavedModel format. You can do this by following some of the examples in `model_conversions`; these scripts follow the model definition from the TF model's definition in `deepface/deepface/models/` and output a SavedModel object from the HDF5 binary.

Once this is done, convert this model to the ONNX format by using the `convert_savedmodel_to_onnx` function in `savedmodel_2_onnx.py`. You can then convert this ONNX model to PyTorch format using the `convert_onnx_to_torch` function in `onnx_2_torch.py`. 

Once this is done, go to the model definition path in `deepface/deepface/models/` and define a script that captures the structure of the model in PyTorch and loads either the ONNX or PyTorch checkpoint. The model class name also has to be registered in the following:
- `deepface/config/confidence.py`
- `deepface/config/threshold.py`
- `deepface/modules/modeling.py`
- `deepface/tests/test_verify.py`
- `deepface/tests/visual-test.py`

Once this model is registered, go to the root directory and run `pip install -e .` in your virtual environment. Your converted model can now be called using the DeepFace wrapper!