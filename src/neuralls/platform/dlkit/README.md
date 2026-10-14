# DLKit adapters

Adapters between neuralls and DLKit. DLKit owns model construction and schema validation; this package only loads checkpoints and turns solver arrays into model inputs.

## Modules

- `predictor_adapter.py`: `DLKitPredictor` and `DLKitAdapter`, which implement torchalg's `PredictorPort` and `PredictorAdapter`. `DLKitPredictor.model_class_name` records the loaded torch model's architecture at load time; `DLKitNeuralPreconditioner` (a `NeuralPreconditioner` subclass, since torchalg keeps its predictor private) exposes it so display labels can name the architecture. `_prepare_model_input` adds the batch axis and casts to float64 on the predictor device.
- `inference_adapter.py`: `DLKitInferencePredictor`, the batch-inference port used by application code.
- `callbacks.py`: Lightning callbacks used by DLKit training jobs.
- `_prediction_outputs.py`: normalization of DLKit prediction results into tensors.

Solver arrays are dense here. A CSR matrix must be densified in composition before it reaches a neural predictor; this package does not densify.
