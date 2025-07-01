import tensorflow as tf

print(f"TensorFlow Version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print(f"Found {len(gpus)} GPU(s):")
    for i, gpu in enumerate(gpus):
        print(f"  - GPU[{i}]: {gpu.name}")
else:
    print("Failed to find any GPU devices.")

print("Is GPU available for TensorFlow?", tf.test.is_gpu_available(cuda_only=True))