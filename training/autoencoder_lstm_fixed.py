import tensorflow as tf
import numpy as np
import os

# Konfigurasi model yang kompatibel TFLite
def create_compatible_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(60, 3)),
        tf.keras.layers.LSTM(8, return_sequences=True),
        tf.keras.layers.LSTM(8),
        tf.keras.layers.RepeatVector(60),
        tf.keras.layers.LSTM(8, return_sequences=True),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(3))
    ])
    return model

def train_and_export():
    # Load/generate data
    from .synthetic_data import generate_vibration_data
    generate_vibration_data()
    X_train = np.load('datasets/raw/train_data.npy')
    
    # Build and train model
    model = create_compatible_model()
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, X_train, epochs=5, batch_size=16)
    
    # Convert to TFLite with compatibility settings
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter._experimental_lower_tensor_list_ops = False
    
    tflite_model = converter.convert()
    
    # Save outputs
    os.makedirs('edge_model', exist_ok=True)
    with open('edge_model/quantized_model.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print(f"✅ Model converted! Size: {len(tflite_model)/1024:.2f} KB")

if __name__ == "__main__":
    train_and_export()
