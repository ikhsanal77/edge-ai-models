import tensorflow as tf
import numpy as np
import os

# Konfigurasi
TIMESTEPS = 60
FEATURES = 3
EPOCHS = 5
BATCH_SIZE = 16

# Buat model sederhana
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(8, input_shape=(TIMESTEPS, FEATURES), 
        tf.keras.layers.RepeatVector(TIMESTEPS),
        tf.keras.layers.LSTM(8, return_sequences=True),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(FEATURES))
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def main():
    try:
        # 1. Load data
        data_path = '../datasets/raw/train_data.npy'
        print(f"Mencoba load data dari: {data_path}")
        
        if not os.path.exists(data_path):
            print("❌ Data tidak ditemukan!")
            return
            
        X_train = np.load(data_path)
        print(f"✅ Data berhasil di-load. Shape: {X_train.shape}")
        
        # 2. Normalisasi sederhana
        X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
        
        # 3. Buat dan training model
        model = create_model()
        model.summary()
        
        # 4. Siapkan folder output
        os.makedirs('../edge_model', exist_ok=True)
        model_path = '../edge_model/autoencoder.h5'
        
        # 5. Training langsung simpan model
        model.fit(X_train, X_train, 
                 epochs=EPOCHS, 
                 batch_size=BATCH_SIZE,
                 validation_split=0.2)
        
        model.save(model_path)
        print(f"✅ Model disimpan di: {model_path}")
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")

if __name__ == "__main__":
    print("="*50)
    print("MEMULAI TRAINING MODEL")
    print("="*50)
    main()
