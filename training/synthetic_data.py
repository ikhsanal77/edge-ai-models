import numpy as np
import pandas as pd
import os

def generate_vibration_data(num_samples=3000):
    print("Memulai generate data...")
    t = np.linspace(0, 10*np.pi, num_samples)
    normal_data = np.column_stack([
        0.5 * np.sin(t) + 0.1 * np.random.normal(size=num_samples),
        0.3 * np.cos(t) + 0.1 * np.random.normal(size=num_samples),
        0.2 * np.sin(0.5*t) + 0.1 * np.random.normal(size=num_samples)
    ])
    
    # Pastikan folder ada
    os.makedirs('../datasets/raw', exist_ok=True)
    
    # Simpan data mentah
    raw_path = '../datasets/raw/synthetic_vibration.csv'
    pd.DataFrame(normal_data).to_csv(raw_path, index=False)
    
    # Buat data training (windows)
    timesteps = 60
    windows = []
    for i in range(len(normal_data) - timesteps + 1):
        windows.append(normal_data[i:i+timesteps])
    
    train_path = '../datasets/raw/train_data.npy'
    np.save(train_path, np.array(windows))
    
    print(f"✅ Data disimpan di: {train_path}")
    print(f"Shape data: {len(windows)}, {timesteps}, 3")

if __name__ == "__main__":
    generate_vibration_data()
