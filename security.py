import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Tkinterウィンドウを作成
root = tk.Tk()
root.title("Autoencoderを使用した異常検出")

# ディレクトリパスを入力するテキストボックスを作成
directory_path_entry = tk.Entry(root)
directory_path_entry.pack()

# 画像表示用のキャンバスを作成
canvas = tk.Canvas(root, width=200, height=200)
canvas.pack()

# 結果表示エリアを作成
result_text = tk.Text(root, height=10, width=40)
result_text.pack()

# Autoencoderモデルの定義
class ComplexAutoencoder(nn.Module):
    def __init__(self):
        super(ComplexAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(140*150, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 140*150),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# モデルのインスタンスを作成
complex_autoencoder = ComplexAutoencoder()

# モデルの重みを保存する関数を定義
def save_model_weights(model, model_weights_path):
    torch.save(model.state_dict(), model_weights_path)

# モデルの重みを読み込む関数を定義
def load_model_weights(model, model_weights_path):
    model.load_state_dict(torch.load(model_weights_path))
    model.eval()

# 選択したディレクトリ内の画像データを使用してAutoencoderモデルを訓練して結果を表示
def train_and_display_results(selected_folder_path):
    # 選択したディレクトリ内の画像データを読み込む
    selected_inputs, selected_image_names = load_images_from_directory(selected_folder_path)

    # 損失関数とオプティマイザの定義
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(complex_autoencoder.parameters(), lr=0.001)

    # モデルの学習
    num_epochs = 10
    for epoch in range(num_epochs):
        for i, data in enumerate(selected_inputs):
            optimizer.zero_grad()
            outputs = complex_autoencoder(data.view(-1, 140*150))
            loss = criterion(outputs, data.view(-1, 140*150))
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 学習が完了したらモデルの重みを保存
    save_model_weights(complex_autoencoder, 'autoencoder_weights.pth')

    print("学習が完了しました。")

    # Autoencoderモデルで選択したディレクトリ内の画像データを評価
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, "選択したディレクトリ内の画像データに関する結果:\n")
    with torch.no_grad():
        anomaly_threshold = 0.0347
        for i, data in enumerate(selected_inputs):
            outputs = complex_autoencoder(data.view(-1, 140*150))
            loss = criterion(outputs, data.view(-1, 140*150))
            result_text.insert(tk.END, f"{selected_image_names[i]} - Loss: {loss.item():.4f}\n")
            if loss.item() > anomaly_threshold:
                result_text.insert(tk.END, "Anomaly detected!\n\n", ("anomaly",))
            else:
                result_text.insert(tk.END, "No anomaly detected.\n\n")

# ディレクトリを選択して学習と結果表示するボタンのクリックイベントハンドラ
def select_directory_and_train():
    folder_path = directory_path_entry.get()

    if not folder_path:
        messagebox.showerror("エラー", "ディレクトリが選択されていません。")
        return
    
    train_and_display_results(folder_path)

select_folder_button = tk.Button(root, text="ディレクトリを選択して学習と結果表示", command=select_directory_and_train)
select_folder_button.pack(pady=20)

# 学習データを追加しないボタンのクリックイベントハンドラ
def add_data_button_clicked():
    # 既存のモデルの重みを読み込む
    load_model_weights(complex_autoencoder, 'autoencoder_weights.pth')
    # 既存のモデルで結果を表示
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, "既存のモデルで評価中...\n")
    evaluate_existing_model(complex_autoencoder)

add_data_button = tk.Button(root, text="学習データを追加しない", command=add_data_button_clicked)
add_data_button.pack(pady=20)

def load_images_from_directory(directory_path):
    # 画像フォルダのパスを設定
    image_folder_path = directory_path
    # 画像ファイルの拡張子
    image_extensions = ['.png']
    # 画像データとファイル名を格納するリスト
    images = []

    # 画像ファイルのリストを取得
    image_files = [file for file in os.listdir(image_folder_path) if os.path.splitext(file)[1].lower() in image_extensions]

    # 画像を数値データに変換してリストに格納
    new_image_size = (140, 150)
    for image_file in image_files:
        image_path = os.path.join(image_folder_path, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, new_image_size)
        images.append(image)

    # imagesリストをNumPy配列に変換
    images_array = np.array(images)
    # 画素値を0から1の範囲に正規化
    images_array = images_array.astype('float32') / 255.0
    # NumPy配列をPyTorchテンソルに変換
    inputs = torch.tensor(images_array, dtype=torch.float32)

    return inputs, image_files

# 既存のモデルを使用して評価を行う関数
def evaluate_existing_model(model):
    # ディレクトリを選択して結果を表示
    folder_path = directory_path_entry.get()
    if not folder_path:
        messagebox.showerror("エラー", "ディレクトリが選択されていません。")
        return
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, "ディレクトリを選択中...\n")
    selected_inputs, selected_image_names = load_images_from_directory(folder_path)
    criterion = nn.MSELoss()
    anomaly_threshold = 0.0347
    with torch.no_grad():
        for i, data in enumerate(selected_inputs):
            outputs = model(data.view(-1, 140*150))
            loss = criterion(outputs, data.view(-1, 140*150))
            result_text.insert(tk.END, f"{selected_image_names[i]} - Loss: {loss.item():.4f}\n")
            if loss.item() > anomaly_threshold:
                result_text.insert(tk.END, "Anomaly detected!\n\n", ("anomaly",))
            else:
                result_text.insert(tk.END, "No anomaly detected.\n\n")

root.mainloop()
