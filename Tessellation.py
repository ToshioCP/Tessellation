import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import tkinter as tk
from tkinter import ttk, font
from PIL import Image, ImageTk
import os

# --- 基本設定 ---
A = np.array([0.0, np.sqrt(3)])
B = np.array([-1.0, 0.0])
C = np.array([0.0, 0.0])
D = np.array([1.0, 0.0])
# radius = 0.5
radius = 1
X = 6 # 4 copies horozontally
Y = 3 # 3 copies vertically

# === 省略せずに元の関数群をここに貼り付けてください ===
# generate_tile, translate_tile, generate_hexagon, rotate, bezier_curve,
# random_point_in_circle, draw_tessellation など全て
def rotate(points, center, angle_deg):
    angle_rad = np.radians(angle_deg)
    R = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                  [np.sin(angle_rad),  np.cos(angle_rad)]])
    return (R @ (points - center).T).T + center

# --- ランダム円内点 ---
def random_point_in_circle(center, radius):
    r = radius * np.sqrt(np.random.rand())
    theta = 2 * np.pi * np.random.rand()
    return center + r * np.array([np.cos(theta), np.sin(theta)])

# --- 3次ベジェ曲線 ---
def bezier_curve(p0, p1, p2, p3, n=200):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t)**3 * p0 + 3 * (1 - t)**2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3

# --- プログラムX：1タイル生成 ---
def generate_tile(a, b, c, d):
    mid_AB = (a + b) / 2
    p1 = random_point_in_circle(mid_AB, radius)
    p2 = random_point_in_circle(mid_AB, radius)
    s, t = (p1, p2) if euclidean(p1, a) <= euclidean(p2, a) else (p2, p1)
    curve1 = bezier_curve(a, s, t, b)

    distances_to_C = np.linalg.norm(curve1 - c, axis=1)
    index_U = np.argmin(distances_to_C)
    u = curve1[index_U]

    mid_UC = (u + c) / 2
    p3 = random_point_in_circle(mid_UC, radius)
    p4 = random_point_in_circle(mid_UC, radius)
    p, q = (p3, p4) if euclidean(p3, u) <= euclidean(p4, u) else (p4, p3)
    curve2 = bezier_curve(u, p, q, c)

    triangle = np.array([a, b, d, a])
    ac_line = np.array([a, c])
    return {
        'triangle': triangle,
        'ac_line': ac_line,
        'curve1': curve1,
        'curve2': curve2
    }

def translate_tile(a, tile):
    tile_a = tile['triangle'][0]
    tile_b = tile['triangle'][1]
    tile_d = tile['triangle'][2]
    tile_c = tile['ac_line'][1]
    v = a - tile_a
    triangle = np.array([tile_a + v, tile_b + v, tile_d + v, tile_a + v])
    ac_line = np.array([tile_a + v, tile_c + v])
    curve1 = tile['curve1'] + v
    curve2 = tile['curve2'] + v
    return {
        'triangle': triangle,
        'ac_line': ac_line,
        'curve1': curve1,
        'curve2': curve2
    }
    
# Create a hexagon by a triangle tile A, B, C, D
# Copy the tile by the rotation 0, 60, 120, 180, 240, 300 degrees around point A.
# Generated hexagon is a set (an array) of six tiles.
def generate_hexagon(tile):
    angles = [0, 60, 120, 180, 240, 300]
    hexagon = []
    a = tile['triangle'][0]

    for angle in angles:
        rot_triangle = rotate(tile['triangle'], a, angle)
        rot_ac = rotate(tile['ac_line'], a, angle)
        rot_curve1 = rotate(tile['curve1'], a, angle)
        rot_curve2 = rotate(tile['curve2'], a, angle)
        hexagon.append({'triangle': rot_triangle, 'ac_line': rot_ac, 'curve1': rot_curve1, 'curve2': rot_curve2})

    return hexagon

# （省略）: あなたが提供した関数部分は省略しないで貼り付けてください
# 下にだけ draw_tessellation の一部を修正した点を書いておきます：

def draw_tessellation(show_triangles=True, with_frame=True, filename="output.png"):
    fig, ax = plt.subplots(figsize=(10, 10))
    base_tile = generate_tile(A, B, C, D)
    for x in range(X):
        x_offset = 3*x
        for y in range(Y):
            if x % 2 == 0:
                y_offset = 2*np.sqrt(3)*y
            else:
                y_offset = 2*np.sqrt(3)*y + np.sqrt(3)
            h_a = A + np.array([x_offset, y_offset])
            hx = generate_hexagon(translate_tile(h_a, base_tile))
            for tile in hx:
                if show_triangles:
                    ax.plot(tile['triangle'][:, 0], tile['triangle'][:, 1], 'k-', lw=1)
                    ax.plot(tile['ac_line'][:, 0], tile['ac_line'][:, 1], 'k-', lw=1)
                ax.plot(tile['curve1'][:, 0], tile['curve1'][:, 1], 'r-', lw=2)
                ax.plot(tile['curve2'][:, 0], tile['curve2'][:, 1], 'r-', lw=2)

    if with_frame:
        x0, y0 = 0, np.sqrt(3)
        x1, y1 = 12, np.sqrt(3)*6
        frame = [
            [(x0, y0), (x1, y0)],
            [(x1, y0), (x1, y1)],
            [(x1, y1), (x0, y1)],
            [(x0, y1), (x0, y0)],
        ]
        for (p1, p2) in frame:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', lw=2)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)

    ax.set_aspect('equal')
    ax.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1, dpi=150)
    plt.close()

# --- GUI部分 ---
class TessellationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tessellation Viewer")

        # キャンバス
        self.image_label = tk.Label(self.root)
        self.image_label.pack()

        # ボタン
        style = ttk.Style()
        style.configure("Big.TButton", font=("Helvetica", 16), padding=10)

        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)

        redraw_button = ttk.Button(button_frame, text="Redraw", style="Big.TButton", command=self.redraw)
        quit_button = ttk.Button(button_frame, text="Quit", style="Big.TButton", command=self.root.quit)
        redraw_button.pack(side=tk.LEFT, padx=10)
        quit_button.pack(side=tk.LEFT, padx=10)

        self.filename = "current_tessellation.png"
        self.redraw()  # 初期描画

    def redraw(self):
        draw_tessellation(show_triangles=False, with_frame=True, filename=self.filename)
        self.show_image()

    def show_image(self):
        img = Image.open(self.filename)

        # アスペクト比を保って最大1100x750にリサイズ
        max_width, max_height = 1100, 750
        img_width, img_height = img.size

        scale = min(max_width / img_width, max_height / img_height)
        new_size = (int(img_width * scale), int(img_height * scale))

        img = img.resize(new_size, Image.Resampling.LANCZOS)

        self.photo = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.photo)
        self.image_label.image = self.photo

# --- 実行 ---
if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1200x900")
    app = TessellationApp(root)
    root.mainloop()
