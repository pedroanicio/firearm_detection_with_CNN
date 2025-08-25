import numpy as np
import matplotlib.pyplot as plt
import PIL.ImageDraw as ImageDraw
from PIL import Image

X = np.load('imagens_X.npy')
y = np.load('labels_y.npy')

def desenhar_bbox(imagem, bbox):
    x1 = int(bbox[0] * 224)
    y1 = int(bbox[1] * 224)
    x2 = int(bbox[2] * 224)
    y2 = int(bbox[3] * 224)

    imagem_com_box = (imagem * 255).astype(np.uint8)
    pil_img = Image.fromarray(imagem_com_box)
    draw = ImageDraw.Draw(pil_img)
    draw.rectangle([x1, y1, x2, y2], outline='green', width=2)

    return np.array(pil_img)

for i in range(len(X)):
    imagem = X[i]
    bbox = y[i]

    if bbox[4] == 1:
        imagem_box = desenhar_bbox(imagem, bbox)
        classe = "arma"
    else:
        imagem_box = (imagem * 255).astype(np.uint8)
        classe = "sem arma"

    plt.figure(figsize=(4, 4))
    plt.imshow(imagem_box)
    plt.title(f"Classe: {classe}")
    plt.axis("off")
    plt.show()
