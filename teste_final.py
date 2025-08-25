import tensorflow as tf
import numpy as np
import cv2

# carregar modelo
modelo = tf.keras.models.load_model("modelo_final_detecta_arma.h5", compile=False)


def preprocessar_imagem(caminho_imagem):
    img = cv2.imread(caminho_imagem)                # le a imagem em BGR
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # converte para RGB (para o modelo)
    img_resized = cv2.resize(img_rgb, (224, 224))   # redimensiona para entrada da rede
    img_normalizada = img_resized / 255.0           # normaliza valores 0-1
    return img, np.expand_dims(img_normalizada, axis=0)  # retorna original + processada


caminho = "p.jpg"  


imagem_original, imagem_input = preprocessar_imagem(caminho)

# predicao
pred_bbox, pred_class = modelo.predict(imagem_input)

# probabilidade prevista para arma
probabilidade_arma = float(pred_class[0][0])
bbox = pred_bbox[0]  

print(f"Probabilidade de arma: {probabilidade_arma:.2f}")

# desenhar bounding box 
if probabilidade_arma >= 0.88:  # threshold
    h, w, _ = imagem_original.shape
    x_min = int(bbox[0] * w)
    y_min = int(bbox[1] * h)
    x_max = int(bbox[2] * w)
    y_max = int(bbox[3] * h)

    cv2.rectangle(imagem_original, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
    cv2.putText(imagem_original, f"Arma: {probabilidade_arma:.2f}",
                (x_min, max(y_min - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 0, 0), 2)
else:
    cv2.putText(imagem_original, "Sem arma", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

# redimensionar img
largura_max = 1200
altura_max = 1200
h, w, _ = imagem_original.shape
escala = min(largura_max / w, altura_max / h)
nova_largura = int(w * escala)
nova_altura = int(h * escala)
imagem_redimensionada = cv2.resize(imagem_original, (nova_largura, nova_altura))


cv2.imshow("Resultado", imagem_redimensionada)
cv2.waitKey(0)
cv2.destroyAllWindows()
