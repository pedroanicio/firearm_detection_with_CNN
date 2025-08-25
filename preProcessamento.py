import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm

#path
COM_ARMA = 'dataset/com_arma'
SEM_ARMA = 'dataset/sem_arma'

# image size
SIZE = 224

X = []
Y = []

def processa_img_e_xml(img, xml): 

    get_img = cv2.imread(img)
    if get_img is None:
        return None, None

    # dimensões da imagem
    altura, largura = get_img.shape[:2]

    #redimensionar (224x224)
    img_redimensionada = cv2.resize(get_img, (SIZE, SIZE))
    img_redimensionada = img_redimensionada / 255

    #xml
    tree = ET.parse(xml)
    root = tree.getroot()
    object = root.find('object')
    if object is None:
        return None, None
    
    
    bndbox = object.find('bndbox')
    xmin = int(bndbox.find('xmin').text)
    ymin = int(bndbox.find('ymin').text)
    xmax = int(bndbox.find('xmax').text)
    ymax = int(bndbox.find('ymax').text)

    # Normalizar coordenadas
    x_min_norm = xmin / largura
    y_min_norm = ymin / altura
    x_max_norm = xmax / largura
    y_max_norm = ymax / altura

    bbox = [x_min_norm, y_min_norm, x_max_norm, y_max_norm, 1]  # Classe 1 = com arma

    return img_redimensionada, bbox

def processar_imagens():
    # processar imagens com arma
    for filename in tqdm(os.listdir(COM_ARMA), desc="Imagens com arma"):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            nome_base = os.path.splitext(filename)[0]
            caminho_img = os.path.join(COM_ARMA, filename)
            caminho_xml = os.path.join(COM_ARMA, nome_base + '.xml')

            if os.path.exists(caminho_xml):
                imagem, bbox = processa_img_e_xml(caminho_img, caminho_xml)
                if imagem is not None:
                    X.append(imagem)
                    Y.append(bbox)

    #processar imagens sem arma
    for filename in tqdm(os.listdir(SEM_ARMA), desc="Imagens sem arma"):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            caminho_img = os.path.join(SEM_ARMA, filename)
            img = cv2.imread(caminho_img)
            if img is None:
                continue

            img_resized = cv2.resize(img, (SIZE, SIZE))
            img_resized = img_resized / 255.0
            bbox_vazia = [0, 0, 0, 0, 0]  # Nada detectado, classe 0

            X.append(img_resized)
            Y.append(bbox_vazia)

    return np.array(X), np.array(Y)

x_redimensionado, y_redimensionado = processar_imagens()

np.save('imagens_X.npy', x_redimensionado)
np.save('labels_y.npy', y_redimensionado)

print("Pré processamento finalizado")