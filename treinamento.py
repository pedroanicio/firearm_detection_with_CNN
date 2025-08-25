import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.callbacks import ModelCheckpoint, EarlyStopping


try:
    from sklearn.model_selection import train_test_split
    PERMISSAO_SKLEARN = True
except:
    PERMISSAO_SKLEARN = False

# carregar dados pre processados
X = np.load('imagens_X.npy')  # (N, 224, 224, 3) normalizado
y = np.load('labels_y.npy')   # (N, 5) -> [x1, y1, x2, y2, classe]

print("Shape X:", X.shape)
print("Shape y:", y.shape)

X = X.astype('float32')
y = y.astype('float32')

y_bbox = y[:, :4]
y_class = y[:, 4:]

VAL_SPLIT = 0.2

if PERMISSAO_SKLEARN:
    X_train, X_val, y_bbox_train, y_bbox_val, y_class_train, y_class_val = train_test_split(
        X, y_bbox, y_class, test_size=VAL_SPLIT, random_state=42, shuffle=True
    )
else:
    N = X.shape[0]
    idx = np.arange(N)
    np.random.seed(42)
    np.random.shuffle(idx)
    split_at = int(N * (1 - VAL_SPLIT))
    train_idx = idx[:split_at]
    val_idx = idx[split_at:]
    X_train, X_val = X[train_idx], X[val_idx]
    y_bbox_train, y_bbox_val = y_bbox[train_idx], y_bbox[val_idx]
    y_class_train, y_class_val = y_class[train_idx], y_class[val_idx]

print("Treino X:", X_train.shape, "Val X:", X_val.shape)

# Criar datasets 
BATCH_SIZE = 16

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, {'bbox': y_bbox_train, 'class': y_class_train}))
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, {'bbox': y_bbox_val, 'class': y_class_val}))

# criar variações das imagens como cor, saturação, contraste para "aumentar" a variedade da base de dados
def augment(image, labels):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.1)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    image = tf.image.random_saturation(image, 0.9, 1.1)

    return image, labels

# aplica augmentation só no treino
train_dataset = train_dataset.map(augment).shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# modelo de CNN
def criar_modelo(input_shape=(224,224,3)):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, (3,3), padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(128, (3,3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    bbox_output = layers.Dense(4, activation='sigmoid', name='bbox')(x)
    class_output = layers.Dense(1, activation='sigmoid', name='class')(x)

    return keras.Model(inputs=inputs, outputs=[bbox_output, class_output])

model = criar_modelo()
model.summary()

#compilar
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss={'bbox': keras.losses.Huber(), 'class': keras.losses.BinaryCrossentropy()},
    loss_weights={'bbox': 2.0, 'class': 1.0},
    metrics={'class': ['accuracy']}
)

# Callbacks
checkpoint_cb = ModelCheckpoint(
    'melhor_modelo.h5',  
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)
earlystop_cb = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

# treinamento
history = model.fit(
    train_dataset,
    epochs=50,
    validation_data=val_dataset,
    callbacks=[checkpoint_cb, earlystop_cb]
)

# salvar modelo final
model.save('modelo_final_detecta_arma.h5')  

# Teste rapido 
best_model = keras.models.load_model('melhor_modelo.h5')
img_ex = X_val[0]
gt_bbox = y_bbox_val[0]
gt_class = y_class_val[0]

img_batch = np.expand_dims(img_ex, axis=0)
pred_bbox, pred_class = best_model.predict(img_batch)
pred_bbox = pred_bbox[0]
pred_class = pred_class[0][0]

x1 = int(pred_bbox[0] * 224)
y1 = int(pred_bbox[1] * 224)
x2 = int(pred_bbox[2] * 224)
y2 = int(pred_bbox[3] * 224)

print("GT class:", float(gt_class[0]), "GT bbox:", gt_bbox)
print("Pred class prob:", float(pred_class), "Pred bbox (normalized):", pred_bbox)
print("Pred bbox (pixels):", (x1, y1, x2, y2))
