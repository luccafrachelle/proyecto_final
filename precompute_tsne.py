import numpy as np
from sklearn import datasets
from sklearn.manifold import TSNE
import joblib
import os

# Crear directorio para los modelos si no existe
if not os.path.exists('tsne_models'):
    os.makedirs('tsne_models')

# Cargar datos
print("Cargando datos...")
digits = datasets.load_digits(n_class=10)
X = digits.data
y = digits.target
images = digits.images

# Normalizar imágenes para visualización
images_normalized = images / 16.0  # Normalizar a [0,1]

# Guardar datos base
print("Guardando datos base...")
np.save('tsne_models/X.npy', X)
np.save('tsne_models/y.npy', y)
np.save('tsne_models/images.npy', images_normalized)

# Parámetros reducidos para tener aproximadamente 50 casos
perplexities = [5, 15, 30, 45]
learning_rates = [50, 200, 500]
n_iters = [500, 1000]
init_methods = ["pca", "random"]

print("Entrenando y guardando modelos t-SNE...")
for perplexity in perplexities:
    for lr in learning_rates:
        for n_iter in n_iters:
            for init in init_methods:
                print(f"Entrenando: perplexity={perplexity}, lr={lr}, n_iter={n_iter}, init={init}")
                
                # Crear y entrenar modelo t-SNE
                tsne = TSNE(
                    n_components=2,
                    perplexity=perplexity,
                    learning_rate=lr,
                    n_iter=n_iter,
                    init=init,
                    random_state=42
                )
                
                # Entrenar el modelo
                tsne.fit(X)
                
                # Guardar el modelo
                model_name = f"tsne_models/p{perplexity}_lr{lr}_iter{n_iter}_{init}.joblib"
                joblib.dump(tsne, model_name)

print("¡Listo! Los modelos han sido guardados en el directorio tsne_models/") 