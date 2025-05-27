import numpy as np
from sklearn import datasets
from sklearn.manifold import TSNE
import umap
import joblib
import os
import json

# Definir las combinaciones de parámetros a precalcular
TSNE_PARAM_GRID = {
    'perplexity': [5, 15, 30, 50],
    'learning_rate': [100.0, 200.0, 500.0],
    'n_iter': [500], # Mantener n_iter e init fijos por ahora para simplificar el grid
    'init': ['random'],
    'random_state': [42]
}

UMAP_PARAM_GRID = {
    'n_neighbors': [5, 15, 30, 50],
    'min_dist': [0.01, 0.1, 0.5],
    'n_components': [2],
    'random_state': [42]
}

OUTPUT_DIR = 'precomputed_embeddings'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Cargando datos...")
digits = datasets.load_digits(n_class=10)
X = digits.data
y = digits.target
print(f"Datos cargados. Dimensiones: {X.shape}")

# --- Precalcular t-SNE ---15
print("Precalculando t-SNE para diferentes combinaciones...")
tsne_configs = []
for perplexity in TSNE_PARAM_GRID['perplexity']:
    for learning_rate in TSNE_PARAM_GRID['learning_rate']:
        # Usar valores fijos para n_iter, init y random_state definidos en el grid
        tsne_params = {
            'perplexity': perplexity,
            'learning_rate': learning_rate,
            'n_iter': TSNE_PARAM_GRID['n_iter'][0], # Tomar el primer (único) valor
            'init': TSNE_PARAM_GRID['init'][0],   # Tomar el primer (único) valor
            'random_state': TSNE_PARAM_GRID['random_state'][0] # Tomar el primer (único) valor
        }
        print(f"  Calculando t-SNE con parámetros: {tsne_params}")
        try:
            tsne = TSNE(**tsne_params, n_jobs=-1)
            X_tsne = tsne.fit_transform(X)
            
            # Generar nombre de archivo basado en parámetros
            filename = f"tsne_p{perplexity}_lr{learning_rate:.0f}.joblib".replace(".", "p") # Reemplazar punto en lr flotante
            output_path = os.path.join(OUTPUT_DIR, filename)
            joblib.dump(X_tsne, output_path)
            print(f"  Resultados guardados en {output_path}")
            
            # Guardar la configuración usada
            tsne_configs.append(tsne_params)
            
        except Exception as e:
            print(f"  Error calculando t-SNE para {tsne_params}: {e}")

# Guardar las configuraciones de t-SNE usadas
tsne_configs_path = os.path.join(OUTPUT_DIR, 'tsne_configs.json')
with open(tsne_configs_path, 'w') as f:
    json.dump(tsne_configs, f, indent=4)
print(f"Configuraciones t-SNE guardadas en {tsne_configs_path}")

# --- Precalcular UMAP ---56
print("\nPrecalculando UMAP para diferentes combinaciones...")
umap_configs = []
for n_neighbors in UMAP_PARAM_GRID['n_neighbors']:
    for min_dist in UMAP_PARAM_GRID['min_dist']:
        # Usar valores fijos para n_components y random_state definidos en el grid
        umap_params = {
            'n_neighbors': n_neighbors,
            'min_dist': min_dist,
            'n_components': UMAP_PARAM_GRID['n_components'][0], # Tomar el primer (único) valor
            'random_state': UMAP_PARAM_GRID['random_state'][0] # Tomar el primer (único) valor
        }
        print(f"  Calculando UMAP con parámetros: {umap_params}")
        try:
            umap_reducer = umap.UMAP(**umap_params)
            X_umap = umap_reducer.fit_transform(X)

            # Generar nombre de archivo basado en parámetros
            filename = f"umap_nn{n_neighbors}_md{min_dist}".replace(".", "p") + ".joblib"
            output_path = os.path.join(OUTPUT_DIR, filename)
            joblib.dump(X_umap, output_path)
            print(f"  Resultados guardados en {output_path}")

            # Guardar la configuración usada
            umap_configs.append(umap_params)
            
        except Exception as e:
            print(f"  Error calculando UMAP para {umap_params}: {e}")

# Guardar las configuraciones de UMAP usadas
umap_configs_path = os.path.join(OUTPUT_DIR, 'umap_configs.json')
with open(umap_configs_path, 'w') as f:
    json.dump(umap_configs, f, indent=4)
print(f"Configuraciones UMAP guardadas en {umap_configs_path}")

print("\nPrecálculo de todas las combinaciones completado.") 