import streamlit as st
import numpy as np
# import matplotlib.pyplot as plt # No necesario si solo usamos plotly
from sklearn import datasets
from sklearn.manifold import TSNE # Necesario para definir TSNEWithHistory si se vuelve a necesitar, o solo para referencia
import pandas as pd
import json # Importar json para leer las configuraciones
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import spearmanr
import joblib
import os
from sklearn.metrics.pairwise import euclidean_distances
# import scipy.stats as stats # No necesario si eliminamos la parte teórica
import umap
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import plotly.figure_factory as ff

# Configuración de la página
st.set_page_config(
    page_title="Visualización Interactiva t-SNE/UMAP",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Parámetros para precalcular
TSNE_PARAMS = {
    'perplexity': 30,
    'learning_rate': 200.0,
    'n_iter': 500,
    'init': 'random',
    'random_state': 42
}

UMAP_PARAMS = {
    'n_neighbors': 15,
    'min_dist': 0.1,
    'n_components': 2,
    'random_state': 42
}

OUTPUT_DIR = 'precomputed_embeddings'

# Funciones de caché y carga de datos
@st.cache_data
def load_data():
    digits = datasets.load_digits(n_class=10)
    return digits.data, digits.target, digits.images

@st.cache_data
def load_precomputed_configs():
    tsne_configs_path = os.path.join(OUTPUT_DIR, 'tsne_configs.json')
    umap_configs_path = os.path.join(OUTPUT_DIR, 'umap_configs.json')
    
    if not os.path.exists(tsne_configs_path) or not os.path.exists(umap_configs_path):
        st.error("Error: No se encontraron los archivos de configuración precalculados.")
        st.stop()
    
    with open(tsne_configs_path, 'r') as f:
        tsne_configs = json.load(f)
    with open(umap_configs_path, 'r') as f:
        umap_configs = json.load(f)
    
    return tsne_configs, umap_configs

@st.cache_data
def load_embedding(algo_type, params):
    if algo_type == 'tsne':
        perplexity = params['perplexity']
        learning_rate = params['learning_rate']
        filename = f"tsne_p{perplexity}_lr{learning_rate:.0f}.joblib".replace(".", "p")
    elif algo_type == 'umap':
        n_neighbors = params['n_neighbors']
        min_dist = params['min_dist']
        filename = f"umap_nn{n_neighbors}_md{min_dist}".replace(".", "p") + ".joblib"
    else:
        st.error("Tipo de algoritmo desconocido.")
        return None
    
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    if not os.path.exists(filepath):
        st.warning(f"El embedding precalculado para {algo_type} con estos parámetros no se encontró.")
        return None
        
    return joblib.load(filepath)

# Cargar datos
X, y, images = load_data()
tsne_configs, umap_configs = load_precomputed_configs()

# Extraer valores únicos para los sliders
tsne_perplexities = sorted(list(set([c['perplexity'] for c in tsne_configs])))
tsne_learning_rates = sorted(list(set([c['learning_rate'] for c in tsne_configs])))
umap_n_neighbors = sorted(list(set([c['n_neighbors'] for c in umap_configs])))
umap_min_dist = sorted(list(set([c['min_dist'] for c in umap_configs])))

# Crear pestañas
tab1, tab2 = st.tabs(["t-SNE", "UMAP"])

# Inicializar el estado de la sesión si no existe
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "t-SNE"

# Función para mostrar los parámetros de t-SNE (ahora en el cuerpo principal)
def show_tsne_params():
    st.header("Parámetros t-SNE")
    
    # Slider para perplexity
    selected_perplexity = st.select_slider(
        "Perplexity",
        options=tsne_perplexities,
        value=30,
        help="Controla el balance entre la estructura local y global"
    )
    
    # Slider para learning rate
    selected_learning_rate = st.select_slider(
        "Learning Rate",
        options=tsne_learning_rates,
        value=200.0,
        help="Controla la velocidad de convergencia"
    )
    
    # Parámetros fijos
    st.markdown("**Parámetros Fijos:**")
    st.text(f"Iteraciones: {tsne_configs[0]['n_iter']}")
    st.text(f"Inicialización: {tsne_configs[0]['init']}")
    
    # Métricas de calidad
    st.markdown("---")
    st.markdown("**Métricas de Calidad:**")
    show_metrics = st.checkbox("Mostrar métricas detalladas", value=False, key="tsne_metrics_tab") # Cambiar key
    
    return selected_perplexity, selected_learning_rate, show_metrics

# Función para mostrar los parámetros de UMAP (ahora en el cuerpo principal)
def show_umap_params():
    st.header("Parámetros UMAP")
    
    # Slider para n_neighbors
    selected_n_neighbors = st.select_slider(
        "Número de Vecinos",
        options=umap_n_neighbors,
        value=15,
        help="Controla el balance entre estructura local y global"
    )
    
    # Slider para min_dist
    selected_min_dist = st.select_slider(
        "Distancia Mínima",
        options=umap_min_dist,
        value=0.1,
        help="Controla qué tan compactos son los clusters"
    )
    
    # Parámetros fijos
    st.markdown("**Parámetros Fijos:**")
    st.text(f"Componentes: {umap_configs[0]['n_components']}")
    
    # Métricas de calidad
    st.markdown("---")
    st.markdown("**Métricas de Calidad:**")
    show_metrics = st.checkbox("Mostrar métricas detalladas", value=False, key="umap_metrics_tab") # Cambiar key
    
    return selected_n_neighbors, selected_min_dist, show_metrics

# Pestaña t-SNE
with tab1:
    st.session_state.active_tab = "t-SNE"
    st.title("Visualización t-SNE") # Título principal
    
    # Crear columnas para la visualización y los parámetros
    col_plot, col_params = st.columns([3, 1]) # Proporción 3:1 para el gráfico vs parámetros
    
    with col_params:
        # Mostrar los parámetros de t-SNE en la columna de parámetros
        st.header("Parámetros")
        selected_perplexity, selected_learning_rate, show_metrics = show_tsne_params() # Llama a la función sin st.sidebar
        
    with col_plot:
        # Cargar embedding (necesario antes de la visualización)
        selected_tsne_params = {
            'perplexity': selected_perplexity,
            'learning_rate': selected_learning_rate,
            'n_iter': tsne_configs[0]['n_iter'],
            'init': tsne_configs[0]['init'],
            'random_state': 42
        }
        X_tsne = load_embedding('tsne', selected_tsne_params)
        
        if X_tsne is not None:
            # Visualización principal (mostrar en la columna de gráfico)
            df_tsne = pd.DataFrame(X_tsne, columns=['x', 'y'])
            df_tsne['digito'] = y.astype(str)
            
            fig = px.scatter(
                df_tsne,
                x='x',
                y='y',
                color='digito',
                title="t-SNE Embedding", # Título para el gráfico
                labels={'x': 'Componente 1', 'y': 'Componente 2'},
                color_discrete_sequence=px.colors.qualitative.Set1, # Usar escala discreta
                height=600 # Aumentar la altura del gráfico
            )
            
            fig.update_traces(marker=dict(size=8))
            fig.update_layout(
                title_x=0.5,
                showlegend=True,
                legend_title="Dígito"
            )
            st.plotly_chart(fig, use_container_width=True) # use_container_width=True ya lo hace ancho
    
    st.markdown("--- - ---") # Separador debajo de las columnas
    
    # Métricas de calidad (mostrar debajo de las columnas)
    if show_metrics:
        st.subheader("Métricas de Calidad t-SNE") # Subtítulo para métricas
        col_metrics = st.columns(3) # Columnas para las métricas
        
        distances_original = euclidean_distances(X)
        distances_tsne = euclidean_distances(X_tsne)
        correlation, p_value = spearmanr(distances_original.flatten(), distances_tsne.flatten())
        
        with col_metrics[0]:
            st.metric("Correlación de Spearman", f"{correlation:.4f}")
        
        with col_metrics[1]:
            silhouette = silhouette_score(X_tsne, y)
            st.metric("Silhouette Score", f"{silhouette:.4f}")
        
        with col_metrics[2]:
            calinski = calinski_harabasz_score(X_tsne, y)
            st.metric("Calinski-Harabasz", f"{calinski:.4f}")
        
        # Matriz de distancias
        st.subheader("Matriz de Distancias Euclidianas (primeras 50 muestras)")
        fig_dist = go.Figure(data=go.Heatmap(
            z=distances_tsne[:50, :50],
            colorscale='Viridis',
            showscale=True,
            hoverongaps=False,
            hoverinfo='z'
        ))
        fig_dist.update_layout(
            title="", # Título vacío, el subtítulo ya lo indica
            xaxis_title="Muestra",
            yaxis_title="Muestra",
            height=400 # Reducir altura
        )
        st.plotly_chart(fig_dist, use_container_width=True)

# Pestaña UMAP
with tab2:
    st.session_state.active_tab = "UMAP"
    st.title("Visualización UMAP") # Título principal
    
    # Crear columnas para la visualización y los parámetros
    col_plot, col_params = st.columns([3, 1]) # Proporción 3:1 para el gráfico vs parámetros
    
    with col_params:
        # Mostrar los parámetros de UMAP en la columna de parámetros
        st.header("Parámetros")
        selected_n_neighbors, selected_min_dist, show_metrics = show_umap_params() # Llama a la función sin st.sidebar

    with col_plot:
        # Cargar embedding (necesario antes de la visualización)
        selected_umap_params = {
            'n_neighbors': selected_n_neighbors,
            'min_dist': selected_min_dist,
            'n_components': 2,
            'random_state': 42
        }
        X_umap = load_embedding('umap', selected_umap_params)

        if X_umap is not None:
            # Visualización principal (mostrar en la columna de gráfico)
            df_umap = pd.DataFrame(X_umap, columns=['x', 'y'])
            df_umap['digito'] = y.astype(str)

            fig = px.scatter(
                df_umap,
                x='x',
                y='y',
                color='digito',
                title="UMAP Embedding", # Título para el gráfico
                labels={'x': 'Componente 1', 'y': 'Componente 2'},
                color_discrete_sequence=px.colors.qualitative.Set1, # Usar escala discreta
                height=600 # Aumentar la altura del gráfico
            )

            fig.update_traces(marker=dict(size=8))
            fig.update_layout(
                title_x=0.5,
                showlegend=True,
                legend_title="Dígito"
            )
            st.plotly_chart(fig, use_container_width=True) # use_container_width=True ya lo hace ancho
            
    st.markdown("--- - ---") # Separador debajo de las columnas
    
    # Métricas de calidad (mostrar debajo de las columnas)
    if show_metrics:
        st.subheader("Métricas de Calidad UMAP") # Subtítulo para métricas
        col_metrics = st.columns(3) # Columnas para las métricas
        
        distances_original = euclidean_distances(X)
        distances_umap = euclidean_distances(X_umap)
        correlation, p_value = spearmanr(distances_original.flatten(), distances_umap.flatten())
        
        with col_metrics[0]:
            st.metric("Correlación de Spearman", f"{correlation:.4f}")
        
        with col_metrics[1]:
            silhouette = silhouette_score(X_umap, y)
            st.metric("Silhouette Score", f"{silhouette:.4f}")
        
        with col_metrics[2]:
            calinski = calinski_harabasz_score(X_umap, y)
            st.metric("Calinski-Harabasz", f"{calinski:.4f}")
        
        # Matriz de distancias
        st.subheader("Matriz de Distancias Euclidianas (primeras 50 muestras)")
        fig_dist = go.Figure(data=go.Heatmap(
            z=distances_umap[:50, :50],
            colorscale='Viridis',
            showscale=True,
            hoverongaps=False,
            hoverinfo='z'
        ))
        fig_dist.update_layout(
            title="", # Título vacío, el subtítulo ya lo indica
            xaxis_title="Muestra",
            yaxis_title="Muestra",
            height=400 # Reducir altura
        )
        st.plotly_chart(fig_dist, use_container_width=True)

# Información del dataset (en la barra lateral)
st.sidebar.markdown("---")
st.sidebar.markdown(f"""
**Información del Dataset:**
- Muestras: {X.shape[0]}
- Dimensiones originales: {X.shape[1]}
- Dimensiones reducidas: 2
""")

# Información adicional sobre los parámetros precalculados (en la barra lateral footer)
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Nota sobre Parámetros:**
Los parámetros seleccionables corresponden a los valores precalculados.
Otros parámetros (como n_iter para t-SNE) se mantuvieron fijos durante el precálculo.
""") 