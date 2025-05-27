import json

# Cargar los resultados
with open('tsne_results.json', 'r') as f:
    results = json.load(f)

# Extraer y ordenar las configuraciones
configs = []
for key in results.keys():
    # Parsear la clave que tiene el formato "p{perplexity}_lr{learning_rate}_iter{n_iter}_{init}"
    parts = key.split('_')
    perplexity = int(parts[0][1:])
    lr = int(parts[1][2:])
    n_iter = int(parts[2][4:])
    init = parts[3]
    
    configs.append({
        'perplexity': perplexity,
        'learning_rate': lr,
        'n_iter': n_iter,
        'init': init
    })

# Ordenar las configuraciones
configs.sort(key=lambda x: (x['perplexity'], x['learning_rate'], x['n_iter'], x['init']))

# Imprimir las configuraciones disponibles
print("\nConfiguraciones disponibles:")
print("-" * 80)
print(f"{'Perplexity':<10} {'Learning Rate':<15} {'Iteraciones':<12} {'Inicialización':<15}")
print("-" * 80)

for config in configs:
    print(f"{config['perplexity']:<10} {config['learning_rate']:<15} {config['n_iter']:<12} {config['init']:<15}")

print("\nTotal de configuraciones:", len(configs)) 