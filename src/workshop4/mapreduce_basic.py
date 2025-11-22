# ------------------------------
# Shuffle (agrupación) simple
# ------------------------------
def shuffle(mapped):
    """
    Agrupa una lista de tuplas (clave, valor) en un diccionario:
    { clave: [valores,...], ... }
    """
    groups = {}
    for k, v in mapped:
        if k in groups:
            groups[k].append(v)
        else:
            groups[k] = [v]
    return groups

# ------------------------------
# Algoritmo 1: Contador de Lecturas
# ------------------------------
def map_count_reads(records):
    """
    Map: por cada registro emite (ubicacion_sensor, 1)
    Usa 'sensorLocation' como ubicación.
    """
    out = []
    for r in records:
        loc = r.get("sensorLocation")
        if loc is not None:
            out.append((loc, 1))
    return out

def reduce_count_reads(shuffled):
    """
    Reduce: suma los 1s por ubicación -> total de lecturas.
    Devuelve un diccionario {ubicacion: total}
    """
    totals = {}
    for k, values in shuffled.items():
        s = 0
        for val in values:
            s += val
        totals[k] = s
    return totals

def top_n_sensors(counts, n=10):
    """
    Devuelve una lista con las n ubicaciones con más lecturas:
    [(ubicacion, conteo), ...]
    """
    items = list(counts.items())
    items.sort(key=lambda kv: kv[1], reverse=True)
    return items[:n]

# ------------------------------
# Algoritmo 2: Promedio de Contaminación (AQI)
# ------------------------------
def map_avg_aqi(records):
    """
    Map: por cada registro emite (ciudad, aqiValue)
    Usa 'sensorLocation' y 'aqiValue'.
    """
    out = []
    for r in records:
        loc = r.get("sensorLocation")
        aqi = r.get("aqiValue")
        if loc is not None and (isinstance(aqi, int) or isinstance(aqi, float)):
            out.append((loc, float(aqi)))
    return out

def reduce_avg_aqi(shuffled):
    """
    Reduce: calcula promedio por ciudad.
    Devuelve {ciudad: promedio}
    """
    averages = {}
    for k, vals in shuffled.items():    # Bucle (k, vals), se gestiona por las keys 
        total = 0.0
        count = 0
        for v in vals:        # Bucle para recorrer todos los valores 'vals' de la key 'k'
            total += v
            count += 1
        if count > 0:
            averages[k] = total / count
        else:
            averages[k] = float('nan')
    return averages

def city_with_max_min(averages):
    """
    Retorna ((ciudad_max, val), (ciudad_min, val)).
    Si averages está vacío retorna (("N/A", nan), ("N/A", nan)).
    """
    if not averages:
        return (("N/A", float('nan')), ("N/A", float('nan')))
    items = list(averages.items())
    # Encontrar max y min manualmente
    max_item = items[0]
    min_item = items[0]
    for it in items[1:]:
        if it[1] > max_item[1]:
            max_item = it
        if it[1] < min_item[1]:
            min_item = it
    return max_item, min_item

# ------------------------------
# Ejemplo de uso
# ------------------------------
if __name__ == "__main__":
    # Datos de ejemplo (simulan los registros del JSON que enviaste)
    records = [
        {"sensorLocation": "Bogota", "aqiValue": 120},
        {"sensorLocation": "Bogota", "aqiValue": 130},
        {"sensorLocation": "Medellin", "aqiValue": 80},
        {"sensorLocation": "Cali", "aqiValue": 200},
        {"sensorLocation": "Medellin", "aqiValue": 90},
        {"sensorLocation": "Cali", "aqiValue": 150},
        {"sensorLocation": "Bogota", "aqiValue": 100},
        {"sensorLocation": "Cali", "aqiValue": 220},
        {"sensorLocation": "Bogota"},   # sin aqiValue -> ignorado por map_avg_aqi
        {"aqiValue": 50},               # sin sensorLocation -> ignorado
    ]

    # Algoritmo 1: contador de lecturas
    mapped1 = map_count_reads(records)
    shuffled1 = shuffle(mapped1)
    counts = reduce_count_reads(shuffled1)
    print("Conteo de lecturas por ubicación:")
    for loc, c in counts.items():
        print(f"  {loc}: {c} lecturas")
    print("Top sensores (más activos):", top_n_sensors(counts, n=3))

    # Algoritmo 2: promedio AQI por ciudad
    mapped2 = map_avg_aqi(records)
    shuffled2 = shuffle(mapped2)
    averages = reduce_avg_aqi(shuffled2)
    print("\nPromedio AQI por ciudad:")
    for loc, a in averages.items():
        print(f"  {loc}: {a:.2f}")
    max_city, min_city = city_with_max_min(averages)
    print(f"\nCiudad con mayor contaminación (promedio AQI): {max_city[0]} -> {max_city[1]:.2f}")
    print(f"Ciudad con menor contaminación (promedio AQI): {min_city[0]} -> {min_city[1]:.2f}")
