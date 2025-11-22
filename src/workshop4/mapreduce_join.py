# ------------------------------
# Shuffle simple: agrupa por clave
# ------------------------------
def shuffle(mapped):
    groups = {}
    for k, v in mapped:
        if k in groups:
            groups[k].append(v)
        else:
            groups[k] = [v]
    return groups

# ------------------------------
# Map para cada fuente
# ------------------------------
def map_pollution(records, key_field="sensorLocation"):
    """
    Map para lecturas de contaminación.
    Emite (key_field, {'type':'pollution', 'data': <registro_sin_clave>})
    """
    out = []
    for r in records:
        key = r.get(key_field)
        if key is None:
            continue
        # copia del registro sin romper el original (opcional)
        data = {}
        for kk, vv in r.items():
            if kk != key_field:
                data[kk] = vv
        out.append((key, {"type": "pollution", "data": data}))
    return out

def map_weather(records, key_field="sensorLocation"):
    """
    Map para lecturas de clima.
    Emite (key_field, {'type':'weather', 'data': <registro_sin_clave>})
    """
    out = []
    for r in records:
        key = r.get(key_field)
        if key is None:
            continue
        data = {}
        for kk, vv in r.items():
            if kk != key_field:
                data[kk] = vv
        out.append((key, {"type": "weather", "data": data}))
    return out

# ------------------------------
# Reduce: realizar el join
# ------------------------------
def reduce_join(shuffled, join_type="inner"):
    """
    join_type: 'inner' (default), 'left', 'right', 'outer'
    - inner: solo claves que tengan ambos tipos
    - left: todas las pollution, si no hay weather -> weather = None
    - right: todas las weather, si no hay pollution -> pollution = None
    - outer: todas las claves; faltantes -> None
    Retorna lista de reportes combinados: cada reporte es dict con keys:
      { 'location': clave, 'pollution': <dict or None>, 'weather': <dict or None> }
    Nota: si hay múltiples registros de cada tipo para una ubicación se hace cross-product.
    """
    reports = []

    for key, values in shuffled.items():
        polls = []
        weathers = []
        for item in values:
            if not isinstance(item, dict):
                continue
            t = item.get("type")
            d = item.get("data")
            if t == "pollution":
                polls.append(d)
            elif t == "weather":
                weathers.append(d)

        if join_type == "inner":
            if not polls or not weathers:
                continue
            # cross-product
            for p in polls:
                for w in weathers:
                    reports.append({"location": key, "pollution": p, "weather": w})
        elif join_type == "left":
            if polls:
                if weathers:
                    for p in polls:
                        for w in weathers:
                            reports.append({"location": key, "pollution": p, "weather": w})
                else:
                    for p in polls:
                        reports.append({"location": key, "pollution": p, "weather": None})
        elif join_type == "right":
            if weathers:
                if polls:
                    for p in polls:
                        for w in weathers:
                            reports.append({"location": key, "pollution": p, "weather": w})
                else:
                    for w in weathers:
                        reports.append({"location": key, "pollution": None, "weather": w})
        elif join_type == "outer":
            if polls and weathers:
                for p in polls:
                    for w in weathers:
                        reports.append({"location": key, "pollution": p, "weather": w})
            elif polls and not weathers:
                for p in polls:
                    reports.append({"location": key, "pollution": p, "weather": None})
            elif weathers and not polls:
                for w in weathers:
                    reports.append({"location": key, "pollution": None, "weather": w})
        else:
            # join_type desconocido -> tratar como inner
            if polls and weathers:
                for p in polls:
                    for w in weathers:
                        reports.append({"location": key, "pollution": p, "weather": w})

    return reports

# ------------------------------
# Función de utilidad: pipeline completo
# ------------------------------
def mapreduce_join(pollution_records, weather_records, key_field="sensorLocation", join_type="inner"):
    mapped = []
    mapped.extend(map_pollution(pollution_records, key_field=key_field))
    mapped.extend(map_weather(weather_records, key_field=key_field))
    shuffled = shuffle(mapped)
    reports = reduce_join(shuffled, join_type=join_type)
    return reports

# ------------------------------
# Ejemplo de uso
# ------------------------------
if __name__ == "__main__":
    # Datos de ejemplo (dos "datasets" distintos)
    pollution_records = [
        {"sensorLocation": "Bogota", "aqiValue": 120, "pollutionType": "PM2.5", "timestamp": "2025-11-20T10:00:00Z"},
        {"sensorLocation": "Bogota", "aqiValue": 140, "pollutionType": "NO2", "timestamp": "2025-11-20T10:05:00Z"},
        {"sensorLocation": "Medellin", "aqiValue": 80, "pollutionType": "PM2.5", "timestamp": "2025-11-20T09:50:00Z"},
        {"sensorLocation": "Cali", "aqiValue": 200, "pollutionType": "O3", "timestamp": "2025-11-20T11:00:00Z"},
    ]

    weather_records = [
        {"sensorLocation": "Bogota", "temperature": 15.2, "humidity": 70, "windSpeed": 3.4, "timestamp": "2025-11-20T10:00:30Z"},
        {"sensorLocation": "Medellin", "temperature": 22.0, "humidity": 65, "windSpeed": 1.2, "timestamp": "2025-11-20T09:50:10Z"},
        # Nota: Cali no tiene lectura de clima en este ejemplo
    ]

    # Hacemos join tipo 'inner' (solo ubicaciones con ambos)
    inner_reports = mapreduce_join(pollution_records, weather_records, join_type="inner")
    print("INNER JOIN (solo ubicaciones con pollution y weather):")
    for r in inner_reports:
        print(r)

    # Hacemos join tipo 'left' (todas las pollution; si falta weather -> None)
    left_reports = mapreduce_join(pollution_records, weather_records, join_type="left")
    print("\nLEFT JOIN (todas las pollution; weather puede ser None):")
    for r in left_reports:
        print(r)

    # Hacemos join tipo 'outer' (todas las claves)
    outer_reports = mapreduce_join(pollution_records, weather_records, join_type="outer")
    print("\nOUTER JOIN (todas las claves):")
    for r in outer_reports:
        print(r)
