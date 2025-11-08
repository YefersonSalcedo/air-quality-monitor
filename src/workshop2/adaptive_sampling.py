import json
import random
from collections import defaultdict
from datetime import datetime
import math


class AdaptiveSampler:
    """Muestreador adaptativo basado en comportamiento.

    - base_rate: tasa base de muestreo (antes de aplicar pesos adaptativos)
    - min_rate / max_rate: límites en la probabilidad final
    - pollutant_thresholds: umbrales para normalizar niveles (diccionario)
    - pollutant_type_weights: pesos base por tipo de contaminante
    - city_weights: pesos adaptativos por ciudad (se pueden actualizar desde FrequencyMomentsAnalyzer)
    """

    def __init__(self, base_rate=0.1, min_rate=0.01, max_rate=0.9, seed=None,
                 pollutant_thresholds=None, pollutant_type_weights=None):
        if seed is not None:
            random.seed(seed)
        self.base_rate = float(base_rate)
        self.min_rate = float(min_rate)
        self.max_rate = float(max_rate)
        # umbrales por defecto (valores usados para normalizar y estimar severidad)
        self.pollutant_thresholds = pollutant_thresholds or {
            'PM25': 35.0,
            'NO2': 100.0,
            'CO': 9.0,
        }
        # pesos base por tipo de contaminante (importancia relativa)
        self.pollutant_type_weights = pollutant_type_weights or {
            'PM2.5': 1.2,
            'PM25': 1.2,
            'NO2': 1.0,
            'CO': 0.8,
        }
        # pesos por ciudad adaptativos (inicialmente 1.0)
        self.city_weights = defaultdict(lambda: 1.0)

    def _to_float(self, v):
        try:
            return float(v)
        except Exception:
            return None

    def compute_alert_severity(self, record):
        """Devuelve un escalar >=0 que indica severidad estimada de la lectura.
        Se combina: valor relativo al umbral, si hubo alertIssued (boole), y peso por tipo.
        """
        aq = record.get('airQualityData') or {}
        pollutant_type = record.get('pollutionType') or ''
        # preferir campo PM25 o PM2.5
        pm = self._to_float(aq.get('PM25') if 'PM25' in aq else aq.get('PM2.5'))
        no2 = self._to_float(aq.get('NO2'))
        co = self._to_float(aq.get('CO') if 'CO' in aq else None)

        severity = 0.0
        # contribución normalizada de cada contaminante
        if pm is not None:
            thr = self.pollutant_thresholds.get('PM25', 35.0)
            severity += max(0.0, (pm / thr))
        if no2 is not None:
            thr = self.pollutant_thresholds.get('NO2', 100.0)
            severity += max(0.0, (no2 / thr))
        if co is not None:
            thr = self.pollutant_thresholds.get('CO', 9.0)
            severity += max(0.0, (co / thr))

        # multiplicador si la lectura ya tiene una alerta emitida
        if record.get('alertIssued'):
            severity *= 1.5

        # pesos por tipo de contaminación (si aplican)
        type_weight = 1.0
        if pollutant_type in self.pollutant_type_weights:
            type_weight = self.pollutant_type_weights[pollutant_type]
        severity *= type_weight

        # asegurar no-negatividad
        if severity < 0:
            severity = 0.0
        return severity

    def compute_weight_for_record(self, record):
        """Combina severidad y peso por ciudad para producir un peso adaptativo usado en el muestreo."""
        severity = self.compute_alert_severity(record)
        city = record.get('sensorLocation') or 'unknown'
        city_w = float(self.city_weights.get(city, 1.0))
        # función de combinación: sqrt para atenuar grandes severidades y ciudad multiplicativa
        weight = (math.sqrt(severity) if severity > 0 else 1.0) * city_w
        # asegurar mínimo
        if weight <= 0:
            weight = 1.0
        return weight

    def get_sampling_probability(self, record):
        """Devuelve la probabilidad final de muestreo para la lectura.
        p = clamp(base_rate * normalized_weight, min_rate, max_rate)
        Normalizamos weight usando una función sigmoide para mantener p en rango razonable.
        """
        raw_w = self.compute_weight_for_record(record)
        # normalizar con una sigmoide suave para evitar explosiones
        # sig(x) = 1 / (1 + e^{-k*(x-1)}) ; usar k pequeño para suavizar
        k = 0.7
        sig = 1.0 / (1.0 + math.exp(-k * (raw_w - 1.0)))
        # combinar con base rate: elevar sig para darle dinamismo
        p = self.base_rate + (self.base_rate * sig * 2.0)
        # limitar
        if p < self.min_rate:
            p = self.min_rate
        if p > self.max_rate:
            p = self.max_rate
        return float(p)

    def should_sample(self, record):
        """Decisión aleatoria de muestreo según la probabilidad calculada."""
        p = self.get_sampling_probability(record)
        return random.random() < p

    def update_city_weight(self, city, multiplier):
        """Actualiza el peso adaptativo de una ciudad multiplicándolo por multiplier.
        Use valores >1 para aumentar su importancia, <1 para disminuir.
        """
        if city is None:
            return
        self.city_weights[city] = float(self.city_weights.get(city, 1.0)) * float(multiplier)
        # opcional: limitar rango razonable
        if self.city_weights[city] < 0.1:
            self.city_weights[city] = 0.1
        if self.city_weights[city] > 10.0:
            self.city_weights[city] = 10.0


class FrequencyMomentsAnalyzer:
    """Analizador de frecuencia de alertas (Momento 1) y generador de patrones temporales.

    - critical_thresholds: umbrales usados para considerar una lectura como alerta crítica
      (si 'alertIssued' True o si un contaminante excede su umbral crítico)

    - internal counters:
      city_counts: conteo absoluto de alertas críticas por ciudad (Momento 1)
      time_buckets[city][bucket] = conteo por intervalo de tiempo (e.g., por hora)
      alert_log: lista de alertas (regs) detectadas
    """

    def __init__(self, critical_thresholds=None):
        self.critical_thresholds = critical_thresholds or {
            'PM25': 150.0,   # ejemplo: nivel crítico alto
            'NO2': 200.0,
            'CO': 15.0,
        }
        self.city_counts = defaultdict(int)
        # time_buckets: city -> bucket_key -> count
        self.time_buckets = defaultdict(lambda: defaultdict(int))
        self.alert_log = []

    def _parse_time_bucket(self, ts, window='hour'):
        """Convierte timestamp a una llave de bucket según ventana: 'hour','day','minute'.
        Acepta datetimes o strings.
        """
        if ts is None:
            return 'unknown'
        if isinstance(ts, str):
            # intentar ISO variantes
            for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%S %z", "%Y-%m-%dT%H:%M:%S"):
                try:
                    t = datetime.strptime(ts, fmt)
                    ts = t
                    break
                except Exception:
                    continue
            # si no se pudo parsear, intentar fromisoformat
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                except Exception:
                    return 'unknown'
        if not isinstance(ts, datetime):
            return 'unknown'
        if window == 'hour':
            return ts.strftime('%Y-%m-%dT%H:00:00')
        if window == 'day':
            return ts.strftime('%Y-%m-%d')
        if window == 'minute':
            return ts.strftime('%Y-%m-%dT%H:%M:00')
        # por defecto hour
        return ts.strftime('%Y-%m-%dT%H:00:00')

    def is_critical(self, record):
        """Define si una lectura es crítica: si alertIssued True o si algún contaminante excede umbral crítico."""
        if record.get('alertIssued'):
            return True
        aq = record.get('airQualityData') or {}
        try:
            pm = float(aq.get('PM25')) if aq.get('PM25') is not None else None
        except Exception:
            pm = None
        try:
            no2 = float(aq.get('NO2')) if aq.get('NO2') is not None else None
        except Exception:
            no2 = None
        try:
            co = float(aq.get('CO')) if aq.get('CO') is not None else None
        except Exception:
            co = None
        if pm is not None and pm > float(self.critical_thresholds.get('PM25', 150.0)):
            return True
        if no2 is not None and no2 > float(self.critical_thresholds.get('NO2', 200.0)):
            return True
        if co is not None and co > float(self.critical_thresholds.get('CO', 15.0)):
            return True
        return False

    def ingest(self, record, time_window='hour'):
        """Procesa una lectura; si es crítica actualiza counters y almacena en log."""
        if not isinstance(record, dict):
            return
        if not self.is_critical(record):
            return
        city = record.get('sensorLocation') or 'unknown'
        self.city_counts[city] += 1
        bucket = self._parse_time_bucket(record.get('timestamp'), window=time_window)
        self.time_buckets[city][bucket] += 1
        # guardar un resumen ligero en alert_log
        self.alert_log.append({
            '_id': record.get('_id'),
            'city': city,
            'timestamp': record.get('timestamp'),
            'airQualityData': record.get('airQualityData'),
            'pollutionType': record.get('pollutionType'),
        })

    def compute_moment1_by_city(self):
        """Devuelve el Momento 1 (conteo absoluto) por ciudad como dict city->count."""
        return dict(self.city_counts)

    def rank_zones_by_frequency(self, top_n=None):
        items = list(self.city_counts.items())
        items.sort(key=lambda x: x[1], reverse=True)
        return items[:top_n] if top_n is not None else items

    def temporal_patterns(self, city=None):
        """Si city es None devuelve patterns para todas las ciudades; si city dado devuelve el dict de buckets->count."""
        if city is None:
            return {c: dict(buckets) for c, buckets in self.time_buckets.items()}
        return dict(self.time_buckets.get(city, {}))

    def write_alert_log(self, path, as_json_lines=True):
        """Escribe alert_log a archivo; por defecto en JSON Lines para facilidad de ingestión."""
        if as_json_lines:
            with open(path, 'w', encoding='utf-8') as f:
                for rec in self.alert_log:
                    f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        else:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.alert_log, f, ensure_ascii=False, indent=2)

    def clear(self):
        """Resetea contadores y logs."""
        self.city_counts.clear()
        self.time_buckets.clear()
        self.alert_log = []


# --- funciones de integración y utilidades ---

def process_stream_with_adaptive_sampling(records_iter, sampler, analyzer, time_window='hour', write_sampled_to=None):
    """Procesa un iterable de registros:
    - decide muestreamiento con 'sampler.should_sample'
    - si la lectura se muestrea, opcionalmente la guarda en archivo JSON Lines en write_sampled_to
    - alimenta el 'analyzer' con todas las lecturas críticas (analyzer.ingest)
    - permite ajustar pesos adaptativos cada N registros (simple heuristic)

    Retorna un dict con: {'sampled_count', 'total_count'}
    """
    sampled_count = 0
    total_count = 0
    sampled_file = None
    if write_sampled_to:
        sampled_file = open(write_sampled_to, 'w', encoding='utf-8')

    # heurística para actualizar pesos: cada M registros mirar las ciudades top y aumentar pesos
    M = 500
    for rec in records_iter:
        total_count += 1
        # primero, dejar que el analyzer vea la lectura si es crítica
        analyzer.ingest(rec, time_window=time_window)
        # decidir si muestrear
        if sampler.should_sample(rec):
            sampled_count += 1
            if sampled_file:
                sampled_file.write(json.dumps(rec, ensure_ascii=False) + '\n')
        # cada M registros, adaptar pesos según frecuencias observadas
        if total_count % M == 0:
            # por simplicidad: incrementar peso de top-3 ciudades en analyzer
            ranking = analyzer.rank_zones_by_frequency(top_n=3)
            for city, cnt in ranking:
                # multiplicador proporcional (más alertas -> más aumento)
                multiplier = 1.0 + min(1.0, cnt / max(1.0, float(max(1, cnt))))
                sampler.update_city_weight(city, 1.1 + (cnt / 100.0))
    if sampled_file:
        sampled_file.close()
    return {'sampled_count': sampled_count, 'total_count': total_count}


def load_json_stream(path, encoding='utf-8'):
    """Carga un JSON con una lista de registros y devuelve un iterador (lista) para procesar.
    (No es streaming real de líneas para mantener compatibilidad con la plantilla).
    """
    with open(path, 'r', encoding=encoding) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError('JSON root must be a list')
    return data


# Ejemplo de uso si se ejecuta directamente
if __name__ == '__main__':
    # inicializar componentes
    sampler = AdaptiveSampler(base_rate=0.05, min_rate=0.01, max_rate=0.8, seed=42)
    analyzer = FrequencyMomentsAnalyzer()

    # intentar cargar archivo 'extended_sensors.json' (usa la plantilla extendida)
    path = 'extended_sensors.json'
    try:
        records = load_json_stream(path)
    except FileNotFoundError:
        print("No se encontró 'extended_sensors.json' en el directorio actual. Coloca el archivo o modifica la ruta.")
        records = []

    if records:
        result = process_stream_with_adaptive_sampling(records, sampler, analyzer, time_window='hour', write_sampled_to='sampled_output.jsonl')
        print('Total procesados:', result['total_count'], 'Muestreados:', result['sampled_count'])
        print('Top ciudades por alertas críticas:', analyzer.rank_zones_by_frequency(top_n=10))
        # exportar log de alertas
        analyzer.write_alert_log('alerts_log.jsonl')
        print('Alert log escrito a alerts_log.jsonl')
