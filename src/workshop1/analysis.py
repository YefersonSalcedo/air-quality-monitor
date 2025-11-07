import json
from collections import defaultdict
from datetime import datetime


class HashTable:
    def __init__(self):
        # almacena el último registro por _id (si aplicable)
        self.by_reading_id = {}
        # almacena listas de registros por sensor (sensorId o sensorLocation)
        self.by_sensor = defaultdict(list)
        # lista de todos los registros en orden de inserción
        self.all_records = []
        # campo elegido para identificar sensores (determinado al cargar/insertar)
        self.sensor_key_field = None

    # Helpers
    def _choose_sensor_key_field(self, rec):
        """Determina cuál campo usar como identificador de sensor si aún no está decidido.
        Prioridad: 'sensorId' -> 'sensorLocation'.
        """
        if self.sensor_key_field:
            return
        if 'sensorId' in rec:
            self.sensor_key_field = 'sensorId'
        elif 'sensorLocation' in rec:
            self.sensor_key_field = 'sensorLocation'

    def _ensure_numeric_aq(self, rec):
        """Intenta convertir PM25 y NO2 a float dentro de rec['airQualityData'].
        Si no existen o no son convertibles, deja el valor como None.
        """
        aq = rec.get('airQualityData') or {}

        def _to_float(x):
            try:
                return float(x)
            except Exception:
                return None

        if isinstance(aq, dict):
            aq['PM25'] = _to_float(aq.get('PM25'))
            aq['NO2'] = _to_float(aq.get('NO2'))
            rec['airQualityData'] = aq

    def _parse_ts(self, rec):
        ts = rec.get('timestamp')
        if not ts:
            return None
        if isinstance(ts, datetime):
            return ts
        # intentar parseo ISO (varias variantes)
        for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%S %z", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(ts, fmt)
            except Exception:
                continue
        return None

    # Operaciones Básicas
    def insert(self, record):
        """Inserta un registro (no reemplaza por defecto). Normaliza valores.
        Espera que record tenga al menos 'sensorLocation' o 'sensorId' para agrupar por sensor.
        """
        if not isinstance(record, dict):
            raise TypeError('record must be a dict')

        # normalizar campos de calidad
        self._ensure_numeric_aq(record)
        # decidir campo sensor si es la primera inserción
        self._choose_sensor_key_field(record)

        # mantener por reading id si existe
        rid = record.get('_id')
        if rid is not None:
            self.by_reading_id[str(rid)] = record

        # agrupar por sensor
        key_field = self.sensor_key_field or 'sensorLocation'
        sensor_id = record.get(key_field)
        if sensor_id is not None:
            self.by_sensor[str(sensor_id)].append(record)

        # guardar en lista global
        self.all_records.append(record)

    def load_json(self, path, encoding='utf-8'):
        """Carga un JSON (lista) y lo inserta. Devuelve cantidad insertada."""
        with open(path, 'r', encoding=encoding) as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError('JSON root must be a list')
        inserted = 0
        for rec in data:
            try:
                self.insert(rec)
                inserted += 1
            except Exception:
                continue
        return inserted

    def search(self, key):
        """Busca por _id (reading id). Devuelve el registro o None."""
        return self.by_reading_id.get(str(key))

    def update(self, key, new_values):
        """Actualiza un registro por _id. Devuelve 1 si se actualizó, 0 si no existe."""
        rec = self.by_reading_id.get(str(key))
        if not rec:
            return 0
        rec.update(new_values)
        if 'airQualityData' in new_values:
            self._ensure_numeric_aq(rec)
        return 1

    def stats(self):
        return {
            'total_records': len(self.all_records),
            'unique_sensors': len(self.by_sensor),
            'unique_readings': len(self.by_reading_id)
        }

    # Funciones analiticas
    def get_city_aggregates(self):
        """Agrupa por sensorLocation y calcula conteo, min, max, avg para PM25 y NO2."""
        city = {}
        for rec in self.all_records:
            loc = rec.get('sensorLocation')
            if not loc:
                continue
            aq = rec.get('airQualityData', {}) or {}
            pm = aq.get('PM25')
            no2 = aq.get('NO2')
            e = city.setdefault(loc, {
                'count': 0,
                'sum_PM25': 0.0,
                'sum_NO2': 0.0,
                'min_PM25': None,
                'max_PM25': None,
                'min_NO2': None,
                'max_NO2': None,
            })
            e['count'] += 1
            if pm is not None:
                e['sum_PM25'] += pm
                e['min_PM25'] = pm if e['min_PM25'] is None else min(e['min_PM25'], pm)
                e['max_PM25'] = pm if e['max_PM25'] is None else max(e['max_PM25'], pm)
            if no2 is not None:
                e['sum_NO2'] += no2
                e['min_NO2'] = no2 if e['min_NO2'] is None else min(e['min_NO2'], no2)
                e['max_NO2'] = no2 if e['max_NO2'] is None else max(e['max_NO2'], no2)
        for loc, e in city.items():
            cnt = e.get('count', 0) or 0
            e['avg_PM25'] = (e['sum_PM25'] / cnt) if cnt else None
            e['avg_NO2'] = (e['sum_NO2'] / cnt) if cnt else None
        return city

    def rank_cities_by_pollutant(self, pollutant='PM25', top_n=None, descending=True):
        aggs = self.get_city_aggregates()
        key = f'avg_{pollutant}'
        items = [(city, vals.get(key)) for city, vals in aggs.items() if vals.get(key) is not None]
        items.sort(key=lambda x: x[1], reverse=descending)
        return items[:top_n] if top_n is not None else items

    def detect_outliers(self, thresholds=None):
        if thresholds is None:
            thresholds = {'PM25': 35.0, 'NO2': 200.0}
        out = []
        for rec in self.all_records:
            aq = rec.get('airQualityData', {}) or {}
            exceeded = {}
            for pollutant, thr in thresholds.items():
                val = aq.get(pollutant)
                if val is None:
                    continue
                if val > thr:
                    exceeded[pollutant] = {'value': val, 'threshold': thr, 'excess': val - thr}
            if exceeded:
                out.append((rec, exceeded))
        return out

    def global_stats(self):
        sums = {'PM25': 0.0, 'NO2': 0.0}
        mins = {'PM25': None, 'NO2': None}
        maxs = {'PM25': None, 'NO2': None}
        counts = {'PM25': 0, 'NO2': 0}
        for rec in self.all_records:
            aq = rec.get('airQualityData', {}) or {}
            for pollutant in ('PM25', 'NO2'):
                val = aq.get(pollutant)
                if val is None:
                    continue
                counts[pollutant] += 1
                sums[pollutant] += val
                mins[pollutant] = val if mins[pollutant] is None else min(mins[pollutant], val)
                maxs[pollutant] = val if maxs[pollutant] is None else max(maxs[pollutant], val)
        result = {}
        for pollutant in ('PM25', 'NO2'):
            cnt = counts[pollutant]
            result[pollutant] = {
                'count': cnt,
                'min': mins[pollutant] if cnt else None,
                'max': maxs[pollutant] if cnt else None,
                'avg': (sums[pollutant] / cnt) if cnt else None,
            }
        return result
#
#
#
    def get_all_sensor_ids(self):
        """Devuelve la lista de identificadores de sensor disponibles."""
        return list(self.by_sensor.keys())

    def get_sensor_records(self, sensor_id):
        """Devuelve las lecturas asociadas a sensor_id, ordenadas por timestamp si está disponible."""
        recs = list(self.by_sensor.get(str(sensor_id), []))
        def _key_ts(r):
            ts = r.get('timestamp')
            if isinstance(ts, str):
                try:
                    return datetime.fromisoformat(ts.replace('Z', '+00:00'))
                except Exception:
                    return None
            if isinstance(ts, datetime):
                return ts
            return None
        try:
            recs.sort(key=lambda r: _key_ts(r) or datetime.min)
        except Exception:
            pass
        return recs
