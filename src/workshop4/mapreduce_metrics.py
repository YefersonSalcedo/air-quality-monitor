"""
mapreduce_metrics.py
--------------------

Simulador y analizador de costos / rendimiento para una red MapReduce
aplicada al monitoreo de calidad del aire

Características:
- Estima volumen de datos generado por sensores (GB/h, GB/día).
- Modelo simple de tiempo de procesamiento (map, shuffle, reduce, overhead).
- Estima costo de procesamiento (USD) por nodo y costo de almacenamiento (USD/mes).
- Permite simular escenarios "normal" y "critical" sobre un rango de nodos.
- Genera CSV con resultados y tres gráficos (tiempo vs nodos, costo vs nodos, almacenamiento por hora).


Salida por defecto:
- /data/mapreduce_metrics_results.csv
- /data/mapreduce_processing_time.png
- /data/mapreduce_processing_cost.png
- /data/mapreduce_storage_gb_hour.png


"""
from __future__ import annotations
import os
import argparse
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import json
import sys

# -----------------------
# Utils / Model functions
# -----------------------

def estimate_data_volume(num_cities: int,
                         sensors_per_city: int,
                         readings_per_sensor_per_hour: float,
                         hours: float,
                         reading_size_bytes: int = 200) -> Dict[str, float]:
    """
    Estima volumen de datos (bytes) generado en el periodo (hours).
    reading_size_bytes: tamaño promedio por lectura (JSON o binario compacto).
    Devuelve diccionario con total_sensors, total_readings, total_bytes.
    """
    total_sensors = int(num_cities * sensors_per_city)
    total_readings = int(total_sensors * readings_per_sensor_per_hour * hours)
    total_bytes = int(total_readings * reading_size_bytes)
    return {
        "total_sensors": total_sensors,
        "total_readings": total_readings,
        "total_bytes": total_bytes
    }

def bytes_to_gb(bytes_val: float) -> float:
    return bytes_val / (1024 ** 3)

def estimate_storage_cost_gb_per_month(total_gb: float,
                                      price_per_gb_month: float = 0.02) -> float:
    """
    Estima costo mensual de almacenamiento (USD/mes) para la cantidad de GB.
    """
    return total_gb * price_per_gb_month

def estimate_processing_time_seconds(total_bytes: float,
                                     nodes: int,
                                     proc_rate_bytes_per_sec_per_node: float = 50e6,
                                     network_bandwidth_bytes_per_sec_per_node: float = 100e6,
                                     shuffle_multiplier: float = 1.2,
                                     reduce_overhead_factor: float = 0.5,
                                     overhead_seconds: float = 10.0) -> Dict[str, float]:
    """
    Modelo simple de tiempo:
      map_time = total_bytes / (nodes * proc_rate)
      shuffle_data = total_bytes * shuffle_multiplier
      shuffle_time = shuffle_data / (nodes * network_bandwidth)
      reduce_time = reduce_overhead_factor * map_time
      total = map_time + shuffle_time + reduce_time + overhead
    Notas: parámetros son ajustables para simular distintos clusters.
    """
    if nodes < 1:
        raise ValueError("nodes must be >= 1")
    map_time = float(total_bytes) / (nodes * proc_rate_bytes_per_sec_per_node)
    shuffle_data = float(total_bytes) * shuffle_multiplier
    shuffle_time = shuffle_data / (nodes * network_bandwidth_bytes_per_sec_per_node)
    reduce_time = reduce_overhead_factor * map_time
    total_time = map_time + shuffle_time + reduce_time + overhead_seconds
    return {
        "map_time_s": map_time,
        "shuffle_time_s": shuffle_time,
        "reduce_time_s": reduce_time,
        "overhead_s": overhead_seconds,
        "total_time_s": total_time
    }

def estimate_processing_cost(total_time_seconds: float,
                             nodes: int,
                             node_cost_per_hour: float = 0.5) -> float:
    """
    Costo total de procesamiento en USD, considerando el tiempo total y número de nodos.
    """
    hours = float(total_time_seconds) / 3600.0
    return hours * nodes * node_cost_per_hour

# -----------------------
# Scenario runner
# -----------------------

def run_scenario(name: str,
                 num_cities: int = 100,
                 sensors_per_city: int = 100,
                 readings_per_sensor_per_hour: float = 1.0,
                 hours: float = 1.0,
                 nodes_list: List[int] = None,
                 reading_size_bytes: int = 200,
                 proc_rate: float = 50e6,
                 net_bw: float = 100e6,
                 shuffle_mult: float = 1.2,
                 node_cost_per_hour: float = 0.5,
                 storage_price_per_gb_month: float = 0.02) -> pd.DataFrame:
    """
    Ejecuta una simulación para un escenario (nombre) sobre múltiples counts de nodos.
    Retorna un DataFrame con resultados por cada valor de nodes en nodes_list.
    """
    if nodes_list is None:
        nodes_list = [4, 8, 16]
    rows = []
    dv = estimate_data_volume(num_cities, sensors_per_city, readings_per_sensor_per_hour, hours, reading_size_bytes)
    total_bytes = dv["total_bytes"]
    gb_hour = bytes_to_gb(total_bytes)
    gb_day = gb_hour * 24.0
    storage_cost_hourly_data = estimate_storage_cost_gb_per_month(gb_hour, storage_price_per_gb_month)
    storage_cost_daily_data = estimate_storage_cost_gb_per_month(gb_day, storage_price_per_gb_month)
    for nodes in nodes_list:
        times = estimate_processing_time_seconds(total_bytes, nodes,
                                                 proc_rate_bytes_per_sec_per_node=proc_rate,
                                                 network_bandwidth_bytes_per_sec_per_node=net_bw,
                                                 shuffle_multiplier=shuffle_mult)
        proc_cost = estimate_processing_cost(times["total_time_s"], nodes, node_cost_per_hour=node_cost_per_hour)
        rows.append({
            "scenario": name,
            "nodes": nodes,
            "num_cities": num_cities,
            "sensors_per_city": sensors_per_city,
            "readings_per_sensor_per_hour": readings_per_sensor_per_hour,
            "hours": hours,
            "total_sensors": dv["total_sensors"],
            "total_readings": dv["total_readings"],
            "data_GB_hour": round(gb_hour, 6),
            "data_GB_day": round(gb_day, 6),
            "map_s": round(times["map_time_s"], 2),
            "shuffle_s": round(times["shuffle_time_s"], 2),
            "reduce_s": round(times["reduce_time_s"], 2),
            "overhead_s": round(times["overhead_s"], 2),
            "total_time_s": round(times["total_time_s"], 2),
            "processing_cost_USD": round(proc_cost, 6),
            "storage_cost_monthly_USD_hourly_data": round(storage_cost_hourly_data, 6),
            "storage_cost_monthly_USD_daily_data": round(storage_cost_daily_data, 6)
        })
    return pd.DataFrame(rows)

# -----------------------
# Plotting helpers
# -----------------------

def plot_time_vs_nodes(df: pd.DataFrame, out_path: str):
    plt.figure(figsize=(8, 5))
    for scenario in df["scenario"].unique():
        subset = df[df["scenario"] == scenario].sort_values("nodes")
        plt.plot(subset["nodes"], subset["total_time_s"], marker='o', label=scenario)
    plt.xlabel("Número de nodos")
    plt.ylabel("Tiempo total (s)")
    plt.title("Tiempo de procesamiento vs número de nodos")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_cost_vs_nodes(df: pd.DataFrame, out_path: str):
    plt.figure(figsize=(8, 5))
    for scenario in df["scenario"].unique():
        subset = df[df["scenario"] == scenario].sort_values("nodes")
        plt.plot(subset["nodes"], subset["processing_cost_USD"], marker='o', label=scenario)
    plt.xlabel("Número de nodos")
    plt.ylabel("Costo de procesamiento (USD)")
    plt.title("Costo de procesamiento vs número de nodos")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_storage_bar(df: pd.DataFrame, out_path: str):
    storage_summary = df.groupby("scenario").agg({"data_GB_hour": "mean"}).reset_index()
    plt.figure(figsize=(6, 4))
    plt.bar(storage_summary["scenario"], storage_summary["data_GB_hour"])
    plt.xlabel("Escenario")
    plt.ylabel("GB por hora (estimado)")
    plt.title("Volumen de datos por hora (estimado)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# -----------------------
# CLI and main
# -----------------------

def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="mapreduce_metrics.py",
                                     description="Simulador de rendimiento y costos MapReduce para monitoreo del aire")
    parser.add_argument("--cities", type=int, default=500, help="Número de ciudades")
    parser.add_argument("--sensors", type=int, default=200, help="Sensores por ciudad")
    parser.add_argument("--readings", type=float, default=1.0, help="Lecturas por sensor por hora (media)")
    parser.add_argument("--hours", type=float, default=1.0, help="Período simulado en horas")
    parser.add_argument("--reading-size", type=int, default=300, help="Tamaño promedio por lectura (bytes)")
    parser.add_argument("--nodes", type=int, nargs="+", default=[4, 8, 16, 32], help="Lista de conteos de nodos para simular")
    parser.add_argument("--proc-rate", type=float, default=60e6, help="Tasa de procesamiento por nodo (bytes/s)")
    parser.add_argument("--net-bw", type=float, default=150e6, help="Ancho de banda por nodo (bytes/s)")
    parser.add_argument("--shuffle-mult", type=float, default=1.3, help="Multiplicador del tamaño shuffle")
    parser.add_argument("--node-cost", type=float, default=0.6, help="Costo por nodo por hora (USD)")
    parser.add_argument("--storage-price", type=float, default=0.025, help="Precio almacenamiento USD/GB/mes")
    parser.add_argument("--scenario", choices=["normal", "critical", "both"], default="both",
                        help="Escenario para simular")
    parser.add_argument("--out-dir", type=str, default="../../data/", help="Directorio de salida para CSV y gráficos")
    parser.add_argument("--no-plots", action="store_true", help="No generar gráficos (solo CSV)")
    parser.add_argument("--pretty-json", action="store_true", help="Guardar resultados también en JSON bonito")
    return parser.parse_args(argv)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def main(argv: List[str] = None):
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)

    ensure_dir(args.out_dir)
    outputs = []

    # Parámetros base
    base = {
        "num_cities": args.cities,
        "sensors_per_city": args.sensors,
        "readings_per_sensor_per_hour": args.readings,
        "hours": args.hours,
        "reading_size_bytes": args.reading_size,
        "proc_rate": args.proc_rate,
        "net_bw": args.net_bw,
        "shuffle_mult": args.shuffle_mult,
        "node_cost_per_hour": args.node_cost,
        "storage_price_per_gb_month": args.storage_price
    }

    dfs = []
    if args.scenario in ("normal", "both"):
        df_normal = run_scenario("normal",
                                num_cities=base["num_cities"],
                                sensors_per_city=base["sensors_per_city"],
                                readings_per_sensor_per_hour=base["readings_per_sensor_per_hour"],
                                hours=base["hours"],
                                nodes_list=args.nodes,
                                reading_size_bytes=base["reading_size_bytes"],
                                proc_rate=base["proc_rate"],
                                net_bw=base["net_bw"],
                                shuffle_mult=base["shuffle_mult"],
                                node_cost_per_hour=base["node_cost_per_hour"],
                                storage_price_per_gb_month=base["storage_price_per_gb_month"])
        dfs.append(df_normal)

    if args.scenario in ("critical", "both"):
        # Escenario crítico: más lecturas y más metadata por lectura, más shuffle
        critical_readings = max(1.0, args.readings * 6.0)
        df_critical = run_scenario("critical",
                                  num_cities=base["num_cities"],
                                  sensors_per_city=base["sensors_per_city"],
                                  readings_per_sensor_per_hour=critical_readings,
                                  hours=base["hours"],
                                  nodes_list=args.nodes,
                                  reading_size_bytes=int(base["reading_size_bytes"] * 1.33),
                                  proc_rate=base["proc_rate"],
                                  net_bw=base["net_bw"],
                                  shuffle_mult=max(base["shuffle_mult"], 1.5),
                                  node_cost_per_hour=base["node_cost_per_hour"],
                                  storage_price_per_gb_month=base["storage_price_per_gb_month"])
        dfs.append(df_critical)

    if not dfs:
        print("No scenarios executed. Salida vacía.")
        return

    df_results = pd.concat(dfs, ignore_index=True)

    # CSV and JSON output
    csv_path = os.path.join(args.out_dir, "mapreduce_metrics_results.csv")
    df_results.to_csv(csv_path, index=False)
    outputs.append(csv_path)

    if args.pretty_json:
        json_path = os.path.join(args.out_dir, "mapreduce_metrics_results.json")
        df_results.to_dict(orient="records")
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(df_results.to_dict(orient="records"), fh, indent=2, ensure_ascii=False)
        outputs.append(json_path)

    # Plots
    if not args.no_plots:
        time_plot = os.path.join(args.out_dir, "mapreduce_processing_time.png")
        cost_plot = os.path.join(args.out_dir, "mapreduce_processing_cost.png")
        storage_plot = os.path.join(args.out_dir, "mapreduce_storage_gb_hour.png")
        try:
            plot_time_vs_nodes(df_results, time_plot)
            plot_cost_vs_nodes(df_results, cost_plot)
            plot_storage_bar(df_results, storage_plot)
            outputs.extend([time_plot, cost_plot, storage_plot])
        except Exception as e:
            print(f"[WARN] Error generando plots: {e}")

    # Resumen por pantalla
    print("\nSimulación completada. Archivos generados:")
    for p in outputs:
        print(f" - {p}")

    # Print a short table summary to stdout (first few rows)
    pd.set_option("display.width", 120)
    print("\nPrimeras filas de resultados:")
    print(df_results.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
