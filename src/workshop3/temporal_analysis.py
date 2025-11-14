"""
VERSION PRELIMINAR

Este módulo realiza el análisis temporal de la cadena de Markov usando:

1. Estados definidos en el sistema:
    - good
    - moderate
    - unhealthy
    - hazardous

2. Datos del JSON masivo generado por la plantilla:
    Cada registro contiene:
        airQualityState
        previousAirQualityState
        stateTransitionProb
        timeInCurrentState
        expectedStateChangeTime
        absorptionState (bool)
        mixingTime
        criticalEventProb
        neighboringSensors[]
        weatherInfluence{}

NOTA: Este módulo **NO construye matrices**, solo las *usa*.
   Se espera que markov_model.py entregue dichas matrices.

===========================================
REQUERIMIENTO IMPORTANTE:

El archivo markov_model.py DEBE contener:

    def get_transition_matrix(sensor_id: str) -> np.ndarray:
        Retorna la matriz de transición 4x4 del sensor

    def get_stationary_distribution(sensor_id: str) -> np.ndarray:
        Retorna la distribución estacionaria π asociada.

    def get_state_index(state_name: str) -> int:
        Mapea 'good','moderate','unhealthy','hazardous' → 0,1,2,3

Este archivo temporal_analysis.py trabaja ASUMIENDO que esas funciones existen.
===========================================
"""
import numpy as np
from numpy.linalg import inv, eig
from typing import Optional, Dict, List, Tuple

# ---------------------------------------------
# Intento cargar markov_model (se requiere en este proyecto)
# ---------------------------------------------
try:
    import markov_model     # <-- Debe estar creado por Persona 1
except Exception:
    raise ImportError(
        "ERROR: No se encontró markov_model.py.\n"
        "Persona 1 debe entregar un módulo con:\n"
        " - get_transition_matrix(sensor_id)\n"
        " - get_stationary_distribution(sensor_id)\n"
        " - get_state_index(state_name)\n"
    )


# ---------------------------------------------
# Validación de matriz estocástica
# ---------------------------------------------
def _validate_stochastic_matrix(P: np.ndarray) -> None:
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("La matriz P debe ser cuadrada.")
    if not np.allclose(P.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("Cada fila de P debe sumar 1.")


# ---------------------------------------------
# Detección de estados absorbentes
# (estado i es absorbente si P[i,i] = 1 y resto de la fila 0)
# ---------------------------------------------
def detect_absorbing_states(P: np.ndarray) -> List[int]:
    _validate_stochastic_matrix(P)
    absorbing = []
    for i in range(P.shape[0]):
        if np.isclose(P[i, i], 1.0) and np.allclose(P[i] - np.eye(P.shape[0])[i], 0):
            absorbing.append(i)
    return absorbing


# ---------------------------------------------
# Partición Transitorios / Absorbentes
# ---------------------------------------------
def partition_transient_absorbing(P: np.ndarray) -> Tuple[List[int], List[int], np.ndarray, np.ndarray]:
    absorbing = detect_absorbing_states(P)
    transient = [i for i in range(P.shape[0]) if i not in absorbing]

    if len(transient) == 0:
        return transient, absorbing, np.zeros((0, 0)), np.zeros((0, len(absorbing)))

    Q = P[np.ix_(transient, transient)]
    R = P[np.ix_(transient, absorbing)]
    return transient, absorbing, Q, R


# ---------------------------------------------
# Matriz fundamental N = (I - Q)^(-1)
# + tiempos y probabilidades de absorción
# ---------------------------------------------
def fundamental_matrix_and_absorption(P: np.ndarray) -> Dict[str, np.ndarray]:
    transient, absorbing, Q, R = partition_transient_absorbing(P)
    if len(transient) == 0:
        return {"transient_idxs": [], "absorbing_idxs": absorbing, "N": np.zeros((0, 0)),
                "t": np.zeros((0,)), "B": np.zeros((0, len(absorbing)))}

    I = np.eye(Q.shape[0])
    try:
        N = inv(I - Q)
    except Exception:
        N = np.linalg.pinv(I - Q)

    t = N @ np.ones((N.shape[0],))                     # Tiempo esperado a absorción
    B = N @ R if R.size > 0 else np.zeros((N.shape[0], 0))  # Probabilidad de caer en cada absorbente

    return {
        "transient_idxs": transient,
        "absorbing_idxs": absorbing,
        "N": N,
        "t": t,
        "B": B
    }


# ---------------------------------------------
# Tiempo esperado hasta absorción por estado
# ---------------------------------------------
def expected_absorption_times(P: np.ndarray) -> Dict[int, float]:
    res = fundamental_matrix_and_absorption(P)
    t = res["t"]
    transient = res["transient_idxs"]
    return {idx: float(ti) for idx, ti in zip(transient, t)}


# ---------------------------------------------
# Probabilidad de absorción hacia cada estado crítico
# ---------------------------------------------
def absorption_probabilities(P: np.ndarray) -> Dict[Tuple[int, int], float]:
    res = fundamental_matrix_and_absorption(P)
    transient = res["transient_idxs"]
    absorbing = res["absorbing_idxs"]
    B = res["B"]
    out = {}
    for i, ti in enumerate(transient):
        for j, aj in enumerate(absorbing):
            out[(ti, aj)] = float(B[i, j])
    return out


# ---------------------------------------------
# Distancia variacional total
# ---------------------------------------------
def total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * np.sum(np.abs(p - q))


# ---------------------------------------------
# Estimación de tiempo de mezcla para convergencia
# ---------------------------------------------
def estimate_mixing_time(P: np.ndarray,
                         pi_star: Optional[np.ndarray] = None,
                         eps: float = 1e-3,
                         max_steps: int = 5000) -> Tuple[int, float]:
    _validate_stochastic_matrix(P)

    n = P.shape[0]
    pi = np.ones(n) / n  # inicial: distribución uniforme

    # Si no dieron π*, la tomamos desde markov_model
    if pi_star is None:
        try:
            pi_star = markov_model.get_stationary_distribution("GLOBAL")
        except Exception:
            # fallback: eigenvector
            vals, vecs = eig(P.T)
            idx = np.argmin(np.abs(vals - 1.0))
            pi_star = np.real(vecs[:, idx])
            pi_star = pi_star / pi_star.sum()

    for t in range(1, max_steps + 1):
        pi_next = pi @ P
        tvd = total_variation_distance(pi_next, pi_star)
        if tvd < eps:
            return t, float(tvd)
        pi = pi_next

    return max_steps, float(tvd)


# ---------------------------------------------
# Tiempo de estabilización (||π(t+1) − π(t)||₁ < threshold)
# ---------------------------------------------
def stabilization_time(P: np.ndarray,
                       threshold: float = 1e-4,
                       max_steps: int = 5000) -> int:
    n = P.shape[0]
    pi = np.ones(n) / n
    for t in range(1, max_steps + 1):
        pi_next = pi @ P
        delta = np.linalg.norm(pi_next - pi, ord=1)
        if delta < threshold:
            return t
        pi = pi_next
    return max_steps


# ---------------------------------------------
# Simulación de cadena de Markov (trayectoria temporal)
# ---------------------------------------------
def simulate_chain(P: np.ndarray, start_state: int, steps: int = 2000) -> np.ndarray:
    states = np.zeros(steps + 1, dtype=int)
    states[0] = start_state
    for t in range(1, steps + 1):
        states[t] = np.random.choice(P.shape[0], p=P[states[t-1]])
    return states


# ---------------------------------------------
# Función principal para análisis por sensor
# ---------------------------------------------
def analyze_sensor(sensor_id: str) -> Dict:
    """
    Realiza el análisis temporal para un sensor específico.
    Se espera que markov_model.py entregue la matriz mediante:

        P = markov_model.get_transition_matrix(sensor_id)

    """
    P = markov_model.get_transition_matrix(sensor_id)
    _validate_stochastic_matrix(P)

    absorb = detect_absorbing_states(P)
    fa = fundamental_matrix_and_absorption(P)
    t_mix, tvd = estimate_mixing_time(P)
    t_stable = stabilization_time(P)

    return {
        "sensor_id": sensor_id,
        "transition_matrix": P.tolist(),
        "absorbing_states": absorb,
        "expected_absorption_times": expected_absorption_times(P),
        "absorption_probabilities": {
            f"{i}->{j}": p for (i, j), p in absorption_probabilities(P).items()
        },
        "mixing_time": t_mix,
        "mixing_tvd": tvd,
        "stabilization_time": t_stable
    }