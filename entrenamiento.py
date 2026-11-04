# -*- coding: utf-8 -*-
"""
Entrenamiento y validación usados por la app (misma lógica que entrenar_modelos.py).
Trabaja solo con los datos recibidos: no genera ni completa registros.
"""
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

SEMILLA = 42
UMBRAL_ALTO = 2  # nivel_plaga >= 2 -> riesgo alto
OBLIGATORIAS = ["fecha", "lote", "dias_desde_corte", "riegos_ciclo",
                "fumigaciones_ciclo", "temp_media_c", "nivel_plaga"]
OPCIONALES = ["humedad_media_pct", "cobertura_maleza"]
BASE = ["dias_desde_corte", "riegos_ciclo", "fumigaciones_ciclo", "temp_media_c"]
LINEA_BASE = "Línea base (clase mayoritaria)"


def preparar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    faltan = [c for c in OBLIGATORIAS if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas obligatorias: {faltan}")
    if "observaciones" in df.columns:  # ignora la fila de ejemplo de la plantilla
        df = df[~df["observaciones"].astype(str).str.upper().str.startswith("EJEMPLO")]
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["nivel_plaga"] = pd.to_numeric(df["nivel_plaga"], errors="coerce")
    df = df.dropna(subset=["fecha", "lote", "nivel_plaga"])
    df["riesgo"] = (df["nivel_plaga"] >= UMBRAL_ALTO).astype(int)
    return df.reset_index(drop=True)


def elegir_variables(df: pd.DataFrame):
    vars_ = list(BASE)
    for c in OPCIONALES:
        if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().mean() >= 0.8:
            vars_.append(c)
    for c in vars_:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return vars_


def primeras_fechas(df: pd.DataFrame, n: int) -> pd.DataFrame:
    fechas = sorted(df["fecha"].unique())[:n]
    return df[df["fecha"].isin(fechas)].reset_index(drop=True)


def comprobar(df: pd.DataFrame):
    """Devuelve un mensaje de error si no se puede entrenar, o None."""
    if df["lote"].nunique() < 3:
        return "Se necesitan al menos 3 lotes distintos para validar dejando un lote fuera."
    if df["riesgo"].nunique() < 2:
        return ("Todos los registros tienen la misma clase de riesgo, así que no hay nada que aprender. "
                "Repórtalo como resultado; no se deben crear datos artificiales.")
    return None


def crear_modelos():
    cart = make_pipeline(
        SimpleImputer(strategy="median"),
        DecisionTreeClassifier(criterion="gini", max_depth=3, min_samples_leaf=3,
                               class_weight="balanced", random_state=SEMILLA))
    logit = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=SEMILLA))
    base = make_pipeline(SimpleImputer(strategy="median"), DummyClassifier(strategy="most_frequent"))
    return {"CART": cart, "Regresión logística": logit, LINEA_BASE: base}


def metricas(y, p):
    return {"exactitud": accuracy_score(y, p),
            "precisión": precision_score(y, p, pos_label=1, zero_division=0),
            "recall": recall_score(y, p, pos_label=1, zero_division=0),
            "F1": f1_score(y, p, pos_label=1, zero_division=0)}


def validar(df: pd.DataFrame, vars_):
    """Validación dejando un lote fuera. Devuelve (tabla global, tabla por lote, matrices)."""
    X, y, g = df[vars_], df["riesgo"].values, df["lote"].values
    globales, por_lote, matrices = [], [], {}
    for nombre in crear_modelos():
        y_all, p_all = [], []
        for tr, te in LeaveOneGroupOut().split(X, y, g):
            if len(set(y[tr])) < 2:
                continue
            p = crear_modelos()[nombre].fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
            y_all += list(y[te]); p_all += list(p)
            por_lote.append({"modelo": nombre, "lote de prueba": g[te][0], "registros": len(te),
                             **metricas(y[te], p)})
        if y_all:
            globales.append({"modelo": nombre, "registros evaluados": len(y_all), **metricas(y_all, p_all)})
            matrices[nombre] = confusion_matrix(y_all, p_all, labels=[0, 1])
    if not globales:
        raise ValueError("No fue posible validar: revisa que haya variedad de clases en los lotes.")
    return pd.DataFrame(globales).round(3), pd.DataFrame(por_lote).round(3), matrices


def entrenar_final(df: pd.DataFrame, vars_, tg: pd.DataFrame):
    """Entrena con todos los registros y elige el modelo (empate en F1 -> CART, el interpretable)."""
    X, y = df[vars_], df["riesgo"].values
    cand = tg[tg["modelo"] != LINEA_BASE]
    mejor = cand["F1"].max()
    empatados = cand[cand["F1"] >= mejor - 1e-9]["modelo"].tolist()
    elegido = "CART" if "CART" in empatados else empatados[0]
    cart = crear_modelos()["CART"].fit(X, y)
    logit = crear_modelos()["Regresión logística"].fit(X, y)
    reglas = export_text(cart.named_steps["decisiontreeclassifier"], feature_names=vars_)
    base = tg[tg["modelo"] == LINEA_BASE]
    supera_base = bool(base.empty or mejor > base["F1"].iloc[0] + 1e-9)
    return {"modelo": cart if elegido == "CART" else logit, "elegido": elegido, "F1": float(mejor),
            "reglas": reglas, "supera_linea_base": supera_base}


def curva_aprendizaje(df: pd.DataFrame, vars_):
    """F1 (clase riesgo alto) al usar las primeras N fechas de visita, para varios N."""
    n_f = df["fecha"].nunique()
    if n_f < 4:
        return None
    puntos = sorted(set(np.linspace(3, n_f, min(6, n_f - 2)).round().astype(int)))
    filas = []
    for n in puntos:
        sub = primeras_fechas(df, n)
        if comprobar(sub):
            continue
        try:
            tg, _, _ = validar(sub, vars_)
        except ValueError:
            continue
        fila = {"fechas usadas": n}
        for _, r in tg.iterrows():
            if r["modelo"] != LINEA_BASE:
                fila[r["modelo"]] = r["F1"]
        filas.append(fila)
    return pd.DataFrame(filas).set_index("fechas usadas") if len(filas) >= 2 else None
