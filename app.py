# -*- coding: utf-8 -*-
"""
Prototipo web: riesgo de plagas en alfalfa, Valle Jequetepeque.

Pestañas:
  1. Registrar visita     -> ficha digital para construir el dataset desde el celular.
  2. Entrenar con mis datos -> aprende de las visitas y muestra métricas reales.
  3. Estimar riesgo       -> usa el modelo entrenado en la sesión o en modelo/.
  4. Acerca de            -> descripción, métricas y limitaciones.

La app NO inventa resultados: sin datos o sin modelo, lo avisa.
"""
import io
import json
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

import entrenamiento as E

RUTA_MODELO = Path(__file__).parent / "modelo"
COLUMNAS = [
    "fecha", "lote", "dias_desde_corte", "riegos_ciclo",
    "fumigaciones_ciclo", "temp_media_c", "humedad_media_pct",
    "cobertura_maleza", "nivel_plaga", "observaciones"
]
NIVELES_PLAGA = {
    0: "0 - Sin plaga",
    1: "1 - Leve",
    2: "2 - Moderada",
    3: "3 - Severa"
}
MALEZA = {0: "Baja", 1: "Media", 2: "Alta"}

CAMPOS = {
    "dias_desde_corte": dict(
        etiqueta="Días desde el último corte", tipo="slider",
        min=0, max=31, valor=15,
        ayuda="El ciclo de corte dura unos 31 días."
    ),
    "riegos_ciclo": dict(
        etiqueta="Riegos desde el último corte", tipo="int",
        min=0, max=4, valor=2,
        ayuda="Hasta 4 riegos por corte cuando no hay escasez de agua."
    ),
    "fumigaciones_ciclo": dict(
        etiqueta="Fumigaciones desde el último corte", tipo="int",
        min=0, max=20, valor=0
    ),
    "temp_media_c": dict(
        etiqueta="Temperatura media del día (°C)", tipo="float",
        min=5.0, max=45.0, valor=24.0, paso=0.5,
        ayuda="(máxima + mínima) / 2, de SENAMHI."
    ),
    "humedad_media_pct": dict(
        etiqueta="Humedad relativa media (%)", tipo="float",
        min=0.0, max=100.0, valor=75.0, paso=1.0
    ),
    "cobertura_maleza": dict(
        etiqueta="Cobertura de maleza", tipo="opcion",
        opciones=MALEZA, valor=0
    ),
}

st.set_page_config(
    page_title="Riesgo de plagas en alfalfa",
    page_icon="🌱",
    layout="wide"
)


# ------------------------------------------------------------------
# DISEÑO VISUAL (solo CSS; no modifica la lógica)
def aplicar_estilo():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #f4f8f5 0%, #edf4ef 100%);
        color: #20352b;
    }
    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .encabezado {
        background: linear-gradient(120deg, #123d2b, #20734c);
        padding: 30px 35px;
        border-radius: 22px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 8px 25px rgba(18, 61, 43, 0.18);
    }
    .encabezado h1 {
        color: white !important;
        font-size: 34px !important;
        font-weight: 800 !important;
        margin: 0;
        padding-bottom: 8px;
    }
    .encabezado p {
        color: #d9f3e4;
        font-size: 15px;
        margin: 0;
    }
    .encabezado .etiqueta {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.25);
        border-radius: 30px;
        padding: 5px 13px;
        font-size: 12px;
        margin-bottom: 14px;
        letter-spacing: 0.5px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: white;
        padding: 10px;
        border-radius: 15px;
        box-shadow: 0 3px 12px rgba(0,0,0,0.04);
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        border-radius: 10px;
        padding: 0 18px;
        color: #527060;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: #e0f2e6 !important;
        color: #17633e !important;
    }
    h1, h2, h3 {
        color: #174b33 !important;
        font-weight: 750 !important;
    }
    h2 {
        border-bottom: 2px solid #dcebe1;
        padding-bottom: 8px;
    }
    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #e0ebe4;
        border-radius: 16px;
        padding: 18px 20px;
        box-shadow: 0 4px 14px rgba(20,70,45,0.05);
    }
    [data-testid="stMetricLabel"] {
        color: #648071 !important;
        font-weight: 600;
    }
    [data-testid="stMetricValue"] {
        color: #17633e !important;
        font-weight: 800;
    }
    .stButton > button,
    .stDownloadButton > button,
    [data-testid="stFormSubmitButton"] button {
        background: linear-gradient(135deg, #17633e, #258653);
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 10px 20px;
        font-weight: 700;
        transition: all 0.2s ease;
        box-shadow: 0 4px 10px rgba(23,99,62,0.15);
    }
    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background: #124d31;
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(23,99,62,0.22);
    }
    [data-testid="stForm"] {
        background: white;
        border: 1px solid #e0ebe4;
        border-radius: 18px;
        padding: 25px;
        box-shadow: 0 5px 18px rgba(20,70,45,0.05);
    }
    [data-testid="stTable"],
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }
    [data-testid="stAlert"] {
        border-radius: 12px;
    }
    [data-testid="stExpander"] {
        background: white;
        border: 1px solid #dcebe1;
        border-radius: 13px;
        margin-bottom: 10px;
    }
    hr {
        border-color: #dcebe1;
    }
    .pie {
        text-align: center;
        color: #779080;
        font-size: 12px;
        padding-top: 25px;
    }

    /* Overrides para evitar el color rojo del tema de Streamlit */
    :root {
        --primary-color: #238653;
        --secondary-background-color: #edf5ef;
        --text-color: #20352b;
    }
    [data-testid="stAppViewContainer"] {
        background: #f1f6f2;
    }
    [data-testid="stHeader"] {
        background: rgba(241, 246, 242, 0.85);
    }
    [data-baseweb="tab-highlight"],
    [data-baseweb="tab-border"] {
        background-color: #238653 !important;
    }
    button[role="tab"][aria-selected="true"] {
        color: #17633e !important;
        background: #e0f2e6 !important;
        border-bottom: 3px solid #238653 !important;
    }
    button[role="tab"] {
        border-radius: 12px !important;
        font-size: 14px !important;
        min-height: 48px !important;
    }
    [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
        background-color: #238653 !important;
        border-color: #238653 !important;
    }
    [data-testid="stSlider"] [data-baseweb="slider"] > div > div {
        background-color: #238653 !important;
    }
    input:focus, textarea:focus {
        border-color: #238653 !important;
        box-shadow: 0 0 0 1px #238653 !important;
    }
    .stSelectbox [data-baseweb="select"]:focus-within,
    .stDateInput [data-baseweb="input"]:focus-within {
        border-color: #238653 !important;
    }
    /* Títulos y textos con mejor jerarquía */
    .stMarkdown p {
        line-height: 1.65;
    }
    [data-testid="stVerticalBlock"] > [data-testid="stForm"] {
        margin-top: 12px;
    }
    /* Mejor aspecto en pantallas pequeñas */
    @media (max-width: 768px) {
        .block-container {
            padding: 1rem 1rem 2rem;
        }
        .encabezado {
            padding: 23px 20px;
            border-radius: 17px;
        }
        .encabezado h1 {
            font-size: 26px !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 3px;
            padding: 5px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 0 8px;
            font-size: 12px;
        }
    }
    </style>
    """, unsafe_allow_html=True)


# ------------------------------------------------------------------ utilidades
@st.cache_resource
def cargar_modelo():
    f_mod = RUTA_MODELO / "modelo_final.joblib"
    f_meta = RUTA_MODELO / "modelo_final.json"
    if not (f_mod.exists() and f_meta.exists()):
        return None, None, None
    try:
        meta = json.loads(f_meta.read_text(encoding="utf-8"))
        return joblib.load(f_mod), meta, None
    except Exception as e:
        return None, None, str(e)


def campo(var, clave):
    c = CAMPOS[var]
    if c["tipo"] == "slider":
        return st.slider(
            c["etiqueta"], c["min"], c["max"], c["valor"],
            help=c.get("ayuda"), key=clave
        )
    if c["tipo"] == "int":
        return st.number_input(
            c["etiqueta"], min_value=c["min"], max_value=c["max"],
            value=c["valor"], step=1, help=c.get("ayuda"), key=clave
        )
    if c["tipo"] == "float":
        return st.number_input(
            c["etiqueta"], min_value=c["min"], max_value=c["max"],
            value=c["valor"], step=c["paso"], help=c.get("ayuda"), key=clave
        )
    return st.selectbox(
        c["etiqueta"], list(c["opciones"].keys()), index=c["valor"],
        format_func=lambda k: c["opciones"][k], key=clave
    )


def explicar(modelo, X, vars_):
    """Devuelve (titulo, DataFrame o lista) que explica la predicción."""
    pasos = modelo.named_steps
    imp = pasos["simpleimputer"]
    Xi = imp.transform(X)

    if "decisiontreeclassifier" in pasos:
        arbol = pasos["decisiontreeclassifier"]
        ind = arbol.decision_path(Xi)
        hoja = arbol.apply(Xi)[0]
        nodos = ind.indices[ind.indptr[0]:ind.indptr[1]]
        lineas = []
        for n in nodos:
            if n == hoja:
                continue
            f, umbral = arbol.tree_.feature[n], arbol.tree_.threshold[n]
            valor = Xi[0, f]
            lineas.append(
                f"{vars_[f]} = {valor:g} "
                f"{'≤' if valor <= umbral else '>'} {umbral:.2f}"
            )
        return "Camino de decisión del árbol CART", lineas

    if "logisticregression" in pasos:
        xs = pasos["standardscaler"].transform(Xi)[0]
        coef = pasos["logisticregression"].coef_[0]
        df = pd.DataFrame({
            "variable": vars_,
            "aporte": coef * xs
        }).set_index("variable")
        return "Aporte de cada variable (positivo = empuja hacia riesgo alto)", df

    return None, None


# ------------------------------------------------------------------ pestañas
def pestana_estimar(modelo, meta, error):
    sesion = st.session_state.get("entrenado")
    if sesion is not None:
        modelo, meta, error = sesion["modelo"], sesion["meta"], None
        st.caption(
            f"Modelo entrenado en esta sesión con {meta['registros']} registros "
            f"({meta['fechas']} fechas de visita): {meta['modelo']}."
        )

    if error:
        st.error(
            "No se pudo cargar el modelo: " + error +
            " Revisa que scikit-learn tenga la misma versión con la que se entrenó."
        )
        return

    if modelo is None:
        st.info(
            "Todavía no hay un modelo. Registra visitas en la pestaña "
            "**Registrar visita** y luego entrena en **Entrenar con mis datos**. "
            "También puedes copiar `modelo_final.joblib` y "
            "`modelo_final.json` a la carpeta `modelo/`."
        )
        return

    vars_ = meta["variables"]
    desconocidas = [v for v in vars_ if v not in CAMPOS]
    if desconocidas:
        st.error(f"El modelo usa variables que la app no conoce: {desconocidas}")
        return

    st.write("Ingresa las condiciones del lote para estimar su riesgo de plaga.")
    valores = {v: campo(v, f"est_{v}") for v in vars_}

    if st.button("Estimar riesgo", type="primary"):
        X = pd.DataFrame([valores])[vars_]
        pred = int(modelo.predict(X)[0])
        prob = None

        if hasattr(modelo, "predict_proba") and 1 in list(modelo.classes_):
            prob = float(
                modelo.predict_proba(X)[0][list(modelo.classes_).index(1)]
            )

        if pred == 1:
            st.error("**Riesgo ALTO de plaga**")
            st.write(
                "Conviene inspeccionar el lote con prioridad (muestreo de tallos), "
                "registrar el nivel de plaga observado y consultar a un ingeniero "
                "agrónomo antes de decidir una aplicación."
            )
        else:
            st.success("**Riesgo BAJO de plaga**")
            st.write(
                "Se sugiere mantener el monitoreo periódico del lote. "
                "Este resultado no garantiza la ausencia de plagas."
            )

        if prob is not None:
            st.metric("Probabilidad estimada de riesgo alto", f"{prob:.0%}")

        titulo, detalle = explicar(modelo, X, vars_)
        if titulo:
            with st.expander("¿Por qué este resultado?"):
                st.write(f"**{titulo}**")
                if isinstance(detalle, list):
                    for linea in detalle:
                        st.write("• " + linea)
                else:
                    st.bar_chart(detalle)

        st.caption(
            "Prototipo de apoyo a la decisión entrenado con pocos registros: "
            "no sustituye al técnico agrónomo."
        )


def pestana_registro():
    st.write(
        "Ficha digital: registra cada visita a un lote. Al terminar, "
        "**descarga el CSV**: los datos de esta pantalla se pierden si cierras la página."
    )

    if "registros" not in st.session_state:
        st.session_state.registros = []

    with st.form("ficha", clear_on_submit=False):
        c1, c2 = st.columns(2)
        fecha = c1.date_input("Fecha de la visita", value=date.today())
        lote = c2.text_input("Lote", value="L1")

        valores = {}
        for v in [
            "dias_desde_corte", "riegos_ciclo",
            "fumigaciones_ciclo", "temp_media_c"
        ]:
            valores[v] = campo(v, f"reg_{v}")

        con_hum = st.checkbox("Tengo el dato de humedad relativa")
        hum = campo("humedad_media_pct", "reg_humedad_media_pct")
        valores["cobertura_maleza"] = campo(
            "cobertura_maleza", "reg_cobertura_maleza"
        )

        nivel = st.selectbox(
            "Nivel de plaga observado",
            list(NIVELES_PLAGA.keys()),
            format_func=lambda k: NIVELES_PLAGA[k],
            key="reg_nivel",
            help="Usa los criterios acordados con el ingeniero agrónomo."
        )
        obs = st.text_area(
            "Observaciones (fotos tomadas, producto aplicado, incidencias)"
        )
        guardar = st.form_submit_button("Guardar visita", type="primary")

    if guardar:
        if not lote.strip():
            st.error("Escribe el nombre del lote.")
        else:
            st.session_state.registros.append({
                "fecha": fecha.isoformat(),
                "lote": lote.strip(),
                "dias_desde_corte": valores["dias_desde_corte"],
                "riegos_ciclo": valores["riegos_ciclo"],
                "fumigaciones_ciclo": valores["fumigaciones_ciclo"],
                "temp_media_c": valores["temp_media_c"],
                "humedad_media_pct": hum if con_hum else None,
                "cobertura_maleza": valores["cobertura_maleza"],
                "nivel_plaga": nivel,
                "observaciones": obs.strip()
            })
            st.success("Visita guardada.")

    df = pd.DataFrame(st.session_state.registros, columns=COLUMNAS)
    st.write(f"**Registros en esta sesión: {len(df)}**")

    if len(df):
        st.table(df)
        a, b = st.columns(2)
        csv = df.to_csv(index=False).encode("utf-8-sig")
        a.download_button(
            "Descargar CSV", csv,
            file_name="registros_alfalfa.csv", mime="text/csv"
        )
        if b.button("Eliminar el último registro"):
            st.session_state.registros.pop()
            st.rerun()

    with st.expander("Continuar con un CSV descargado antes"):
        subido = st.file_uploader("Archivo CSV", type="csv")
        if subido is not None and st.button("Cargar registros del archivo"):
            previo = pd.read_csv(subido)
            faltan = [c for c in COLUMNAS if c not in previo.columns]
            if faltan:
                st.error(f"Al archivo le faltan columnas: {faltan}")
            else:
                previo = previo[COLUMNAS].astype(object).where(
                    previo[COLUMNAS].notna(), None
                )
                st.session_state.registros = previo.to_dict("records")
                st.rerun()


def pestana_entrenar():
    st.write(
        "Aquí el programa **aprende de tus visitas**: busca qué condiciones "
        "se asociaban con más plaga y guarda ese aprendizaje en un modelo, "
        "que luego usa la pestaña *Estimar riesgo*. Entre más fechas de visita "
        "registres, más ejemplos tiene para aprender."
    )

    fuente = st.radio(
        "¿De dónde tomo los datos?",
        ["Registros de esta sesión", "Subir un archivo (CSV o Excel)"]
    )
    crudo = None

    if fuente.startswith("Registros"):
        if st.session_state.get("registros"):
            crudo = pd.DataFrame(st.session_state["registros"])
        else:
            st.info(
                "Aún no hay registros en esta sesión. Guarda visitas en "
                "**Registrar visita** o sube un archivo."
            )
    else:
        arch = st.file_uploader(
            "Archivo con las visitas (CSV, o Excel con la hoja 'Datos')",
            type=["csv", "xlsx"], key="subida_entrenar"
        )
        if arch is not None:
            try:
                crudo = (
                    pd.read_excel(arch, sheet_name="Datos")
                    if arch.name.lower().endswith("xlsx")
                    else pd.read_csv(arch)
                )
            except Exception as e:
                st.error(f"No pude leer el archivo: {e}")

    if crudo is None:
        return

    try:
        df = E.preparar(crudo)
    except ValueError as e:
        st.error(str(e))
        return

    if df.empty:
        st.warning("No hay registros válidos (revisa fecha, lote y nivel de plaga).")
        return

    vars_ = E.elegir_variables(df)
    n_fechas = int(df["fecha"].nunique())

    a, b, c = st.columns(3)
    a.metric("Registros", len(df))
    b.metric("Lotes", int(df["lote"].nunique()))
    c.metric("Fechas de visita", n_fechas)

    if n_fechas < 3:
        st.warning(
            "Se necesitan al menos 3 fechas de visita distintas para entrenar y validar."
        )
        return

    n = n_fechas if n_fechas == 3 else st.slider(
        "¿Cuántas fechas de visita uso para entrenar? (las primeras, en orden)",
        3, n_fechas, n_fechas
    )
    sub = E.primeras_fechas(df, n)
    st.caption(f"Se usarán {len(sub)} registros de las primeras {n} fechas.")

    err = E.comprobar(sub)
    if err:
        st.error(err)
        return

    if st.button("Entrenar y validar", type="primary"):
        try:
            tg, tl, mats = E.validar(sub, vars_)
            fin = E.entrenar_final(sub, vars_, tg)
        except ValueError as e:
            st.error(str(e))
            return

        st.session_state["entrenado"] = dict(
            modelo=fin["modelo"], elegido=fin["elegido"],
            F1=fin["F1"], reglas=fin["reglas"],
            supera=fin["supera_linea_base"], tg=tg, tl=tl, mats=mats,
            meta=dict(
                modelo=fin["elegido"], variables=vars_,
                clases={"0": "bajo", "1": "alto"},
                umbral_nivel_plaga_alto=E.UMBRAL_ALTO,
                registros=len(sub), fechas=n
            )
        )
        st.rerun()

    res = st.session_state.get("entrenado")
    if res is not None:
        m = res["meta"]
        st.success(
            f"Modelo entrenado con {m['registros']} registros "
            f"({m['fechas']} fechas). Elegido: {res['elegido']} "
            f"(F1 = {res['F1']:.2f}). Ya puedes usarlo en **Estimar riesgo**."
        )

        if not res["supera"]:
            st.warning(
                "Ningún modelo supera a la línea base (predecir siempre la clase "
                "mayoritaria). Con estos datos el aprendizaje no aporta; repórtalo así."
            )

        st.write("**Métricas de validación** (cada lote se reserva una vez como prueba)")
        st.table(res["tg"])

        st.write("**Matrices de confusión** (suma de todas las pruebas)")
        for nombre, cm in res["mats"].items():
            st.write(nombre)
            st.table(pd.DataFrame(
                cm,
                index=["Real: bajo", "Real: alto"],
                columns=["Predicho: bajo", "Predicho: alto"]
            ))

        with st.expander("Reglas aprendidas por el árbol CART"):
            st.code(res["reglas"])

        with st.expander("Resultados por lote de prueba"):
            st.table(res["tl"])

        buf = io.BytesIO()
        joblib.dump(res["modelo"], buf)
        d1, d2, d3 = st.columns(3)
        d1.download_button(
            "Descargar modelo", buf.getvalue(), "modelo_final.joblib"
        )
        d2.download_button(
            "Descargar ficha del modelo",
            json.dumps(m, ensure_ascii=False, indent=2),
            "modelo_final.json", mime="application/json"
        )
        d3.download_button(
            "Descargar métricas",
            res["tg"].to_csv(index=False).encode("utf-8-sig"),
            "metricas_globales.csv", mime="text/csv"
        )

    if st.checkbox("Ver cómo cambia el resultado al usar más fechas de visita"):
        curva = E.curva_aprendizaje(df, vars_)
        if curva is None:
            st.info(
                "Se necesitan al menos 4 fechas de visita con variedad de clases "
                "en varios lotes."
            )
        else:
            st.line_chart(curva)
            st.caption(
                "F1 de la clase riesgo alto al usar las primeras N fechas. "
                "Con pocos registros la curva puede subir y bajar: sirve para ver "
                "si más datos ayudan, no para prometerlo."
            )


def pestana_acerca(modelo, meta):
    st.subheader("Acerca del prototipo")
    st.write(
        "Herramienta de apoyo para estimar el **riesgo de plagas (bajo o alto)** "
        "en lotes de alfalfa del Valle Jequetepeque, a partir de variables de "
        "manejo (días desde el corte, riegos, fumigaciones, maleza) y de clima "
        "(temperatura y humedad). El riesgo alto corresponde a un nivel de plaga "
        "observado de 2 o 3 en una escala de 0 a 3."
    )

    sesion = st.session_state.get("entrenado")
    if sesion is not None:
        meta = sesion["meta"]

    if meta:
        st.write(
            f"**Modelo cargado:** {meta.get('modelo', 'desconocido')}. "
            f"**Variables:** {', '.join(meta.get('variables', []))}."
        )

    f_met = RUTA_MODELO / "metricas_globales.csv"
    if sesion is not None:
        st.write("**Métricas de validación** (modelo entrenado en esta sesión):")
        st.table(sesion["tg"])
    elif f_met.exists():
        st.write("**Métricas de validación** (cada lote se reservó una vez como prueba):")
        st.table(pd.read_csv(f_met))
    else:
        st.caption(
            "Las métricas aparecerán aquí cuando se copie "
            "`metricas_globales.csv` a la carpeta `modelo/`."
        )

    st.write("**Limitaciones**")
    st.write(
        "• Estudio piloto con 4 lotes y pocos registros: las métricas son "
        "inestables y no se pueden generalizar a todo el valle.\n\n"
        "• Los cuatro lotes se cortan el mismo día, por lo que el efecto de "
        "la fase del ciclo no se separa del efecto del clima de esas semanas.\n\n"
        "• No identifica especies de plagas; estima el nivel de daño.\n\n"
        "• No sustituye al ingeniero agrónomo."
    )
    st.caption(
        "Universidad Nacional de Trujillo, Facultad de Ciencias Físicas y Matemáticas. "
        "V. M. Suclupe Tirado, D. J. Vásquez Silva y J. Bytton Alcántara, 2026."
    )


def main():
    aplicar_estilo()

    st.markdown("""
    <div class="encabezado">
        <div class="etiqueta">🌿 SISTEMA DE MONITOREO AGRÍCOLA</div>
        <h1>Riesgo de plagas en alfalfa</h1>
        <p>Valle Jequetepeque, La Libertad · Prototipo de apoyo a la decisión</p>
    </div>
    """, unsafe_allow_html=True)

    modelo, meta, error = cargar_modelo()
    t1, t2, t3, t4 = st.tabs([
        "📝 Registrar visita",
        "🧠 Entrenar con mis datos",
        "🌱 Estimar riesgo",
        "ℹ️ Acerca de"
    ])

    with t1:
        pestana_registro()
    with t2:
        pestana_entrenar()
    with t3:
        pestana_estimar(modelo, meta, error)
    with t4:
        pestana_acerca(modelo, meta)


main()
