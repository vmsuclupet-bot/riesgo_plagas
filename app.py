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
