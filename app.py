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
