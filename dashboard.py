"""PriceIQ Dashboard — recomendación de precio cuando cambia el costo del proveedor.

Streamlit app. Correr con:
   streamlit run dashboard.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

import recommender as R

st.set_page_config(page_title="PriceIQ", layout="wide", page_icon="💰")

# ---------------------------------------------------------------------------
# Carga inicial (cacheada por sesión)
# ---------------------------------------------------------------------------
@st.cache_resource
def boot():
    R._load_all()
    return R.list_skus()

with st.spinner("Cargando modelos (primera vez ~30s)…"):
    skus = boot()

st.title("💰 PriceIQ — Recomendación de precio")
st.caption("Cuando el proveedor cambia el costo, ¿a qué precio vender?")

# ---------------------------------------------------------------------------
# SIDEBAR: selección de SKU + nuevo costo
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("1️⃣ Producto y nuevo costo")
    # Buscador
    search = st.text_input("Buscar SKU (por descripción)", placeholder="Coca, plátano, pollo…")
    if search:
        opts = skus[skus.descripcion.str.contains(search, case=False, na=False) |
                    skus.cve_art.str.contains(search, case=False, na=False)]
    else:
        opts = skus.head(50)
    if len(opts) == 0:
        st.warning("Sin coincidencias")
        st.stop()
    sku_label = st.selectbox(
        "Seleccionar producto",
        options=opts.cve_art.tolist(),
        format_func=lambda x: f"[{x}] {opts[opts.cve_art==x].descripcion.iloc[0][:50]}"
    )

    state = R.get_sku_state(sku_label)
    st.divider()
    st.markdown(f"**{state['descripcion']}**")
    st.caption(f"Familia: `{state['fam']}` · Marca: `{state['marca']}` · factor_uds={state['factor_uds']}")

    cols = st.columns(2)
    cols[0].metric("Precio actual (unit)", f"${state['precio_actual']:.2f}")
    cols[1].metric("Costo actual (unit)", f"${state['costo_actual_unitario']:.2f}")

    st.divider()
    st.subheader("Nuevo costo del proveedor")
    modo = st.radio("Forma de entrada", ["% de cambio", "Costo directo (unit)"], horizontal=True)
    if modo == "% de cambio":
        pct = st.slider("Cambio del costo", min_value=-30, max_value=80, value=10, step=1, format="%d%%")
        new_cost = state["costo_actual_unitario"] * (1 + pct/100)
    else:
        new_cost = st.number_input("Nuevo costo unitario ($)", min_value=0.01,
                                    value=float(state["costo_actual_unitario"] * 1.10),
                                    step=0.10, format="%.2f")
    st.metric("Nuevo costo unit", f"${new_cost:.2f}",
              delta=f"{(new_cost/state['costo_actual_unitario']-1)*100:+.1f}%")

    st.divider()
    objetivo = st.selectbox("Optimizar para", ["margen", "ingreso", "cantidad"], index=0)
    conservatism = st.slider(
        "Conservadurismo", 0.0, 1.0, 0.5, 0.1,
        help="0 = recomendar puramente por modelo (riesgo de extrapolación). "
             "1 = quedarse muy cerca del precio actual. 0.5 = balance."
    )

# ---------------------------------------------------------------------------
# SECCIÓN 1: Precio recomendado
# ---------------------------------------------------------------------------
st.header("2️⃣ Precio recomendado")
rec  = R.recommend_price(sku_label, new_cost, objetivo=objetivo, conservatism=conservatism)
rng  = R.recommended_range(rec, threshold=0.95, min_credibilidad=0.6)
card = R.confidence_card(sku_label, rec)
curve = rec["curva"]

# Tarjeta de confianza primero — para enmarcar todo
# Usamos colores con buen contraste en MODO CLARO y OSCURO (texto siempre oscuro).
color_map = {
    "ALTA":  {"bg":"#d1fae5", "bd":"#10b981"},
    "MEDIA": {"bg":"#fef3c7", "bd":"#f59e0b"},
    "BAJA":  {"bg":"#fee2e2", "bd":"#ef4444"},
}
sty = color_map.get(card['nivel'], {"bg":"#f3f4f6", "bd":"#6b7280"})
st.markdown(
    f"<div style='background:{sty['bg']};padding:14px;border-radius:8px;"
    f"border-left:5px solid {sty['bd']};color:#1f2937;'>"
    f"<div style='font-size:1.05em;font-weight:700;color:#111827;'>"
    f"{card['emoji']} Confianza {card['nivel']} ({card['score']*100:.0f}/100)"
    f"</div>"
    f"<div style='font-size:0.92em;color:#1f2937;margin-top:6px;line-height:1.45;'>"
    f"{card['texto']}</div>"
    f"</div>",
    unsafe_allow_html=True
)
st.write("")

# Recomendación: rango Y punto óptimo
# Usamos 2 columnas anchas arriba (rango + precio óptimo) y 2 más abajo (demanda + margen)
# para que el rango "$XX.XX – $XX.XX" quepa sin truncarse.
st.subheader("🎯 Zona recomendada")
cA, cB = st.columns(2)
cA.markdown(
    f"<div style='padding:12px 16px;border:1px solid #d1d5db;border-radius:8px;"
    f"background:#f9fafb;color:#111827;'>"
    f"<div style='font-size:0.85em;color:#6b7280;'>Rango seguro de precios</div>"
    f"<div style='font-size:1.6em;font-weight:700;color:#065f46;margin-top:2px;'>"
    f"${rng['lo']:.2f} &nbsp;–&nbsp; ${rng['hi']:.2f}</div>"
    f"<div style='font-size:0.78em;color:#6b7280;margin-top:4px;'>"
    f"Margen ≥95% del óptimo · credibilidad ≥60%</div>"
    f"</div>",
    unsafe_allow_html=True,
)
cB.metric("💡 Precio óptimo dentro del rango",
          f"${rec['precio_recomendado']:.2f}",
          delta=f"{(rec['precio_recomendado']/state['precio_actual']-1)*100:+.1f}% vs ${state['precio_actual']:.2f} actual")

c3, c4 = st.columns(2)
c3.metric("Demanda esperada/día",
          f"{rec['qty_esperada']:.1f} unidades",
          delta=f"IC80% [{rec['qty_lo']:.1f} – {rec['qty_hi']:.1f}]")
c4.metric("Margen esperado/día",
          f"${rec['margen_esperado_diario']:.2f}",
          delta=f"{rec['delta_margen_diario']:+.2f} vs precio actual")

# Componentes de la tarjeta de confianza expandidos
with st.expander("🔍 Desglose de la confianza"):
    cc1, cc2, cc3, cc4, cc5 = st.columns(5)
    cc1.metric("Credibilidad modelo", f"{card['componentes']['credibilidad_modelo']*100:.0f}%",
               help="Qué tan cerca del rango histórico de precios. <50% = extrapolación riesgosa.")
    cc2.metric("Precisión familia", f"{card['componentes']['precision_familia']*100:.0f}%",
               help=f"1 - MAPE del modelo en cambios reales en familia {state['fam']} "
                    f"({card['n_eventos_fam']} eventos en backtest quasi-experimental).")
    cc3.metric("Cobertura IC familia", f"{card['componentes']['cobertura_familia']*100:.0f}%",
               help="% de eventos reales donde la realidad cayó dentro del IC80% predicho. "
                    "Target: 80%.")
    cc4.metric("Acierto dirección familia", f"{card['componentes']['direccion_familia']*100:.0f}%",
               help="% de cambios de precio donde el modelo acertó si la demanda iba a subir o bajar.")
    cc5.metric("Suficiencia datos SKU", f"{card['componentes']['suficiencia_sku']*100:.0f}%",
               help=f"{card['n_dias_sku']} días de historia / 500 (saturado a 100%).")

# Gráfico curva
st.subheader("Curva precio → demanda y margen")
tab1, tab2, tab3 = st.tabs(["Demanda", "Margen", "Ingreso"])

with tab1:
    base = alt.Chart(curve).encode(x=alt.X("precio:Q", title="Precio unitario ($)"))
    area = base.mark_area(opacity=0.25, color="steelblue").encode(
        y=alt.Y("qty_lo:Q", title="Demanda esperada (unidades/día)"),
        y2="qty_hi:Q")
    line = base.mark_line(color="steelblue", strokeWidth=2).encode(y="qty_med:Q")
    zona = alt.Chart(pd.DataFrame({"lo":[rng['lo']], "hi":[rng['hi']]})).mark_rect(
        opacity=0.15, color="green").encode(x="lo:Q", x2="hi:Q")
    rule_act = alt.Chart(pd.DataFrame({"x":[state['precio_actual']]})).mark_rule(
        color="gray", strokeDash=[5,5]).encode(x="x:Q")
    rule_rec = alt.Chart(pd.DataFrame({"x":[rec['precio_recomendado']]})).mark_rule(
        color="darkgreen", strokeWidth=2).encode(x="x:Q")
    st.altair_chart((zona + area + line + rule_act + rule_rec).properties(height=300).interactive(),
                    use_container_width=True)
    st.caption("Banda verde: rango recomendado. Línea verde oscura: precio óptimo. "
               "Línea gris: precio actual. Área azul: IC80% calibrado de la demanda.")

with tab2:
    base = alt.Chart(curve).encode(x=alt.X("precio:Q", title="Precio unitario ($)"))
    area = base.mark_area(opacity=0.25, color="seagreen").encode(
        y=alt.Y("margen_lo:Q", title="Margen esperado ($/día)"),
        y2="margen_hi:Q")
    line = base.mark_line(color="seagreen", strokeWidth=2).encode(y="margen_med:Q")
    line_pen = base.mark_line(color="purple", strokeWidth=1.5, strokeDash=[4,2]).encode(
        y=alt.Y("margen_med_penalizado:Q", title="Margen esperado ($/día)"))
    zona = alt.Chart(pd.DataFrame({"lo":[rng['lo']], "hi":[rng['hi']]})).mark_rect(
        opacity=0.15, color="green").encode(x="lo:Q", x2="hi:Q")
    rule_rec = alt.Chart(pd.DataFrame({"x":[rec['precio_recomendado']]})).mark_rule(
        color="darkgreen", strokeWidth=2).encode(x="x:Q")
    rule_act = alt.Chart(pd.DataFrame({"x":[state['precio_actual']]})).mark_rule(
        color="gray", strokeDash=[5,5]).encode(x="x:Q")
    st.altair_chart((zona + area + line + line_pen + rule_act + rule_rec)
                    .properties(height=300).interactive(),
                    use_container_width=True)
    st.caption("Línea sólida: margen esperado por el modelo. Línea punteada morada: "
               "margen *ajustado por credibilidad* (lo que el optimizador maximiza). "
               "Banda verde: zona recomendada (margen ≥95% del óptimo + credibilidad ≥60%).")

with tab3:
    base = alt.Chart(curve).encode(x=alt.X("precio:Q", title="Precio unitario ($)"))
    area = base.mark_area(opacity=0.25, color="orange").encode(
        y=alt.Y("ingreso_lo:Q", title="Ingreso esperado ($/día)"),
        y2="ingreso_hi:Q")
    line = base.mark_line(color="darkorange", strokeWidth=2).encode(y="ingreso_med:Q")
    rule_rec = alt.Chart(pd.DataFrame({"x":[rec['precio_recomendado']]})).mark_rule(
        color="firebrick", strokeWidth=2).encode(x="x:Q")
    st.altair_chart((area + line + rule_rec).properties(height=300).interactive(),
                    use_container_width=True)

with st.expander("📊 Elasticidad estructural (modelo causal jerárquico)"):
    if state['elasticidad_estructural'] is not None:
        st.markdown(f"**β_estructural = {state['elasticidad_estructural']:+.3f}**")
        st.caption(
            "Estimada por modelo de efectos mixtos Bayesiano con IV (costo del proveedor "
            "como instrumento). Es interpretable: una subida de 10% en precio cambia la "
            f"demanda en ~{state['elasticidad_estructural']*10:.1f}% según este modelo. "
            "Para predicción de cantidad usamos LightGBM (no interpretable como elasticidad "
            "pero más preciso); ambos son complementarios.")
    else:
        st.info("Sin elasticidad estructural para este SKU.")

# ---------------------------------------------------------------------------
# SECCIÓN 2: Canibalización
# ---------------------------------------------------------------------------
st.header("3️⃣ Canibalización esperada")
subs = R.estimate_canibalization(sku_label, rec['precio_recomendado'])
if len(subs) == 0:
    st.info("Este SKU no tiene sustitutos identificados en embeddings.")
else:
    st.caption("Productos más similares por embedding (Item2Vec). β_cross_shrunk = "
               "elasticidad cruzada estimada con pooling jerárquico (NaN si no se identificó).")
    df_show = subs[["descripcion","fam","marca","similitud_embedding",
                    "beta_cross_shrunk","delta_qty_pct","tipo_vecino_canasta"]].copy()
    df_show.columns = ["Producto","Familia","Marca","Similitud","β_ij","Δ demanda %","Tipo (canasta)"]
    df_show["Similitud"] = df_show["Similitud"].apply(lambda x: f"{x:.3f}")
    df_show["β_ij"] = df_show["β_ij"].apply(lambda x: "—" if pd.isna(x) else f"{x:+.2f}")
    df_show["Δ demanda %"] = df_show["Δ demanda %"].apply(
        lambda x: "—" if pd.isna(x) else f"{x:+.1f}%")
    st.dataframe(df_show, use_container_width=True, hide_index=True)
    n_cuantif = subs.beta_cross_shrunk.notna().sum()
    if n_cuantif > 0:
        st.caption(f"⚠️ {n_cuantif} sustitutos con efecto cuantificado. Si tu precio sube,"
                   " su demanda esperada cambia como muestra Δ.")

# ---------------------------------------------------------------------------
# SECCIÓN 3: Benchmark de promociones
# ---------------------------------------------------------------------------
st.header("4️⃣ Benchmark de promociones históricas")
bench = R.get_promo_benchmarks(sku_label, k=8)
if len(bench) == 0:
    st.info("No hay promociones históricas registradas para esta familia.")
else:
    own = bench[bench.scope == "este_sku"]
    fam_p = bench[bench.scope == "misma_familia"]
    if len(own) > 0:
        st.subheader("Promociones de este SKU")
        d = own[["NumProm","fec_min","fec_max","dias_promo","descuento_pct",
                 "uplift_pct","qty_baseline_per_day","qty_during_per_day"]].copy()
        d.columns = ["Promo","Desde","Hasta","Días","Descuento","Uplift","Baseline/d","Durante/d"]
        d["Descuento"] = d["Descuento"].apply(lambda x: f"{x*100:.0f}%")
        d["Uplift"]    = d["Uplift"].apply(lambda x: f"{x:+.0f}%")
        d["Baseline/d"] = d["Baseline/d"].apply(lambda x: f"{x:.1f}")
        d["Durante/d"] = d["Durante/d"].apply(lambda x: f"{x:.1f}")
        st.dataframe(d, use_container_width=True, hide_index=True)
    if len(fam_p) > 0:
        with st.expander(f"Promociones similares en la familia {state['fam']}"):
            d = fam_p[["descripcion","NumProm","fec_min","fec_max","dias_promo",
                       "descuento_pct","uplift_pct"]].copy()
            d.columns = ["Producto","Promo","Desde","Hasta","Días","Descuento","Uplift"]
            d["Descuento"] = d["Descuento"].apply(lambda x: f"{x*100:.0f}%")
            d["Uplift"] = d["Uplift"].apply(lambda x: f"{x:+.0f}%")
            st.dataframe(d, use_container_width=True, hide_index=True)

st.divider()

# Validación quasi-experimental del modelo (evidencia)
with st.expander("📐 ¿Cómo sé que puedo confiar en estas predicciones? — Validación del modelo"):
    try:
        cred_all = pd.read_parquet("data/credibilidad_por_familia.parquet")
        ev_all = pd.read_parquet("data/backtest_quasi_events.parquet")
        st.markdown(
            f"**Backtest quasi-experimental sobre cambios reales de precio**: "
            f"detectamos {len(ev_all):,} eventos en los últimos 60 días donde el precio "
            f"de un SKU cambió ≥5%. Para cada evento entrenamos el modelo con datos *anteriores* "
            f"al cambio y comparamos su predicción contra la demanda observada *después* del cambio."
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Eventos validados", f"{len(ev_all):,}")
        c2.metric("MAPE mediano", f"{ev_all.mape_post.median():.1f}%",
                  help="Error mediano del modelo al predecir cantidad después de un cambio real de precio")
        c3.metric("Cobertura IC empírica",
                  f"{ev_all.in_ci_calib_post_rate.mean()*100:.1f}%",
                  help="% de eventos donde la realidad cayó dentro del IC80% calibrado. Target: 80%.")
        c4.metric("Acierto de dirección",
                  f"{ev_all.direccion_correcta.mean()*100:.1f}%",
                  help="% de eventos donde el modelo acertó si la demanda iba a subir o bajar")

        if state['fam'] in cred_all.fam.values:
            row = cred_all[cred_all.fam == state['fam']].iloc[0]
            st.markdown(f"**Para la familia de este SKU ({state['fam']}):** "
                        f"{int(row.n_eventos)} eventos, MAPE {row.mape_med:.1f}%, "
                        f"cobertura {row.cobertura_IC:.1f}%, dirección {row.direccion_acc:.1f}%.")
    except Exception as e:
        st.info(f"Backtest no disponible: {e}")

st.caption(
    "**Tecnología:** LightGBM (predicción) + Item2Vec embeddings + "
    "Empirical Bayes jerárquico (elasticidad estructural) + IV-2SLS (corrección de "
    "endogeneidad) + Conformal Prediction Adaptativo (calibración de IC) + "
    "Validación quasi-experimental sobre cambios reales. "
    "Datos: 22 meses de ventas, 7.3K SKUs."
)
