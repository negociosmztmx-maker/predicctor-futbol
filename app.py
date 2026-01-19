import streamlit as st
import pandas as pd
from scipy.stats import poisson
import glob

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Predicctor Pro de Fútbol", 
    page_icon="⚽", 
    layout="wide"
)

# --- CAPA 1: LÓGICA DEL NEGOCIO (MODELO) ---

class PredictorFutbol:
    """Encargado de los cálculos matemáticos de probabilidad y valor."""
    
    def calcular_probabilidad_partido(self, lamb_home: float, lamb_away: float) -> dict:
        prob_local, prob_empate, prob_visitante = 0.0, 0.0, 0.0
        # Iteramos resultados posibles (0-0 hasta 9-9)
        for i in range(10):
            for j in range(10):
                p = poisson.pmf(i, lamb_home) * poisson.pmf(j, lamb_away)
                if i > j: prob_local += p
                elif i == j: prob_empate += p
                else: prob_visitante += p
        return {"local": prob_local, "empate": prob_empate, "visitante": prob_visitante}

    def analizar_valor(self, prob_real: float, cuota: float) -> tuple:
        """Calcula si hay valor esperado positivo (EV)."""
        if cuota <= 1.0: return "N/A", 0.0
        ev = (prob_real * cuota) - 1
        etiqueta = "✅ SI" if ev > 0 else "⛔ NO"
        return etiqueta, ev * 100

class AnalizadorLiga:
    """Maneja los datos históricos, limpieza y estadísticas."""
    
    def __init__(self, df):
        self.df = df
        self._limpiar()

    def _limpiar(self):
        # Estandarizamos nombres de columnas clave
        cols = {
            'HomeTeam': 'local', 'AwayTeam': 'visitante', 
            'FTHG': 'goles_local', 'FTAG': 'goles_visitante'
        }
        self.df = self.df.rename(columns=cols)
        # Nos aseguramos de tener solo las columnas necesarias y sin nulos
        needed = ['local', 'visitante', 'goles_local', 'goles_visitante']
        valid_cols = self.df.columns.intersection(needed)
        self.df = self.df[valid_cols].dropna()

    def obtener_equipos(self):
        return sorted(self.df['local'].unique())

    def obtener_fuerzas(self):
        # Cálculo vectorizado de fuerzas de ataque y defensa
        media = (self.df['goles_local'].mean() + self.df['goles_visitante'].mean()) / 2
        
        loc = self.df.groupby('local')[['goles_local', 'goles_visitante']].mean()
        vis = self.df.groupby('visitante')[['goles_visitante', 'goles_local']].mean()
        
        equipos = pd.DataFrame()
        # Ataque: Promedio de goles marcados / media liga
        equipos['f_ataque'] = ((loc['goles_local'] + vis['goles_visitante']) / 2) / media
        # Defensa: Promedio de goles recibidos / media liga
        equipos['f_defensa'] = ((loc['goles_visitante'] + vis['goles_local']) / 2) / media
        return equipos

    def obtener_racha(self, equipo: str, n: int = 5) -> str:
        """Analiza los últimos n partidos y devuelve iconos: ✅ ❌ ➖"""
        filtro = (self.df['local'] == equipo) | (self.df['visitante'] == equipo)
        partidos = self.df[filtro].tail(n) 
        
        racha = []
        for _, row in partidos.iterrows():
            es_local = row['local'] == equipo
            goles_f = row['goles_local'] if es_local else row['goles_visitante']
            goles_c = row['goles_visitante'] if es_local else row['goles_local']
            
            if goles_f > goles_c: racha.append("✅")
            elif goles_f < goles_c: racha.append("❌")
            else: racha.append("➖")
        
        return " ".join(racha)

def convertir_cuota(valor):
    """Convierte cuotas Americanas a Decimales automáticamente."""
    if valor < 0: return (100 / abs(valor)) + 1 
    elif valor >= 10: return (valor / 100) + 1  
    return valor 

# --- CAPA 2: INTERFAZ GRÁFICA (VISTA STREAMLIT) ---

st.title("⚽ Sistema de Predicción Inteligente")
st.markdown("""
Esta herramienta utiliza la **Distribución de Poisson** para calcular probabilidades reales 
y el criterio de **Kelly/Valor Esperado** para encontrar oportunidades de inversión.
""")

# 1. SIDEBAR
st.sidebar.header("📂 Base de Datos")
archivos_csv = glob.glob("*.csv")

if not archivos_csv:
    st.error("⚠️ No se encontraron archivos .csv.")
else:
    archivo_selec = st.sidebar.selectbox("Selecciona la Liga:", archivos_csv)
    
    try:
        df_raw = pd.read_csv(archivo_selec)
        analizador = AnalizadorLiga(df_raw)
        fuerzas = analizador.obtener_fuerzas()
        equipos = analizador.obtener_equipos()
        st.sidebar.success(f"Cargados {len(df_raw)} partidos.")
        
        # 2. SELECCIÓN DE EQUIPOS
        st.subheader("1️⃣ Configuración del Partido")
        col1, col2 = st.columns(2)
        with col1:
            local = st.selectbox("Equipo Local (🏠)", equipos)
        with col2:
            visitante = st.selectbox("Equipo Visitante (✈️)", [e for e in equipos if e != local])

        # 3. INPUT DE CUOTAS
        st.subheader("2️⃣ Cuotas del Mercado")
        c1, c2, c3 = st.columns(3)
        cuota_l_in = c1.number_input(f"Victoria {local}", value=0.0, step=0.1)
        cuota_e_in = c2.number_input("Empate", value=0.0, step=0.1)
        cuota_v_in = c3.number_input(f"Victoria {visitante}", value=0.0, step=0.1)

        # BOTÓN DE ACCIÓN
        if st.button("🚀 ANALIZAR OPORTUNIDADES", type="primary"):
            
            # Cálculo de Lambdas
            f_atq_l = fuerzas.loc[local, 'f_ataque']
            f_def_l = fuerzas.loc[local, 'f_defensa']
            f_atq_v = fuerzas.loc[visitante, 'f_ataque']
            f_def_v = fuerzas.loc[visitante, 'f_defensa']

            lambda_loc = f_atq_l * f_def_v * 1.3 
            lambda_vis = f_atq_v * f_def_l

            predictor = PredictorFutbol()
            probs = predictor.calcular_probabilidad_partido(lambda_loc, lambda_vis)
            
            odds_l = convertir_cuota(cuota_l_in)
            odds_e = convertir_cuota(cuota_e_in)
            odds_v = convertir_cuota(cuota_v_in)

            dec_l, val_l = predictor.analizar_valor(probs['local'], odds_l)
            dec_e, val_e = predictor.analizar_valor(probs['empate'], odds_e)
            dec_v, val_v = predictor.analizar_valor(probs['visitante'], odds_v)

            racha_l = analizador.obtener_racha(local)
            racha_v = analizador.obtener_racha(visitante)

            # --- RESULTADOS VISUALES ---
            st.divider()
            
            # RACHAS
            st.markdown("### 🔥 Forma Reciente (Últimos 5)")
            col_r1, col_r2 = st.columns(2)
            col_r1.info(f"**{local}**: {racha_l}")
            col_r2.info(f"**{visitante}**: {racha_v}")

            # MÉTRICAS
            st.markdown("### 📊 Métricas de Poder")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Ataque Local", f"{f_atq_l:.2f}")
            m2.metric("Defensa Local", f"{f_def_l:.2f}", delta_color="inverse")
            m3.metric("Ataque Visita", f"{f_atq_v:.2f}")
            m4.metric("Defensa Visita", f"{f_def_v:.2f}", delta_color="inverse")

            # TABLA DE DECISIÓN
            st.markdown("### 🏆 Decisión de Inversión")
            datos = {
                "Opción": [local, "Empate", visitante],
                "Prob Real": [f"{probs['local']*100:.1f}%", f"{probs['empate']*100:.1f}%", f"{probs['visitante']*100:.1f}%"],
                "Cuota": [f"{odds_l:.2f}", f"{odds_e:.2f}", f"{odds_v:.2f}"],
                "Valor (EV)": [f"{val_l:.2f}%", f"{val_e:.2f}%", f"{val_v:.2f}%"],
                "¿Apostar?": [dec_l, dec_e, dec_v]
            }
            
            # ESTILIZADO CONDICIONAL DE PANDAS
            df_res = pd.DataFrame(datos)
            def color_filas(val):
                color = '#d4edda' if '✅' in str(val) else '#f8d7da' if '⛔' in str(val) else ''
                return f'background-color: {color}'
            
            st.dataframe(df_res.style.map(color_filas, subset=['¿Apostar?']), use_container_width=True)

            # RECOMENDACIÓN FINAL
            mejor = max([(val_l, local), (val_e, "Empate"), (val_v, visitante)])
            if mejor[0] > 0:
                st.success(f"💎 **Oportunidad:** {mejor[1]} (Valor: +{mejor[0]:.2f}%)")
            else:
                st.warning("⚠️ No hay valor en este mercado.")

    except Exception as e:
        st.error(f"Error: {e}")
