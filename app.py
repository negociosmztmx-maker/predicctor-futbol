import streamlit as st
import pandas as pd
from scipy.stats import poisson
import requests
import glob

# --- CONFIGURACIÓN GLOBAL ---
st.set_page_config(
    page_title="Predicctor Pro (Hybrid)", 
    page_icon="⚽", 
    layout="wide"
)

# --- CAPA 1: MOTOR MATEMÁTICO (CORE) ---
class PredictorFutbol:
    """Cerebro matemático: Poisson y Kelly."""
    
    def calcular_probabilidades(self, lamb_home, lamb_away):
        prob_local, prob_empate, prob_visitante = 0.0, 0.0, 0.0
        # Iteramos marcadores probables (0-0 a 9-9)
        for i in range(10):
            for j in range(10):
                weight = poisson.pmf(i, lamb_home) * poisson.pmf(j, lamb_away)
                if i > j: prob_local += weight
                elif i == j: prob_empate += weight
                else: prob_visitante += weight
        return {"local": prob_local, "empate": prob_empate, "visitante": prob_visitante}

    def analizar_valor(self, prob_real, cuota):
        """Calcula Valor Esperado (EV)."""
        if cuota <= 1.0: return "N/A", 0.0
        ev = (prob_real * cuota) - 1
        etiqueta = "✅ SI" if ev > 0 else "⛔ NO"
        return etiqueta, ev * 100

def convertir_cuota(valor):
    """Detecta y convierte cuotas Americanas a Decimales."""
    if valor < 0: return (100 / abs(valor)) + 1 # Americana Negativa (-110)
    elif valor >= 10: return (valor / 100) + 1  # Americana Positiva (+200)
    return valor # Decimal normal

# --- CAPA 2: INGESTIÓN DE DATOS (ETL) ---

@st.cache_data(ttl=3600) # Cache de 1 hora para no saturar la API
def cargar_desde_api(liga_code):
    """Conecta a football-data.org y devuelve un DataFrame limpio."""
    try:
        api_key = st.secrets["FOOTBALL_API_KEY"]
    except FileNotFoundError:
        return None, "❌ Falta configurar .streamlit/secrets.toml"
    except KeyError:
        return None, "❌ La clave FOOTBALL_API_KEY no está en secrets.toml"

    headers = {'X-Auth-Token': api_key}
    url = f"https://api.football-data.org/v4/competitions/{liga_code}/matches?status=FINISHED"
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return None, f"Error API: {response.status_code} - {response.reason}"

    data = response.json()
    matches = data.get('matches', [])
    
    if not matches:
        return None, "La API no devolvió partidos (¿Quizás la temporada apenas inicia?)"

    # Normalizamos a nuestra estructura estándar
    datos_limpios = []
    for m in matches:
        datos_limpios.append({
            'local': m['homeTeam']['name'],
            'visitante': m['awayTeam']['name'],
            'goles_local': m['score']['fullTime']['home'],
            'goles_visitante': m['score']['fullTime']['away']
        })
    
    return pd.DataFrame(datos_limpios), "OK"

def cargar_desde_csv(ruta_archivo):
    """Carga CSV local y normaliza columnas."""
    try:
        df = pd.read_csv(ruta_archivo)
        # Mapeo de nombres estándar de football-data.co.uk
        rename_map = {
            'HomeTeam': 'local', 'AwayTeam': 'visitante',
            'FTHG': 'goles_local', 'FTAG': 'goles_visitante'
        }
        df = df.rename(columns=rename_map)
        # Filtramos solo lo necesario
        required = ['local', 'visitante', 'goles_local', 'goles_visitante']
        return df[required].dropna(), "OK"
    except Exception as e:
        return None, str(e)

# --- CAPA 3: ANÁLISIS DE DATOS ---
class AnalizadorLiga:
    def __init__(self, df):
        self.df = df # Asumimos que el DF ya viene limpio y normalizado

    def obtener_equipos(self):
        equipos = set(self.df['local'].unique()) | set(self.df['visitante'].unique())
        return sorted(list(equipos))

    def obtener_fuerzas(self):
        # Medias globales
        mean_g_home = self.df['goles_local'].mean()
        mean_g_away = self.df['goles_visitante'].mean()
        media_liga_total = (mean_g_home + mean_g_away) / 2

        # Agrupaciones
        home_stats = self.df.groupby('local')[['goles_local', 'goles_visitante']].mean()
        away_stats = self.df.groupby('visitante')[['goles_visitante', 'goles_local']].mean()

        equipos = pd.DataFrame()
        
        # Fuerza Ataque: (Goles metidos en casa + Goles metidos fuera) / 2 / media liga
        equipos['f_ataque'] = ((home_stats['goles_local'] + away_stats['goles_visitante']) / 2) / media_liga_total
        
        # Fuerza Defensa: (Goles recibidos en casa + Goles recibidos fuera) / 2 / media liga
        equipos['f_defensa'] = ((home_stats['goles_visitante'] + away_stats['goles_local']) / 2) / media_liga_total
        
        # Rellenar NaN con 1.0 (promedio) por si faltan datos
        return equipos.fillna(1.0)

    def obtener_racha(self, equipo, n=5):
        # Filtro de partidos del equipo
        mask = (self.df['local'] == equipo) | (self.df['visitante'] == equipo)
        last_n = self.df[mask].tail(n)
        
        racha_iconos = []
        for _, row in last_n.iterrows():
            es_local = row['local'] == equipo
            gf = row['goles_local'] if es_local else row['goles_visitante']
            gc = row['goles_visitante'] if es_local else row['goles_local']
            
            if gf > gc: racha_iconos.append("✅")
            elif gf < gc: racha_iconos.append("❌")
            else: racha_iconos.append("➖")
            
        return " ".join(racha_iconos) if racha_iconos else "Sin datos"

# --- CAPA 4: INTERFAZ DE USUARIO (GUI) ---

st.title("📡 Sistema de Predicción Híbrido")
st.markdown("Algoritmo de **Poisson** conectado a datos en tiempo real (API) o históricos (CSV).")

# --- BARRA LATERAL: SELECCIÓN DE FUENTE ---
st.sidebar.header("Configuración de Datos")
modo_datos = st.sidebar.radio("Fuente de Datos:", ["☁️ API (En Vivo)", "📂 CSV (Local)"])

df_activo = None
mensaje_status = ""

if modo_datos == "☁️ API (En Vivo)":
    # Diccionario de ligas soportadas por el plan gratis
    LIGAS_API = {
        "Premier League (ING)": "PL",
        "La Liga (ESP)": "PD",
        "Serie A (ITA)": "SA",
        "Bundesliga (ALE)": "BL1",
        "Ligue 1 (FRA)": "FL1",
        "Eredivisie (HOL)": "DED",
        "Championship (ING 2)": "ELC",
        "Primeira Liga (POR)": "PPL",
        "Libertadores (SUD)": "CLI" 
    }
    liga_sel = st.sidebar.selectbox("Selecciona Competición:", list(LIGAS_API.keys()))
    
    with st.spinner(f"Conectando satélites para {liga_sel}..."):
        df_activo, mensaje = cargar_desde_api(LIGAS_API[liga_sel])
        mensaje_status = mensaje

else: # MODO CSV
    archivos = glob.glob("*.csv")
    if not archivos:
        st.sidebar.error("No hay archivos .csv en la carpeta.")
        mensaje_status = "Error: Faltan CSVs"
    else:
        archivo_sel = st.sidebar.selectbox("Archivo CSV:", archivos)
        df_activo, mensaje = cargar_desde_csv(archivo_sel)
        mensaje_status = mensaje

# --- LÓGICA PRINCIPAL ---

if df_activo is None or df_activo.empty:
    st.warning(f"⚠️ No se pudieron cargar datos. Estado: {mensaje_status}")
    if modo_datos == "☁️ API (En Vivo)":
        st.info("💡 Consejo: Verifica que tengas el archivo `.streamlit/secrets.toml` creado con tu API Key.")
else:
    # Si llegamos aquí, tenemos datos limpios en df_activo
    st.success(f"Datos cargados correctamente: {len(df_activo)} partidos procesados.")
    
    # Instanciamos el analizador
    analizador = AnalizadorLiga(df_activo)
    fuerzas = analizador.obtener_fuerzas()
    equipos = analizador.obtener_equipos()

    st.divider()
    
    # 1. SELECTORES DE EQUIPOS
    col1, col2 = st.columns(2)
    with col1:
        local = st.selectbox("🏠 Equipo Local", equipos)
    with col2:
        # Quitamos al local de la lista de visitantes
        visitante = st.selectbox("✈️ Equipo Visitante", [e for e in equipos if e != local])

    # 2. CUOTAS
    st.markdown("### 💰 Cuotas del Mercado")
    c1, c2, c3 = st.columns(3)
    cuota_l_raw = c1.number_input(f"Cuota {local}", 0.0, step=0.1)
    cuota_e_raw = c2.number_input("Cuota Empate", 0.0, step=0.1)
    cuota_v_raw = c3.number_input(f"Cuota {visitante}", 0.0, step=0.1)

    # 3. BOTÓN DE ANÁLISIS
    if st.button("🚀 Calcular Probabilidades y Valor", type="primary"):
        predictor = PredictorFutbol()
        
        # Obtener fuerzas (si es equipo nuevo sin datos, usamos 1.0 por defecto)
        f_atq_l = fuerzas.loc[local, 'f_ataque'] if local in fuerzas.index else 1.0
        f_def_l = fuerzas.loc[local, 'f_defensa'] if local in fuerzas.index else 1.0
        f_atq_v = fuerzas.loc[visitante, 'f_ataque'] if visitante in fuerzas.index else 1.0
        f_def_v = fuerzas.loc[visitante, 'f_defensa'] if visitante in fuerzas.index else 1.0

        # Cálculo de Lambdas (Goles Esperados)
        # Factor Campo: 1.25 (Ventaja estadística estándar)
        lambda_local = f_atq_l * f_def_v * 1.25
        lambda_visitante = f_atq_v * f_def_l

        # Probabilidades
        probs = predictor.calcular_probabilidades(lambda_local, lambda_visitante)
        
        # Conversión de Cuotas
        od_l = convertir_cuota(cuota_l_raw)
        od_e = convertir_cuota(cuota_e_raw)
        od_v = convertir_cuota(cuota_v_raw)

        # Análisis de Valor (EV)
        tag_l, ev_l = predictor.analizar_valor(probs['local'], od_l)
        tag_e, ev_e = predictor.analizar_valor(probs['empate'], od_e)
        tag_v, ev_v = predictor.analizar_valor(probs['visitante'], od_v)

        # Rachas
        racha_l = analizador.obtener_racha(local)
        racha_v = analizador.obtener_racha(visitante)

        # --- RESULTADOS VISUALES ---
        st.divider()
        
        # Encabezado Rachas
        st.subheader("🔥 Momentum")
        rc1, rc2 = st.columns(2)
        rc1.info(f"**{local}**: {racha_l}")
        rc2.info(f"**{visitante}**: {racha_v}")

        # Métricas
        st.subheader("📊 Métricas (Ataque/Defensa)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Ataque Local", f"{f_atq_l:.2f}")
        m2.metric("Defensa Local", f"{f_def_l:.2f}", delta_color="inverse")
        m3.metric("Ataque Visita", f"{f_atq_v:.2f}")
        m4.metric("Defensa Visita", f"{f_def_v:.2f}", delta_color="inverse")

        # Tabla Final
        st.subheader("🏆 Decisión de Inversión")
        datos_tabla = {
            "Resultado": [local, "Empate", visitante],
            "Prob. Real": [f"{probs['local']*100:.1f}%", f"{probs['empate']*100:.1f}%", f"{probs['visitante']*100:.1f}%"],
            "Cuota (Dec)": [f"{od_l:.2f}", f"{od_e:.2f}", f"{od_v:.2f}"],
            "Valor (EV)": [f"{ev_l:.2f}%", f"{ev_e:.2f}%", f"{ev_v:.2f}%"],
            "¿Apostar?": [tag_l, tag_e, tag_v]
        }
        
        df_res = pd.DataFrame(datos_tabla)
        
        # Estilos condicionales
        def colorear(val):
            if "✅" in str(val): return 'background-color: #d4edda; color: black'
            if "⛔" in str(val): return 'background-color: #f8d7da; color: black'
            return ''

        st.table(df_res.style.map(colorear, subset=['¿Apostar?']))

        # Mensaje destacado
        mejor_ev = max(ev_l, ev_e, ev_v)
        if mejor_ev > 0:
            st.balloons()
            st.success(f"💎 **OPORTUNIDAD DETECTADA:** Hay valor positivo de +{mejor_ev:.2f}% en este mercado.")
        else:
            st.warning("⚠️ El mercado está bien ajustado o las cuotas son bajas. No hay valor matemático.")
