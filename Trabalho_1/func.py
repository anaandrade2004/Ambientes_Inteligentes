import json
import numpy as np
import random
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
from lxml import etree
import time
import firebase_admin
from firebase_admin import credentials, db
import requests
import sys
import math
import select
import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from scipy.interpolate import make_interp_spline
import matplotlib
import json
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import os


matplotlib.use('Agg') 

cred = credentials.Certificate("ais2425-f5b46-firebase-adminsdk-fbsvc-fe88a13c9c.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://ais2425-f5b46-default-rtdb.europe-west1.firebasedatabase.app/"
})

def estatisticas_basicas():
    with open('combinado.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
    data = pd.DataFrame(data)
    data["temperatura"] = pd.to_numeric(data["temperatura"], errors="coerce")
    data["bpm"] = pd.to_numeric(data["bpm"], errors="coerce")
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")

    data = data.dropna(subset=["temperatura", "bpm", "timestamp"])


    temperaturas_unicas = []
    for temp in data["temperatura"]:
        if not temperaturas_unicas or temp != temperaturas_unicas[-1]:
            temperaturas_unicas.append(temp)

    temperaturas_unicas = pd.Series(temperaturas_unicas)
    

    media_temp = temperaturas_unicas.mean().round(2)
    mediana_temp = temperaturas_unicas.median().round(2)
    moda_tempera = temperaturas_unicas.mode().values.round(2)
    moda_temp = ""
    for i in moda_tempera:
        if moda_temp == "":
            moda_temp = i
        else:
            moda_temp= str(i)+", " + str(moda_temp)
    minimo_temp = temperaturas_unicas.min().round(2)
    maximo_temp = temperaturas_unicas.max().round(2)
    desvio_padrao_temp = temperaturas_unicas.std().round(2)
        

    media_bpm = data["bpm"].mean().round(2)
    mediana_bpm = data["bpm"].median().round(2)
    moda_bpms = data["bpm"].mode().values.round(2)
    moda_bpm=""
    for i in moda_bpms:
        if moda_bpm == "":
            moda_bpm = i
        else:
            moda_bpm= i+", " + moda_bpm
    minimo_bpm = data["bpm"].min().round(2)
    maximo_bpm = data["bpm"].max().round(2)
    desvio_padrao_bpm = data["bpm"].std().round(2)
    

    return {
        "bpm": {
            "Média": media_bpm,
            "Mediana": mediana_bpm,
            "Moda": moda_bpm,
            "Minimo": minimo_bpm,
            "Máximo": maximo_bpm,
            "Desvio_padrão": desvio_padrao_bpm
        },
        
        "temperatura": {
            "Média": media_temp,
            "Mediana": mediana_temp,
            "Moda": moda_temp,
            "Mínimo": minimo_temp,
            "Máximo": maximo_temp,
            "Desvio_padrão": desvio_padrao_temp
        }
        
    }
    



def criar_histograma_bpm():
    with open('combinado.json', 'r', encoding='utf-8') as f:
        dados_json = json.load(f)
        
    data = pd.DataFrame(dados_json)
    data["bpm"] = pd.to_numeric(data["bpm"], errors="coerce")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(data["bpm"].dropna(), bins=20, color='skyblue', edgecolor='black')
    ax.set_title("Histograma de BPM")
    ax.set_xlabel("BPM")
    ax.set_ylabel("Frequência")
    ax.grid(True)

    buffer = BytesIO()
    plt.tight_layout()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    imagem_b64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)

    return imagem_b64


def criar_grafico_temperatura():
    with open('combinado.json', 'r', encoding='utf-8') as f:
        dados_json = json.load(f)
        
    data = pd.DataFrame(dados_json)
    data["temperatura"] = pd.to_numeric(data["temperatura"], errors="coerce")

    if data["temperatura"].notna().sum() == 0:
        return None

    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data["timestamp_minuto"] = data["timestamp"].dt.floor('min')
    temperatura_unica = data.groupby("timestamp_minuto")["temperatura"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(temperatura_unica["timestamp_minuto"], temperatura_unica["temperatura"],
            marker='o', linestyle='-', color='salmon')
    ax.set_title("Temperatura ao longo do tempo")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Temperatura (°C)")
    ax.grid(True, linestyle="--", linewidth=0.5)
    plt.xticks(rotation=45)

    buffer = BytesIO()
    plt.tight_layout()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    imagem_b64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)

    return imagem_b64


def criar_grafico_bpm_por_dia(dia_str):
    with open('combinado.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Converter para DataFrame e tratar dados
    df = pd.DataFrame(data)
    df['bpm'] = pd.to_numeric(df['bpm'], errors='coerce')
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['bpm', 'timestamp'])

    # Filtrar apenas para o dia solicitado
    try:
        dia = pd.to_datetime(dia_str).date()
    except Exception:
        raise ValueError("Formato de data inválido. Use 'YYYY-MM-DD'.")
    mask = df['timestamp'].dt.date == dia
    df_dia = df.loc[mask].sort_values('timestamp')
    if df_dia.empty:
        print(f"Não há dados de BPM para o dia {dia_str}.")
        return None

    # Agregação por minuto
    df_dia = df_dia.set_index('timestamp')
    agg = df_dia['bpm'].resample('1T').agg(['min','mean','max']).dropna()
    if agg.empty:
        print("Após agregação de 1 minuto, não restaram dados para plotar.")
        return None

    # Plotagem
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(agg.index, agg['mean'], marker='o', linestyle='-', label='Média (1-min)')
    ax.fill_between(agg.index, agg['min'], agg['max'], alpha=0.3, label='Intervalo (mín–máx)')
    ax.set_title(f"Variação de BPM em {dia_str} (média por minuto)")
    ax.set_xlabel("Hora")
    ax.set_ylabel("BPM")
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    ax.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()

    # Salvar imagem em base64
    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    imagem_b64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)

    return imagem_b64


def criar_grafico_combinado():

    with open('combinado.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["bpm"] = pd.to_numeric(df["bpm"], errors="coerce")
    df["temperatura"] = pd.to_numeric(df["temperatura"], errors="coerce")
    df["Date"] = df["timestamp"].dt.date.astype(str)

    temperatura_data = df.dropna(subset=["temperatura", "cidade"]).copy()
    df = df.sort_values("timestamp")
    temperatura_data = temperatura_data.sort_values("timestamp")

    fig, ax1 = plt.subplots(figsize=(12, 6))

    if not temperatura_data.empty:
        sns.lineplot(data=temperatura_data, x="timestamp", y="temperatura",
                        hue="cidade", marker="o", ax=ax1, legend='full')
        ax1.set_ylabel("Temperatura (°C)", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        leg1 = ax1.legend(title="Cidade", loc='upper left', bbox_to_anchor=(0, 1.15))
        ax1.add_artist(leg1)

    ax2 = ax1.twinx()
    sns.lineplot(data=df, x="timestamp", y="bpm", hue="Date",
                    marker="o", ax=ax2, palette="coolwarm", legend='full')
    ax2.set_ylabel("BPM", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.legend(title="Data", loc='upper right', bbox_to_anchor=(1, 1.15), ncol=2)

    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %H:%M"))
    ax1.tick_params(axis='x', rotation=45)

    plt.title("Variação de Temperatura e BPM ao longo do tempo", pad=20)
    plt.grid(True, linestyle="--", linewidth=0.5)
    fig.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    imagem_b64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)
    return imagem_b64

    
def criar_matriz_correlacao():
    with open('combinado.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df["bpm"] = pd.to_numeric(df["bpm"], errors="coerce")
    df["temperatura"] = pd.to_numeric(df["temperatura"], errors="coerce")
    df_corr = df[["bpm", "temperatura"]].dropna()

    if df_corr.empty:
        return None

    corr_matrix = df_corr.corr()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True, ax=ax)
    ax.set_title("Matriz de Correlação entre Temperatura e BPM")
    fig.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    imagem_b64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)
    return imagem_b64


def calcular_correlacao_person():
    with open('combinado.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df["bpm"] = pd.to_numeric(df["bpm"], errors="coerce")
    df["temperatura"] = pd.to_numeric(df["temperatura"], errors="coerce")
    df_corr = df.dropna(subset=["bpm", "temperatura"])

    if df_corr.empty:
        return None, None

    coef, p_valor = pearsonr(df_corr["temperatura"], df_corr["bpm"])
    return round(coef, 3), round(p_valor, 3)


def grafico_barras_spline_bpm():

    with open('combinado.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    df["DataHora"] = pd.to_datetime(df["timestamp"])
    df["Data"] = df["DataHora"].dt.date
    df["BPM"] = pd.to_numeric(df["bpm"], errors='coerce')
    bpm_daily_avg = df.groupby("Data")["BPM"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(bpm_daily_avg["Data"].astype(str), bpm_daily_avg["BPM"], color='lightgreen')

    if len(bpm_daily_avg) >= 4:
        x = np.arange(len(bpm_daily_avg))
        x_smooth = np.linspace(x.min(), x.max(), 300)
        y_smooth = make_interp_spline(x, bpm_daily_avg["BPM"])(x_smooth)
        ax.plot(x_smooth, y_smooth, color='black')
    else:
        print("Poucos pontos para spline.")

    ax.set_title("Evolução da média do BPM ao longo dos dias")
    ax.set_xlabel("Data")
    ax.set_ylabel("Média BPM")
    plt.xticks(rotation=45)
    fig.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)
    return img_base64


def grafico_boxplot_bpm():

    with open('combinado.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    df["DataHora"] = pd.to_datetime(df["timestamp"])
    df["Data"] = df["DataHora"].dt.date
    df["BPM"] = pd.to_numeric(df["bpm"], errors='coerce')

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="Data", y="BPM", palette="Set3", ax=ax)
    ax.set_title("Distribuição de BPM por Dia")
    ax.set_xlabel("Data")
    ax.set_ylabel("BPM")
    plt.xticks(rotation=45)
    fig.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)
    return img_base64


def grafico_heatmap_bpm():

    with open('combinado.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df["DataHora"] = pd.to_datetime(df["timestamp"])
    df["Data"] = df["DataHora"].dt.date
    df["Hora"] = df["DataHora"].dt.hour
    df["BPM"] = pd.to_numeric(df["bpm"], errors='coerce')

    heatmap_data = df.groupby(["Data", "Hora"])["BPM"].mean().unstack()

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(heatmap_data, cmap="YlOrRd", annot=True, fmt=".1f", ax=ax)
    ax.set_title("Heatmap da Média de BPM por Hora e Dia")
    ax.set_xlabel("Hora do Dia")
    ax.set_ylabel("Data")
    fig.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)
    return img_base64


def grafico_comparativo_bpm_temp():
    with open('combinado.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df["DataHora"] = pd.to_datetime(df["timestamp"])
    df["Data"] = df["DataHora"].dt.date
    df["BPM"] = pd.to_numeric(df["bpm"], errors='coerce')
    df["Temperatura"] = pd.to_numeric(df["temperatura"], errors='coerce')

    bpm_avg = df.groupby("Data")["BPM"].mean().reset_index()
    temp_avg = df.groupby("Data")["Temperatura"].mean().reset_index()
    bpm_avg["Data"] = bpm_avg["Data"].astype(str)
    temp_avg["Data"] = temp_avg["Data"].astype(str)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    color1 = 'tab:green'
    ax1.set_xlabel("Data")
    ax1.set_ylabel("BPM", color=color1)
    ax1.plot(bpm_avg["Data"], bpm_avg["BPM"], color=color1, marker='o')
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel("Temperatura (°C)", color=color2)
    ax2.plot(temp_avg["Data"], temp_avg["Temperatura"], color=color2, marker='x')
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title("Comparação BPM e Temperatura Média Diária")
    plt.xticks(rotation=45)
    fig.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)
    return img_base64


def grafico_scatter_temp_vs_bpm():
    with open('combinado.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df["Temperatura"] = pd.to_numeric(df["temperatura"], errors='coerce')
    df["BPM"] = pd.to_numeric(df["bpm"], errors='coerce')

    fig, ax = plt.subplots(figsize=(8, 6))
    if "cidade" in df.columns and df["cidade"].notna().any():
        sns.scatterplot(data=df, x="Temperatura", y="BPM", hue="cidade", palette="coolwarm", ax=ax)
    else:
        sns.scatterplot(data=df, x="Temperatura", y="BPM", color="steelblue", ax=ax)

    ax.set_title("Relação entre Temperatura e BPM")
    ax.set_xlabel("Temperatura (°C)")
    ax.set_ylabel("BPM")
    fig.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)
    return img_base64

def grafico_serie_temporal_bpm_temp():
    with open('combinado.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df["DataHora"] = pd.to_datetime(df["timestamp"], errors='coerce')
    df["BPM"] = pd.to_numeric(df["bpm"], errors='coerce')
    df["Temperatura"] = pd.to_numeric(df["temperatura"], errors='coerce')

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=df, x="DataHora", y="BPM", label="BPM", color="green", ax=ax)
    sns.lineplot(data=df, x="DataHora", y="Temperatura", label="Temperatura", color="red", ax=ax)

    ax.set_title("Evolução Temporal de BPM e Temperatura")
    ax.set_xlabel("Data e Hora")
    plt.xticks(rotation=45)
    ax.legend()
    fig.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)
    return img_base64

def grafico_histograma_bpm():

        with open('combinado.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        df["BPM"] = pd.to_numeric(df["bpm"], errors='coerce')

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df["BPM"].dropna(), bins=30, kde=True, color="skyblue", ax=ax)
        ax.set_title("Distribuição dos BPMs")
        ax.set_xlabel("BPM")
        ax.set_ylabel("Frequência")
        fig.tight_layout()

        buffer = BytesIO()
        fig.savefig(buffer, format="png")
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        return img_base64


def grafico_temp_media_por_cidade():

        with open('combinado.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        df["Temperatura"] = pd.to_numeric(df["temperatura"], errors='coerce')

        if "cidade" not in df.columns or df["cidade"].isna().all():
            return None

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=df, x="cidade", y="Temperatura", estimator=np.mean, ci=None, palette="viridis", ax=ax)
        ax.set_title("Temperatura Média por Cidade")
        ax.set_ylabel("Temperatura (°C)")
        plt.xticks(rotation=45)
        fig.tight_layout()

        buffer = BytesIO()
        fig.savefig(buffer, format="png")
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        return img_base64
    
    
    
def extrair_dados_xml():
    caminho_xml = 'static/data/data_bpm.xml'

    with open(caminho_xml, 'r', encoding='utf-8') as arquivo:
        conteudo = arquivo.read()

    conteudo_com_raiz = f"<Records>{conteudo}</Records>"

    parser = etree.XMLParser(recover=True)
    root = etree.fromstring(conteudo_com_raiz.encode('utf-8'), parser=parser)

    dados_por_data = {}

    for record in root.findall('Record'):
        creation_date_full = record.get('creationDate')
        if creation_date_full is None:
            continue

        date_key = creation_date_full.split(" ")[0]

        if date_key not in dados_por_data:
            dados_por_data[date_key] = []

        metadata_list = record.find('HeartRateVariabilityMetadataList')
        if metadata_list is not None:
            for beat in metadata_list.findall('InstantaneousBeatsPerMinute'):
                bpm = beat.get('bpm')
                time_extraction = beat.get('time')
                dados_por_data[date_key].append({
                    "bpm": bpm,
                    "time_extraction": time_extraction
                })

    with open('dados_extraidos.json', 'w', encoding='utf-8') as json_file:
        json.dump(dados_por_data, json_file, indent=4, ensure_ascii=False)

    print("Extração concluída. Dados guardados em 'dados_extraidos.json'.")

    os.remove(caminho_xml)
    print(f"Ficheiro {caminho_xml} apagado após extração.")

    return True



def enviar_bpm_para_firebase():
    caminho_json = "dados_extraidos.json"

    with open(caminho_json, "r", encoding="utf-8") as f:
        dados_por_data = json.load(f)

    ref = db.reference("BPMs")
    dados_firebase = ref.get() or {}

    datas_existentes = set()
    for item in dados_firebase.values():
        if 'Date' in item:
            datas_existentes.add(item['Date'])

    novos_dados = 0
    for date_key, entries in dados_por_data.items():
        if date_key not in datas_existentes:
            data_entry = {"Date": date_key, "Entries": entries}
            ref.push(data_entry)
            print(f"Dados para {date_key} enviados para o Firebase.")
            novos_dados += 1
        else:
            print(f"Dados para {date_key} já existem. Ignorado.")

    print(f"\n{novos_dados} novas datas enviadas." if novos_dados > 0 else "\nNada novo a enviar.")
    os.remove(caminho_json)
    print(f"Ficheiro {caminho_json} apagado após envio.")

    return True



def get_dados_temp():
        res = {}
        ref = db.reference("Temperaturas")
        dados = ref.get()            
        for cidade in dados:
            print(f"Processando dados de: {cidade}")
            for i in dados[cidade]:
                date_obj = datetime.strptime(dados[cidade][i]['Date'], '%d/%m/%Y %H:%M')
                res[str(date_obj)] = {
                    "cidade": cidade,
                    "Temperatura": dados[cidade][i]['Temperatura']
                }

        res = dict(sorted(res.items()))
            
        return res


def get_dados_bpm():
        res = []
        ref = db.reference("BPMs")
        dados = ref.get()
        for id in dados:
            res.append(dados[id])
            
        return res


def combinar_bpm_temp():
        bpm_data = get_dados_bpm()
        temp_data = get_dados_temp()
        resultado = []
        temperaturas = {
            datetime.strptime(k, "%Y-%m-%d %H:%M:%S"): v for k, v in temp_data.items()
        }

        for registo in bpm_data:
            data_str = registo["Date"] 
            for entrada in registo["Entries"]:
                hora_raw = entrada["time_extraction"]  
                hora_corrigida = hora_raw.replace(",", ".")  

                timestamp_str = f"{data_str} {hora_corrigida}"
                try:
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
                except ValueError:
                    timestamp = datetime.strptime(f"{data_str} {hora_raw.split(',')[0]}", "%Y-%m-%d %H:%M:%S")

                temperatura_mais_proxima = None
                menor_diferenca = timedelta(minutes=30)
                for temp_time, temp_info in temperaturas.items():
                    diferenca = abs(timestamp - temp_time)
                    if diferenca < menor_diferenca:
                        menor_diferenca = diferenca
                        temperatura_mais_proxima = temp_info

                resultado.append({
                    "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "bpm": entrada["bpm"],
                    "temperatura": temperatura_mais_proxima["Temperatura"] if temperatura_mais_proxima else None,
                    "cidade": temperatura_mais_proxima["cidade"] if temperatura_mais_proxima else None
                })
        resultado = sorted(resultado, key=lambda x: datetime.strptime(x["timestamp"], "%Y-%m-%d %H:%M:%S"))
        with open("combinado.json", "w", encoding="utf-8") as f:
            json.dump(resultado, f, indent=4, ensure_ascii=False)
            
        print("\nDados combinados com sucesso! Resultados salvos em 'combinado.json'")
        return 


def gerar_variacao_temperatura(dia, mes, ano, cidade, temp_min, temp_max):
    try:
        data = datetime(ano, mes, dia)
        amplitude = temp_max - temp_min

        for hora in range(24):
            for minuto in [0]:
                progresso = hora / 24 + minuto / (24 * 60)
                offset = 6/24
                temp_variacao = math.sin((progresso - offset) * 2 * math.pi)
                temp_variacao = temp_variacao * 0.5 + 0.5
                temp_variacao = math.pow(temp_variacao, 1.5)
                temperatura = temp_min + (temp_variacao * amplitude)
                temperatura += (random.random() - 0.5) * 0.3
                temperatura = round(temperatura, 1)

                timestamp = data + timedelta(hours=hora, minutes=minuto)
                timestamp_str = timestamp.strftime("%d/%m/%Y %H:%M")

                previsao = {"Date": timestamp_str, "Temperatura": temperatura}
                ref = db.reference(f"Temperaturas/{cidade}")
                ref.push(previsao)

        print(f"Dados de temperatura para {cidade} gerados e enviados com sucesso.")
        return True
    except Exception as e:
        return False


melhor_modelo = None
melhor_modelo_nome = ""
scaler = None

def modelo():
    global melhor_modelo, melhor_modelo_nome, scaler
    try:
        with open('combinado.json', 'r', encoding='utf-8') as f:
            data = json.load(f)

        df = pd.DataFrame(data)
        df["bpm"] = pd.to_numeric(df["bpm"], errors="coerce")
        df["temperatura"] = pd.to_numeric(df["temperatura"], errors="coerce")
        df = df.dropna(subset=["bpm", "temperatura"])

        X = df[["temperatura"]]
        y = df["bpm"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Normalização
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        modelos = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
            "SVR": SVR(),
            "KNeighbors": KNeighborsRegressor(),
            "MLP Regressor": MLPRegressor(random_state=42, max_iter=1000)
        }

        resultados = []

        for nome, modelo in modelos.items():
            if nome in ["SVR", "MLP Regressor"]:
                modelo.fit(X_train_scaled, y_train)
                y_pred = modelo.predict(X_test_scaled)
            else:
                modelo.fit(X_train, y_train)
                y_pred = modelo.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            resultados.append({
                "Modelo": nome,
                "R²": round(r2, 3),
                "MAE": round(mae, 2),
                "RMSE": round(rmse, 2)
            })

        resultados_df = pd.DataFrame(resultados).sort_values(by="R²", ascending=False)
        resultados_df.to_csv("resultados_modelos.csv", index=False)

        melhor_modelo_nome = resultados_df.iloc[0]["Modelo"]
        melhor_modelo = modelos[melhor_modelo_nome]

        if melhor_modelo_nome in ["SVR", "MLP Regressor"]:
            melhor_modelo.fit(scaler.transform(X), y)
        else:
            melhor_modelo.fit(X, y)

        print(f"\nMelhor modelo selecionado: {melhor_modelo_nome}")
        return True
    except Exception as e:
        return False
    


def enviar_pushbullet(titulo, mensagem):
    TOKEN = "o.AmOwcnCN8LNWGjtJagSvXf1etTwCGHCt"
    url = "https://api.pushbullet.com/v2/pushes"
    headers = {
        "Access-Token": TOKEN,
        "Content-Type": "application/json"
    }
    data = {
        "type": "note",
        "title": titulo,
        "body": mensagem
    }
    response = requests.post(url, json=data, headers=headers)
    if response.status_code != 200:
        print("Erro ao enviar Pushbullet:", response.text)
    else:
        print("Notificação Pushbullet enviada!")
        


def obter_anomalias(melhor_modelo, melhor_modelo_nome, scaler):
    try:
        with open('combinado.json', 'r', encoding='utf-8') as f:
            data = json.load(f)

        df = pd.DataFrame(data)
        df["bpm"] = pd.to_numeric(df["bpm"], errors="coerce")
        df["temperatura"] = pd.to_numeric(df["temperatura"], errors="coerce")
        df = df.dropna(subset=["bpm", "temperatura", "timestamp"])

        df["timestamp"] = pd.to_datetime(df["timestamp"])

        X = df[["temperatura"]]
        y_real = df["bpm"]

        if melhor_modelo is None:
            return []

        if melhor_modelo_nome in ["SVR", "MLP Regressor"]:
            X_scaled = scaler.transform(X)
            y_pred = melhor_modelo.predict(X_scaled)
        else:
            y_pred = melhor_modelo.predict(X)

        df["previsto"] = y_pred
        df["erro_abs"] = abs(df["bpm"] - df["previsto"])
        df_anomalias = df[df["erro_abs"] > 10].copy()

        df_anomalias["intervalo_60min"] = df_anomalias["timestamp"].dt.floor("60min")
        df_unicos = df_anomalias.groupby("intervalo_60min").first().reset_index()

        lista_anomalias = df_unicos[["timestamp", "bpm", "temperatura"]].to_dict(orient="records")

        if not df_unicos.empty:
            primeira = df_unicos.iloc[-1]
            bpm = primeira["bpm"]
            temp = primeira["temperatura"]
            timestamp = primeira["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            mensagem = f"BPM: {bpm}, Temp: {temp}, Hora: {timestamp}"
            enviar_pushbullet("⚠️ Anomalia Detetada", mensagem)

        return lista_anomalias

    except Exception as e:
        return []



def aumentar_e_enviar_bpm_sintetico(data_escolhida, modo="normal"):
    try:
        with open("combinado.json", "r", encoding="utf-8") as f:
            dados = json.load(f)

        dados_data = [d for d in dados if d["timestamp"].startswith(data_escolhida)]
        dados_sinteticos = []

        if not dados_data:
            print(f"Nenhum dado real encontrado para {data_escolhida}. Gerando dados do zero.")
            
            hora_inicial = random.randint(0, 12)
            for bloco in range(3):
                hora_bloco = hora_inicial + bloco * 3
                start_time = datetime.strptime(f"{data_escolhida} {hora_bloco:02d}:00:00", "%Y-%m-%d %H:%M:%S")
                num_registos = random.randint(8, 15)
                bpm_base = np.random.normal(loc=75, scale=5, size=num_registos)
                indices_anomalias = set(random.sample(range(num_registos), k=random.randint(1, 3)))
                current_time = start_time

                for i, bpm in enumerate(bpm_base):
                    current_time += timedelta(seconds=random.randint(60, 300))
                    valor_bpm = int(bpm)

                    if modo == "anomalia" and i in indices_anomalias:
                        tipo = random.choice(["alta", "baixa"])
                        valor_bpm = random.randint(150, 200) if tipo == "alta" else random.randint(30, 50)

                    dados_sinteticos.append({
                        "bpm": str(valor_bpm),
                        "time_extraction": current_time.strftime("%H:%M:%S,%f")[:-4]
                    })

        else:
            print(f"Dados reais encontrados para {data_escolhida}. Usando interpolação.")

            dados_data.sort(key=lambda x: x["timestamp"])
            bpm_list = [int(d["bpm"]) for d in dados_data]
            timestamps = [datetime.strptime(d["timestamp"], "%Y-%m-%d %H:%M:%S") for d in dados_data]

            indices = np.arange(len(bpm_list))
            bpm_array = np.array(bpm_list)

            interp_func = interp1d(indices, bpm_array, kind="linear", fill_value="extrapolate")
            novos_indices = np.linspace(0, len(bpm_list) - 1, len(bpm_list) * 2)
            bpm_sintetico = interp_func(novos_indices) + np.random.normal(0, 2, len(novos_indices))

            bpm_min = min(bpm_list)
            bpm_max = max(bpm_list)
            ultimo_timestamp = max(timestamps)

            def novo_timestamp(base):
                return base + timedelta(seconds=random.randint(60, 300))

            current_time = ultimo_timestamp
            num_novos = len(bpm_sintetico[len(bpm_list):])
            indices_anomalias = set(random.sample(range(num_novos), k=random.randint(1, 3)))

            for i, bpm in enumerate(bpm_sintetico[len(bpm_list):]):
                current_time = novo_timestamp(current_time)
                valor_bpm = int(bpm)

                if modo == "anomalia" and i in indices_anomalias:
                    tipo = random.choice(["alta", "baixa"])
                    valor_bpm = random.randint(bpm_max + 20, bpm_max + 50) if tipo == "alta" else random.randint(bpm_min - 40, bpm_min - 10)

                dados_sinteticos.append({
                    "bpm": str(valor_bpm),
                    "time_extraction": current_time.strftime("%H:%M:%S,%f")[:-4]
                })

        # Enviar para Firebase
        ref = db.reference("BPMs")
        dados_firebase = ref.get() or {}
        datas_existentes = {item.get("Date") for item in dados_firebase.values() if isinstance(item, dict)}

        if data_escolhida in datas_existentes:
            print(f"Dados para {data_escolhida} já existem na Firebase.")
            return False

        ref.push({
            "Date": data_escolhida,
            "Entries": dados_sinteticos
        })

        print(f"{len(dados_sinteticos)} BPMs {'com anomalias' if modo == 'anomalia' else 'normais'} enviados para {data_escolhida}.")
        return True

    except Exception as e:
        print(f"Erro ao gerar/enviar BPMs: {e}")
        return False

