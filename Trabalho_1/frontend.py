from flask import Flask, render_template, request, redirect, url_for, make_response
import os
import json
import pandas as pd
import io
import json
app = Flask(__name__)
import func
import threading
import time


graficos_cache = {
    "bpm": None,
    "temp": None,
    "bpm_dia": None,
    "grafico_combinado": None,
    "grafico_correlacao": None,
    "correlacao": (None, None),
    "barras_spline_bpm": None,
    "heatmap_bpm": None,
    "comparativo_bpm_temp": None,
    "scatter_temp_bpm": None,
    "serie_temporal": None,
    "histograma_bpm": None,
    "temp_por_cidade": None,
    "boxplot_bpm": None
}


def atualizar_graficos():
    while True:
        print("Atualizando gráficos e análises no cache...")
        graficos_cache["bpm"] = func.criar_histograma_bpm()
        graficos_cache["temp"] = func.criar_grafico_temperatura()
        graficos_cache["grafico_combinado"] = func.criar_grafico_combinado()
        graficos_cache["grafico_correlacao"] = func.criar_matriz_correlacao()
        graficos_cache["correlacao"] = func.calcular_correlacao_person()
        graficos_cache["barras_spline_bpm"] = func.grafico_barras_spline_bpm()
        graficos_cache["boxplot_bpm"] = func.grafico_boxplot_bpm()
        graficos_cache["heatmap_bpm"] = func.grafico_heatmap_bpm()
        graficos_cache["comparativo_bpm_temp"] = func.grafico_comparativo_bpm_temp()
        graficos_cache["scatter_temp_bpm"] = func.grafico_scatter_temp_vs_bpm()
        graficos_cache["serie_temporal"] = func.grafico_serie_temporal_bpm_temp()
        graficos_cache["histograma_bpm"] = func.grafico_histograma_bpm()
        graficos_cache["temp_por_cidade"] = func.grafico_temp_media_por_cidade()
        func.modelo()
        time.sleep(30)

@app.route("/")
def homepage():
    func.combinar_bpm_temp()
    return render_template("home.html")


@app.route("/estatisticas")
def estatisticas():
    return render_template("estatisticas.html", estatisticas=func.estatisticas_basicas(), title="Lista de Estatisticas")

@app.route("/verificar_anomalias")
def verificar_anomalias():
    lista_anomalias = func.obter_anomalias(func.melhor_modelo, func.melhor_modelo_nome, func.scaler)
    return render_template("home.html", anomalias=lista_anomalias)




@app.route("/salvar", methods=["GET", "POST"])
def guardar_info():
    if request.method == "POST":
        file_type = request.form.get('file_type')
        with open('combinado.json', 'r', encoding='utf-8') as f:
            data = json.load(f)

        df = pd.DataFrame(data)

        if file_type == 'csv':
            output = io.StringIO()
            df.to_csv(output, index=False)
            response = make_response(output.getvalue())
            response.headers["Content-Disposition"] = "attachment; filename=dados.csv"
            response.headers["Content-type"] = "text/csv"
            return response

        elif file_type == 'json':
            json_output = json.dumps(data, indent=2, ensure_ascii=False)
            response = make_response(json_output)
            response.headers["Content-Disposition"] = "attachment; filename=dados.json"
            response.headers["Content-type"] = "application/json"
            return response

    return render_template("save.html")



@app.route("/analise_dados")
def dados():
    dia = request.args.get("dia")
    grafico_bpm_dia = None
    if dia:
        grafico_bpm_dia = func.criar_grafico_bpm_por_dia(dia)

    return render_template("analise.html",
        grafico_bpm=graficos_cache["bpm"],
        grafico_temp=graficos_cache["temp"],
        grafico_combinado=graficos_cache["grafico_combinado"],
        grafico_correlacao=graficos_cache["grafico_correlacao"],
        correlacao=graficos_cache["correlacao"],
        grafico_barras_spline_bpm=graficos_cache["barras_spline_bpm"],
        grafico_boxplot_bpm=graficos_cache["boxplot_bpm"],
        grafico_heatmap_bpm=graficos_cache["heatmap_bpm"],
        grafico_comparativo_bpm_temp=graficos_cache["comparativo_bpm_temp"],
        grafico_scatter_temp_bpm=graficos_cache["scatter_temp_bpm"],
        grafico_serie_temporal=graficos_cache["serie_temporal"],
        grafico_histograma_bpm=graficos_cache["histograma_bpm"],
        grafico_temp_por_cidade=graficos_cache["temp_por_cidade"],
        grafico_bpm_dia=grafico_bpm_dia 
    )




@app.route("/add_dados", methods=["GET", "POST"])
def add_dados():
    mensagem = None
    if request.method == "POST":
        ficheiro = request.files.get("ficheiro")

        if ficheiro and ficheiro.filename.endswith(".xml"):
            caminho = os.path.join("static", "data", "data_bpm.xml")
            ficheiro.save(caminho)

            if func.extrair_dados_xml():
                func.enviar_bpm_para_firebase()

            mensagem = "Ficheiro carregado com sucesso!"
        else:
            mensagem = "Por favor, selecione um ficheiro XML válido."

    return render_template("add_dados.html", mensagem=mensagem)


@app.route("/adicionar", methods=["GET", "POST"])
def adicionar():
    mensagem = None
    mensagem_bpm = None
    aba_ativa = "simples"

    if request.method == "POST":
        form_type = request.form.get("form_type")

        if form_type == "temperatura":
            aba_ativa = "simples"
            try:
                dia = int(request.form["dia"])
                mes = int(request.form["mes"])
                ano = int(request.form["ano"])
                cidade = request.form["cidade"]
                temp_min = float(request.form["temp_min"])
                temp_max = float(request.form["temp_max"])

                sucesso = func.gerar_variacao_temperatura(dia, mes, ano, cidade, temp_min, temp_max)

                if sucesso:
                    mensagem = "Dados de temperatura adicionados com sucesso!"
                else:
                    mensagem = "Erro ao gerar os dados de temperatura."
            except Exception as e:
                print("Erro ao processar temperatura:", e)
                mensagem = "Erro ao processar os dados de temperatura."

        elif form_type == "bpm":
            aba_ativa = "avancados"
            try:
                data_escolhida = request.form["data"]

                modo = "anomalia" if "incluir_anomalias" in request.form else "normal"

                sucesso = func.aumentar_e_enviar_bpm_sintetico(data_escolhida, modo)

                if sucesso:
                    mensagem_bpm = f"BPMs {'com anomalias' if modo == 'anomalia' else 'normais'} enviados com sucesso para {data_escolhida}!"
                else:
                    mensagem_bpm = f"Não foi possível enviar os BPMs para {data_escolhida}."
            except Exception as e:
                print("Erro ao processar BPM:", e)
                mensagem_bpm = "Erro ao processar os dados BPM."

    return render_template("adicionar.html", mensagem=mensagem, mensagem_bpm=mensagem_bpm, aba_ativa=aba_ativa)


if __name__ == "__main__":
    func.combinar_bpm_temp()
    threading.Thread(target=atualizar_graficos, daemon=True).start()
    app.run(host="localhost", port=4002, debug=True)

