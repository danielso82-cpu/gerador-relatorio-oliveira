#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import io
import base64
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, request, send_file, render_template_string
from weasyprint import HTML

# Pasta base (onde está o app.py)
BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)


# -------------------------------------------------------------------
# Funções de apoio
# -------------------------------------------------------------------

def img_to_data_uri(filename: str) -> str:
    """
    Lê uma imagem PNG do disco e devolve um data URI em base64,
    para embutir direto no HTML (não depende de caminho de arquivo).
    """
    path = BASE_DIR / filename
    with open(path, "rb") as img_f:
        b64 = base64.b64encode(img_f.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def classificar_janela(dt):
    """Classifica a janela horária baseada no horário de saída (DTHRSAIDA)."""
    if pd.isna(dt):
        return np.nan
    h = dt.time()
    if h < pd.to_datetime('09:00').time():
        return 'J1'
    if h < pd.to_datetime('10:30').time():
        return 'J2'
    if h < pd.to_datetime('12:00').time():
        return 'J3'
    if h < pd.to_datetime('14:30').time():
        return 'J4'
    return 'J5'


def processar_dados(arquivo_excel, data_referencia):
    """
    Processa a planilha e retorna todos os dados necessários para o relatório.
    `arquivo_excel` pode ser um caminho ou um arquivo em memória (FileStorage).
    """
    df = pd.read_excel(arquivo_excel)

    # Converter colunas de data/hora (se existirem)
    for col in ['DTHRSAIDA', 'EMISSAO', 'HORAGRAV', 'DTHRRET']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    if 'DTHRSAIDA' not in df.columns:
        raise ValueError("A planilha não possui a coluna 'DTHRSAIDA'.")

    data_ref = pd.to_datetime(data_referencia).date()
    base = df[df['DTHRSAIDA'].dt.date == data_ref].copy()

    # Totais gerais
    TOTAL = len(base)
    if 'SITUACAO' in base.columns:
        entregue = (base['SITUACAO'].str.contains('Entregue', case=False, na=False)).sum()
        devolvido = (base['SITUACAO'].str.contains('Devolvido', case=False, na=False)).sum()
        em_entrega = (base['SITUACAO'].str.contains('Em Entrega', case=False, na=False)).sum()
    else:
        entregue = devolvido = em_entrega = 0

    TD_pct = round(100 * devolvido / (entregue + devolvido), 2) if (entregue + devolvido) > 0 else 0.0

    # Classificar janelas
    base['JANELA'] = base['DTHRSAIDA'].apply(classificar_janela)
    janela_counts = base['JANELA'].value_counts().to_dict()

    J1 = janela_counts.get('J1', 0)
    J2 = janela_counts.get('J2', 0)
    J3 = janela_counts.get('J3', 0)
    J4 = janela_counts.get('J4', 0)
    J5 = janela_counts.get('J5', 0)

    meta = {'J1': 30, 'J2': 20, 'J3': 10, 'J4': 30, 'J5': 10}

    J1_pct = round(100 * J1 / meta['J1'], 1) if meta['J1'] else 0
    J2_pct = round(100 * J2 / meta['J2'], 1) if meta['J2'] else 0
    J3_pct = round(100 * J3 / meta['J3'], 1) if meta['J3'] else 0
    J4_pct = round(100 * J4 / meta['J4'], 1) if meta['J4'] else 0
    J5_pct = round(100 * J5 / meta['J5'], 1) if meta['J5'] else 0
    TOTAL_pct = round(100 * TOTAL / 100, 1)  # meta total = 100

    GAP_J1 = meta['J1'] - J1
    GAP_J4 = meta['J4'] - J4

    # Conferência de colunas
    if 'MOTORISTA' not in base.columns or 'BAIRRO' not in base.columns:
        raise ValueError("A planilha precisa ter as colunas 'MOTORISTA' e 'BAIRRO'.")

    # Distribuição 1ª janela
    base_J1 = base[base['JANELA'] == 'J1'].copy()
    base_J1['HORA_SAIDA'] = base_J1['DTHRSAIDA'].dt.strftime('%H:%M')
    J1_dist = base_J1.groupby(['HORA_SAIDA', 'MOTORISTA', 'BAIRRO']).size().reset_index(name='QTD')
    J1_dist = J1_dist.sort_values(['HORA_SAIDA', 'MOTORISTA', 'BAIRRO'])

    # Distribuição 4ª janela
    base_J4 = base[base['JANELA'] == 'J4'].copy()
    base_J4['HORA_SAIDA'] = base_J4['DTHRSAIDA'].dt.strftime('%H:%M')
    J4_dist = base_J4.groupby(['HORA_SAIDA', 'MOTORISTA', 'BAIRRO']).size().reset_index(name='QTD')
    J4_dist = J4_dist.sort_values(['HORA_SAIDA', 'MOTORISTA', 'BAIRRO'])

    # Ranking motoristas
    rank = base.groupby(['MOTORISTA', 'BAIRRO']).size().reset_index(name='QTD')
    rank_tot = rank.groupby('MOTORISTA')['QTD'].sum().reset_index(name='TOTAL_MOTORISTA')
    rank = rank.merge(rank_tot, on='MOTORISTA')
    rank = rank.sort_values(
        ['TOTAL_MOTORISTA', 'QTD', 'MOTORISTA', 'BAIRRO'],
        ascending=[False, False, True, True]
    )
    rank_top20 = rank.head(20)

    return {
        'data_ref': data_ref.strftime('%d/%m/%Y'),
        'TOTAL': int(TOTAL),
        'ENTREGUE': int(entregue),
        'DEVOLVIDO': int(devolvido),
        'EM_ENTREGA': int(em_entrega),
        'TD_PCT': TD_pct,
        'J1': int(J1), 'J2': int(J2), 'J3': int(J3), 'J4': int(J4), 'J5': int(J5),
        'J1_PCT': J1_pct, 'J2_PCT': J2_pct, 'J3_PCT': J3_pct, 'J4_PCT': J4_pct, 'J5_PCT': J5_pct,
        'TOTAL_PCT': TOTAL_pct,
        'GAP_J1': int(GAP_J1),
        'GAP_J4': int(GAP_J4),
        'J1_dist': J1_dist,
        'J4_dist': J4_dist,
        'rank': rank_top20
    }


def gerar_html_tabela(df, colunas):
    """Gera HTML de uma tabela a partir de um DataFrame."""
    html = '<table>\n<thead>\n<tr>\n'
    for col in colunas:
        html += f'<th>{col}</th>\n'
    html += '</tr>\n</thead>\n<tbody>\n'

    for _, row in df.iterrows():
        html += '<tr>'
        for col in df.columns:
            html += f'<td>{row[col]}</td>'
        html += '</tr>\n'

    html += '</tbody>\n</table>'
    return html


def montar_html_relatorio(dados):
    """Monta o HTML completo do relatório, com imagens embutidas em base64."""

    J1_tabela = gerar_html_tabela(
        dados['J1_dist'],
        ['Horário Saída', 'Motorista', 'Bairro', 'Qtde']
    )
    J4_tabela = gerar_html_tabela(
        dados['J4_dist'],
        ['Horário Saída', 'Motorista', 'Bairro', 'Qtde']
    )
    rank_tabela = gerar_html_tabela(
        dados['rank'],
        ['Motorista', 'Bairro', 'Qtde', 'Total Motorista']
    )

    # Imagens embutidas
    logo_src = img_to_data_uri("Logo Oliveira Sem Fundo.png")
    oliver_src = img_to_data_uri("Oliver_RomaneioSF.png")

    html_content = f"""
<html>
<head>
  <meta charset="UTF-8" />
  <title>Relatório {dados['data_ref']} - Oliveira Materiais de Construção</title>
  <style>
    @page {{
        size: A4;
        margin: 20mm;
    }}
    body {{
        margin: 0 auto;
        padding: 20px;
        font-family: Arial, Helvetica, sans-serif;
        font-size: 11pt;
        line-height: 1.4;
    }}
    .header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 15px;
        padding-bottom: 10px;
        border-bottom: 3px solid #008000;
        min-height: 80px;
    }}
    .header .logo-container {{
        flex: 0 0 auto;
        max-width: 35%;
    }}
    .header .logo-container img {{
        max-width: 100%;
        height: auto;
        max-height: 55px;
    }}
    .header .mascote-container {{
        flex: 0 0 auto;
        max-width: 40%;
        text-align: right;
    }}
    .header .mascote-container img {{
        max-width: 100%;
        height: auto;
        max-height: 95px;
    }}
    .title-block {{
        text-align: center;
        margin-top: 15px;
        margin-bottom: 20px;
        font-weight: bold;
        font-size: 16pt;
        color: #008000;
    }}
    .subtitle {{
        text-align: center;
        font-size: 12pt;
        margin-bottom: 25px;
        color: #333;
    }}
    .section-title {{
        font-size: 13pt;
        font-weight: bold;
        color: #008000;
        margin-top: 20px;
        margin-bottom: 10px;
        border-bottom: 2px solid #008000;
        padding-bottom: 5px;
    }}
    .content {{
        margin-top: 10px;
        font-size: 11pt;
        color: #333;
        line-height: 1.6;
    }}
    .footer-bar {{
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        height: 30px;
        background-color: #008000;
        color: #ffffff;
        font-size: 10pt;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
    }}
    .page-break {{
        page-break-after: always;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin-top: 10px;
        margin-bottom: 15px;
        font-size: 10pt;
    }}
    table th {{
        background-color: #008000;
        color: white;
        padding: 8px;
        text-align: left;
        font-weight: bold;
    }}
    table td {{
        padding: 6px;
        border-bottom: 1px solid #ddd;
    }}
    table tr:nth-child(even) {{
        background-color: #f9f9f9;
    }}
  </style>
</head>
<body>
  <!-- Página 1 -->
  <div class="header">
    <div class="logo-container">
      <img src="{logo_src}" alt="Logo Oliveira" />
    </div>
    <div class="mascote-container">
      <img src="{oliver_src}" alt="Mascote Romaneio" />
    </div>
  </div>

  <div class="title-block">
    RELATÓRIO DIÁRIO DE ENTREGAS
  </div>

  <div class="subtitle">
    Data de referência: {dados['data_ref']} – Logística 2025
  </div>

  <div class="section-title">1. RESUMO EXECUTIVO</div>
  <div class="content">
    <p><strong>Total de entregas do dia:</strong> {dados['TOTAL']}</p>
    <p><strong>Entregue:</strong> {dados['ENTREGUE']} &nbsp;|&nbsp;
       <strong>Devolvido:</strong> {dados['DEVOLVIDO']} &nbsp;|&nbsp;
       <strong>Em Entrega:</strong> {dados['EM_ENTREGA']}</p>
    <p><strong>TD% do dia:</strong> {dados['TD_PCT']}%</p>
  </div>

  <div class="section-title">2. PAINEL DE JANELAS – META x REAL</div>
  <div class="content">
    <table>
      <thead>
        <tr>
          <th>Janela</th>
          <th>Meta</th>
          <th>Real</th>
          <th>% da Meta</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>1ª (saídas antes de 09:00)</td>
          <td>30</td>
          <td>{dados['J1']}</td>
          <td>{dados['J1_PCT']}%</td>
        </tr>
        <tr>
          <td>2ª (09:00–10:30)</td>
          <td>20</td>
          <td>{dados['J2']}</td>
          <td>{dados['J2_PCT']}%</td>
        </tr>
        <tr>
          <td>3ª (10:30–12:00)</td>
          <td>10</td>
          <td>{dados['J3']}</td>
          <td>{dados['J3_PCT']}%</td>
        </tr>
        <tr>
          <td>4ª (12:00–14:30)</td>
          <td>30</td>
          <td>{dados['J4']}</td>
          <td>{dados['J4_PCT']}%</td>
        </tr>
        <tr>
          <td>5ª (≥ 14:30)</td>
          <td>10</td>
          <td>{dados['J5']}</td>
          <td>{dados['J5_PCT']}%</td>
        </tr>
        <tr style="font-weight: bold; background-color: #e8f5e9;">
          <td>TOTAL</td>
          <td>100</td>
          <td>{dados['TOTAL']}</td>
          <td>{dados['TOTAL_PCT']}%</td>
        </tr>
      </tbody>
    </table>
  </div>

  <div class="footer-bar">
    Oliveira Materiais de Construção - Logística 2025
  </div>

  <div class="page-break"></div>

  <!-- Página 2 - 1ª Janela -->
  <div class="header">
    <div class="logo-container">
      <img src="{logo_src}" alt="Logo Oliveira" />
    </div>
    <div class="mascote-container">
      <img src="{oliver_src}" alt="Mascote Romaneio" />
    </div>
  </div>

  <div class="section-title">3. DISTRIBUIÇÃO OPERACIONAL – 1ª JANELA (saídas até 09:00)</div>
  <div class="content">
    <p><strong>Entregas na 1ª janela:</strong> {dados['J1']} &nbsp;|&nbsp;
       <strong>Meta:</strong> 30 &nbsp;|&nbsp;
       <strong>Gap vs meta:</strong> {'+' if dados['GAP_J1'] < 0 else ''}{-dados['GAP_J1']} notas.</p>
    <p>Tabela detalhada por horário de saída, motorista e bairro:</p>
    {J1_tabela}
  </div>

  <div class="footer-bar">
    Oliveira Materiais de Construção - Logística 2025
  </div>

  <div class="page-break"></div>

  <!-- Página 3 - 4ª Janela + Ranking -->
  <div class="header">
    <div class="logo-container">
      <img src="{logo_src}" alt="Logo Oliveira" />
    </div>
    <div class="mascote-container">
      <img src="{oliver_src}" alt="Mascote Romaneio" />
    </div>
  </div>

  <div class="section-title">4. DISTRIBUIÇÃO OPERACIONAL – 4ª JANELA (12:00–14:30)</div>
  <div class="content">
    <p><strong>Entregas na 4ª janela:</strong> {dados['J4']} &nbsp;|&nbsp;
       <strong>Meta:</strong> 30 &nbsp;|&nbsp;
       <strong>Gap vs meta:</strong> {'+' if dados['GAP_J4'] < 0 else ''}{-dados['GAP_J4']} notas.</p>
    <p>Tabela detalhada por horário de saída, motorista e bairro:</p>
    {J4_tabela}
  </div>

  <div class="section-title">5. RANKING DE MOTORISTAS COM BAIRROS – TOP 20</div>
  <div class="content">
    <p>Ordenado pelo <strong>total de entregas do motorista</strong>, mantendo os bairros do mesmo motorista em sequência.</p>
    {rank_tabela}
  </div>

  <div class="footer-bar">
    Oliveira Materiais de Construção - Logística 2025
  </div>

</body>
</html>
"""
    return html_content


# -------------------------------------------------------------------
# Rotas Web
# -------------------------------------------------------------------

FORM_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Gerador de Relatório - Logística Oliveira</title>
<style>
body {
  font-family: Arial, sans-serif;
  background: #f4f6f5;
  padding: 30px;
}
.container {
  max-width: 500px;
  margin: auto;
  background: white;
  padding: 20px;
  border-radius: 8px;
  border-top: 6px solid #007a33;
  box-shadow: 0 2px 6px rgba(0,0,0,0.1);
}
h2 {
  color: #007a33;
  text-align: center;
}
label {
  display: block;
  margin-top: 10px;
}
input, button {
  width: 100%;
  padding: 10px;
  margin-top: 5px;
  box-sizing: border-box;
}
button {
  background: #007a33;
  color: white;
  border: none;
  cursor: pointer;
  margin-top: 15px;
}
button:hover {
  background: #005c26;
}
.info {
  font-size: 12px;
  color: #555;
  margin-top: 10px;
}
</style>
</head>
<body>
  <div class="container">
    <h2>📊 Gerador de Relatórios – Logística Oliveira</h2>
    <form method="post" action="/gerar" enctype="multipart/form-data">
      <label>Planilha (.xlsx)</label>
      <input type="file" name="arquivo" accept=".xlsx" required>
      <label>Data de referência</label>
      <input type="date" name="data_ref" required>
      <button type="submit">Gerar PDF</button>
    </form>
    <div class="info">
      O PDF será gerado usando o padrão de janelas (J1–J5) e baixado automaticamente.
    </div>
  </div>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    return render_template_string(FORM_HTML)


@app.route("/gerar", methods=["POST"])
def gerar():
    try:
        arquivo = request.files.get("arquivo")
        data_ref = request.form.get("data_ref")

        if not arquivo or not data_ref:
            return "Planilha e data são obrigatórias.", 400

        dados = processar_dados(arquivo, data_ref)
        html_rel = montar_html_relatorio(dados)

        pdf_io = io.BytesIO()
        HTML(string=html_rel, base_url=str(BASE_DIR)).write_pdf(pdf_io)
        pdf_io.seek(0)

        filename = f"relatorio_oliveira_{data_ref}.pdf"

        return send_file(
            pdf_io,
            as_attachment=True,
            download_name=filename,
            mimetype="application/pdf"
        )

    except Exception as e:
        # Pra debug simples: mostra o erro na tela
        return f"Erro ao gerar relatório: {e}", 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
