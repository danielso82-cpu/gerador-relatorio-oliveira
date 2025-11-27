#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import io
import base64
from datetime import datetime

import numpy as np
import pandas as pd
from flask import Flask, request, send_file, render_template_string
from weasyprint import HTML

# -------------------------------------------------------------------
# Configuração básica do Flask
# -------------------------------------------------------------------
app = Flask(__name__)

# -------------------------------------------------------------------
# Funções auxiliares
# -------------------------------------------------------------------
def classificar_janela(dt):
    """Classifica a janela horária baseada no horário de saída."""
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


def gerar_html_tabela(df, colunas):
    """Gera HTML de uma tabela a partir de um DataFrame."""
    if df is None or df.empty:
        return "<p>Sem dados para exibir.</p>"

    html = '<table>\n<thead>\n<tr>\n'
    # Cabeçalho
    for col in colunas:
        html += f'<th>{col}</th>\n'
    html += '</tr>\n</thead>\n<tbody>\n'

    # Linhas
    for _, row in df.iterrows():
        html += '<tr>'
        for col in df.columns:
            html += f'<td>{row[col]}</td>'
        html += '</tr>\n'

    html += '</tbody>\n</table>'
    return html


def img_to_data_uri(filename):
    """Converte uma imagem local em data URI base64 para embutir no HTML."""
    if not os.path.exists(filename):
        return ""
    with open(filename, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode("ascii")
    ext = os.path.splitext(filename)[1].lower().replace(".", "")
    mime = f"image/{'png' if ext == 'png' else ext}"
    return f"data:{mime};base64,{encoded}"


def preparar_dataframe(df):
    """Garante que as colunas de data estejam em datetime."""
    for col in ['DTHRSAIDA', 'EMISSAO', 'HORAGRAV', 'DTHRRET']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df


# -------------------------------------------------------------------
# Processamento DIÁRIO
# -------------------------------------------------------------------
def processar_dados_diarios(df, data_referencia):
    """
    Processa a planilha (df) e retorna todos os dados necessários
    para o relatório DIÁRIO da data_referencia (YYYY-MM-DD).
    """
    data_ref = pd.to_datetime(data_referencia).date()

    if 'DTHRSAIDA' not in df.columns:
        raise ValueError("A planilha não possui a coluna 'DTHRSAIDA'.")

    mask = df['DTHRSAIDA'].dt.date == data_ref
    base = df[mask].copy()

    if base.empty:
        raise ValueError("Não há entregas para a data selecionada.")

    # Totais gerais
    if 'SITUACAO' in base.columns:
        entregue = (base['SITUACAO'].str.contains('Entregue', case=False, na=False)).sum()
        devolvido = (base['SITUACAO'].str.contains('Devolvido', case=False, na=False)).sum()
        em_entrega = (base['SITUACAO'].str.contains('Em Entrega', case=False, na=False)).sum()
    else:
        entregue = devolvido = em_entrega = 0

    TOTAL = len(base)
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
    TOTAL_pct = round(100 * TOTAL / 100, 1)

    GAP_J1 = meta['J1'] - J1
    GAP_J4 = meta['J4'] - J4

    # Distribuição 1ª janela com horário
    base_J1 = base[base['JANELA'] == 'J1'].copy()
    if not base_J1.empty:
        base_J1['HORA_SAIDA'] = base_J1['DTHRSAIDA'].dt.strftime('%H:%M')
        J1_dist = (
            base_J1.groupby(['HORA_SAIDA', 'MOTORISTA', 'BAIRRO'])
            .size()
            .reset_index(name='QTD')
            .sort_values(['HORA_SAIDA', 'MOTORISTA', 'BAIRRO'])
        )
    else:
        J1_dist = pd.DataFrame(columns=['HORA_SAIDA', 'MOTORISTA', 'BAIRRO', 'QTD'])

    # Distribuição 4ª janela com horário
    base_J4 = base[base['JANELA'] == 'J4'].copy()
    if not base_J4.empty:
        base_J4['HORA_SAIDA'] = base_J4['DTHRSAIDA'].dt.strftime('%H:%M')
        J4_dist = (
            base_J4.groupby(['HORA_SAIDA', 'MOTORISTA', 'BAIRRO'])
            .size()
            .reset_index(name='QTD')
            .sort_values(['HORA_SAIDA', 'MOTORISTA', 'BAIRRO'])
        )
    else:
        J4_dist = pd.DataFrame(columns=['HORA_SAIDA', 'MOTORISTA', 'BAIRRO', 'QTD'])

    # Ranking motoristas
    if 'MOTORISTA' in base.columns and 'BAIRRO' in base.columns:
        rank = base.groupby(['MOTORISTA', 'BAIRRO']).size().reset_index(name='QTD')
        rank_tot = rank.groupby('MOTORISTA')['QTD'].sum().reset_index(name='TOTAL_MOTORISTA')
        rank = rank.merge(rank_tot, on='MOTORISTA')
        rank = rank.sort_values(
            ['TOTAL_MOTORISTA', 'QTD', 'MOTORISTA', 'BAIRRO'],
            ascending=[False, False, True, True]
        )
        rank_top20 = rank.head(20)
    else:
        rank_top20 = pd.DataFrame(columns=['MOTORISTA', 'BAIRRO', 'QTD', 'TOTAL_MOTORISTA'])

    return {
        'data_ref': data_ref.strftime('%d/%m/%Y'),
        'TOTAL': int(TOTAL),
        'ENTREGUE': int(entregue),
        'DEVOLVIDO': int(devolvido),
        'EM_ENTREGA': int(em_entrega),
        'TD_PCT': TD_pct,
        'J1': int(J1),
        'J2': int(J2),
        'J3': int(J3),
        'J4': int(J4),
        'J5': int(J5),
        'J1_PCT': J1_pct,
        'J2_PCT': J2_pct,
        'J3_PCT': J3_pct,
        'J4_PCT': J4_pct,
        'J5_PCT': J5_pct,
        'TOTAL_PCT': TOTAL_pct,
        'GAP_J1': int(GAP_J1),
        'GAP_J4': int(GAP_J4),
        'J1_dist': J1_dist,
        'J4_dist': J4_dist,
        'rank': rank_top20,
    }


# -------------------------------------------------------------------
# Processamento de PERÍODO (com sessões 4,5,6)
# -------------------------------------------------------------------
def processar_dados_periodo(df, data_ini, data_fim):
    """
    Processa a planilha para um PERÍODO (data_ini até data_fim, inclusive)
    SOMANDO os mesmos indicadores do relatório DIÁRIO e gerando:

    - Totais do período (TOTAL, ENTREGUE, DEVOLVIDO, EM_ENTREGA, TD%)
    - Metas de janelas proporcionais ao nº de dias
    - Tabela dia a dia (TOTAL, J1..J5, TD% dia)
    - PRODUTIVIDADE POR MOTORISTA
    - ANÁLISE POR BAIRRO
    - ANÁLISE POR TIPO DE VEÍCULO (TPRODADO)
    """

    di = pd.to_datetime(data_ini).date()
    dfim = pd.to_datetime(data_fim).date()

    if 'DTHRSAIDA' not in df.columns:
        raise ValueError("A planilha não possui a coluna 'DTHRSAIDA'.")

    datas_candidatas = sorted(df['DTHRSAIDA'].dropna().dt.date.unique())
    datas_periodo = [d for d in datas_candidatas if di <= d <= dfim]

    if not datas_periodo:
        raise ValueError("Não há registros no período informado.")

    linhas_dia = []
    TOTAL = ENTREGUE = DEVOLVIDO = EM_ENTREGA = 0
    J1 = J2 = J3 = J4 = J5 = 0

    # Consolidação dia a dia reutilizando a lógica do diário
    for dia in datas_periodo:
        dados_dia = processar_dados_diarios(df, dia.strftime('%Y-%m-%d'))

        if dados_dia['TOTAL'] == 0:
            continue

        TOTAL += dados_dia['TOTAL']
        ENTREGUE += dados_dia['ENTREGUE']
        DEVOLVIDO += dados_dia['DEVOLVIDO']
        EM_ENTREGA += dados_dia['EM_ENTREGA']

        J1 += dados_dia['J1']
        J2 += dados_dia['J2']
        J3 += dados_dia['J3']
        J4 += dados_dia['J4']
        J5 += dados_dia['J5']

        linhas_dia.append({
            'DATA': dia.strftime('%d/%m/%Y'),
            'TOTAL': dados_dia['TOTAL'],
            'J1': dados_dia['J1'],
            'J2': dados_dia['J2'],
            'J3': dados_dia['J3'],
            'J4': dados_dia['J4'],
            'J5': dados_dia['J5'],
            'TD_PCT_DIA': dados_dia['TD_PCT']
        })

    if not linhas_dia:
        raise ValueError("Não há registros válidos no período (após consolidar por dia).")

    janela_dia = pd.DataFrame(linhas_dia)
    n_dias = len(janela_dia)

    TD_PCT = round(100 * DEVOLVIDO / (ENTREGUE + DEVOLVIDO), 2) if (ENTREGUE + DEVOLVIDO) > 0 else 0.0

    # Metas proporcionais ao período
    meta_base = {'J1': 30, 'J2': 20, 'J3': 10, 'J4': 30, 'J5': 10, 'TOTAL': 100}
    META_J1 = meta_base['J1'] * n_dias
    META_J2 = meta_base['J2'] * n_dias
    META_J3 = meta_base['J3'] * n_dias
    META_J4 = meta_base['J4'] * n_dias
    META_J5 = meta_base['J5'] * n_dias
    META_TOTAL = meta_base['TOTAL'] * n_dias

    J1_PCT_META = round(100 * J1 / META_J1, 1) if META_J1 else 0
    J2_PCT_META = round(100 * J2 / META_J2, 1) if META_J2 else 0
    J3_PCT_META = round(100 * J3 / META_J3, 1) if META_J3 else 0
    J4_PCT_META = round(100 * J4 / META_J4, 1) if META_J4 else 0
    J5_PCT_META = round(100 * J5 / META_J5, 1) if META_J5 else 0
    TOTAL_PCT_META = round(100 * TOTAL / META_TOTAL, 1) if META_TOTAL else 0

    # Base do período (para motorista, bairro, veículo)
    base_periodo = df[df['DTHRSAIDA'].notna() &
                      (df['DTHRSAIDA'].dt.date.isin(datas_periodo))].copy()

    if 'SITUACAO' not in base_periodo.columns:
        raise ValueError("A planilha precisa da coluna 'SITUACAO' para os relatórios de período.")

    # ------------------------------------------------------------------
    # 4. PRODUTIVIDADE POR MOTORISTA
    # ------------------------------------------------------------------
    if 'MOTORISTA' not in base_periodo.columns:
        raise ValueError("A planilha não possui a coluna 'MOTORISTA'.")

    base_motoristas = base_periodo[
        base_periodo['MOTORISTA'].str.contains('MOTORISTA', case=False, na=False)
    ].copy()

    ent_motor = base_motoristas[
        base_motoristas['SITUACAO'].str.contains('Entregue', case=False, na=False)
    ].groupby('MOTORISTA').size().rename('ENTREGAS')

    dev_motor = base_motoristas[
        base_motoristas['SITUACAO'].str.contains('Devolvido', case=False, na=False)
    ].groupby('MOTORISTA').size().rename('DEVOLUCOES')

    prod = pd.concat([ent_motor, dev_motor], axis=1).fillna(0)
    prod['ENTREGAS'] = prod['ENTREGAS'].astype(int)
    prod['DEVOLUCOES'] = prod['DEVOLUCOES'].astype(int)
    prod['TOTAL'] = prod['ENTREGAS'] + prod['DEVOLUCOES']

    prod['TD_PCT'] = prod.apply(
        lambda r: round(100 * r['DEVOLUCOES'] / r['TOTAL'], 2) if r['TOTAL'] > 0 else 0.0,
        axis=1
    )

    dias_trabalhados = base_motoristas.groupby('MOTORISTA')['DTHRSAIDA'] \
        .apply(lambda s: s.dt.date.nunique()).rename('DIAS_TRAB')

    prod = prod.join(dias_trabalhados, how='left').fillna({'DIAS_TRAB': 0})
    prod['DIAS_TRAB'] = prod['DIAS_TRAB'].astype(int)

    prod['MEDIA_DIA'] = prod.apply(
        lambda r: round(r['TOTAL'] / r['DIAS_TRAB'], 1) if r['DIAS_TRAB'] > 0 else 0.0,
        axis=1
    )

    prod = prod.sort_values('TOTAL', ascending=False)

    prod_tabela = prod.reset_index()[['MOTORISTA', 'ENTREGAS', 'DEVOLUCOES', 'TOTAL', 'TD_PCT', 'MEDIA_DIA']]

    total_row_motor = {
        'MOTORISTA': 'TOTAL GERAL',
        'ENTREGAS': int(prod_tabela['ENTREGAS'].sum()),
        'DEVOLUCOES': int(prod_tabela['DEVOLUCOES'].sum()),
        'TOTAL': int(prod_tabela['TOTAL'].sum()),
        'TD_PCT': '',
        'MEDIA_DIA': ''
    }
    prod_tabela = pd.concat([prod_tabela, pd.DataFrame([total_row_motor])], ignore_index=True)

    # ------------------------------------------------------------------
    # 5. ANÁLISE POR BAIRRO
    # ------------------------------------------------------------------
    if 'BAIRRO' not in base_periodo.columns:
        raise ValueError("A planilha não possui a coluna 'BAIRRO'.")

    ent_bairro = base_periodo[
        base_periodo['SITUACAO'].str.contains('Entregue', case=False, na=False)
    ].groupby('BAIRRO').size().rename('ENTREGAS')

    dev_bairro = base_periodo[
        base_periodo['SITUACAO'].str.contains('Devolvido', case=False, na=False)
    ].groupby('BAIRRO').size().rename('DEVOLUCOES')

    bairro = pd.concat([ent_bairro, dev_bairro], axis=1).fillna(0)
    bairro['ENTREGAS'] = bairro['ENTREGAS'].astype(int)
    bairro['DEVOLUCOES'] = bairro['DEVOLUCOES'].astype(int)
    bairro['TOTAL'] = bairro['ENTREGAS'] + bairro['DEVOLUCOES']

    bairro['TD_PCT'] = bairro.apply(
        lambda r: round(100 * r['DEVOLUCOES'] / r['TOTAL'], 2) if r['TOTAL'] > 0 else 0.0,
        axis=1
    )

    bairro = bairro.sort_values('TOTAL', ascending=False)
    bairro_tabela = bairro.reset_index()[['BAIRRO', 'ENTREGAS', 'DEVOLUCOES', 'TOTAL', 'TD_PCT']]

    total_row_bairro = {
        'BAIRRO': 'TOTAL GERAL',
        'ENTREGAS': int(bairro_tabela['ENTREGAS'].sum()),
        'DEVOLUCOES': int(bairro_tabela['DEVOLUCOES'].sum()),
        'TOTAL': int(bairro_tabela['TOTAL'].sum()),
        'TD_PCT': ''
    }
    bairro_tabela = pd.concat([bairro_tabela, pd.DataFrame([total_row_bairro])], ignore_index=True)

    # ------------------------------------------------------------------
    # 6. ANÁLISE POR TIPO DE VEÍCULO (TPRODADO)
    # ------------------------------------------------------------------
    if 'TPRODADO' in base_periodo.columns:
        mapa_veiculo = {
            2: "02 - Toco (Caçamba 5m³)",
            5: "05 - Utilitário (HR)",
            6: "06 - Caçamba 3m³",
            7: "07 - VUC (Carroceria 6m)",
            0: "00 - Munk",
            2.0: "02 - Toco (Caçamba 5m³)",
            5.0: "05 - Utilitário (HR)",
            6.0: "06 - Caçamba 3m³",
            7.0: "07 - VUC (Carroceria 6m)",
            0.0: "00 - Munk",
        }

        base_periodo['VEICULO_TIPO'] = base_periodo['TPRODADO'].map(mapa_veiculo)
        base_periodo['VEICULO_TIPO'] = base_periodo['VEICULO_TIPO'].fillna('Outro / Não mapeado')

        ent_veic = base_periodo[
            base_periodo['SITUACAO'].str.contains('Entregue', case=False, na=False)
        ].groupby('VEICULO_TIPO').size().rename('ENTREGAS')

        dev_veic = base_periodo[
            base_periodo['SITUACAO'].str.contains('Devolvido', case=False, na=False)
        ].groupby('VEICULO_TIPO').size().rename('DEVOLUCOES')

        veic = pd.concat([ent_veic, dev_veic], axis=1).fillna(0)
        veic['ENTREGAS'] = veic['ENTREGAS'].astype(int)
        veic['DEVOLUCOES'] = veic['DEVOLUCOES'].astype(int)
        veic['TOTAL'] = veic['ENTREGAS'] + veic['DEVOLUCOES']

        veic['TD_PCT'] = veic.apply(
            lambda r: round(100 * r['DEVOLUCOES'] / r['TOTAL'], 2) if r['TOTAL'] > 0 else 0.0,
            axis=1
        )

        veic = veic.sort_values('TOTAL', ascending=False)
        veic_tabela = veic.reset_index()[['VEICULO_TIPO', 'ENTREGAS', 'DEVOLUCOES', 'TOTAL', 'TD_PCT']]

        total_row_veic = {
            'VEICULO_TIPO': 'TOTAL GERAL',
            'ENTREGAS': int(veic_tabela['ENTREGAS'].sum()),
            'DEVOLUCOES': int(veic_tabela['DEVOLUCOES'].sum()),
            'TOTAL': int(veic_tabela['TOTAL'].sum()),
            'TD_PCT': ''
        }
        veic_tabela = pd.concat([veic_tabela, pd.DataFrame([total_row_veic])], ignore_index=True)
    else:
        veic_tabela = pd.DataFrame(columns=['VEICULO_TIPO', 'ENTREGAS', 'DEVOLUCOES', 'TOTAL', 'TD_PCT'])

    return {
        'data_ini': di.strftime('%d/%m/%Y'),
        'data_fim': dfim.strftime('%d/%m/%Y'),
        'N_DIAS': n_dias,
        'TOTAL': int(TOTAL),
        'ENTREGUE': int(ENTREGUE),
        'DEVOLVIDO': int(DEVOLVIDO),
        'EM_ENTREGA': int(EM_ENTREGA),
        'TD_PCT': TD_PCT,
        'J1': int(J1),
        'J2': int(J2),
        'J3': int(J3),
        'J4': int(J4),
        'J5': int(J5),
        'META_J1': META_J1,
        'META_J2': META_J2,
        'META_J3': META_J3,
        'META_J4': META_J4,
        'META_J5': META_J5,
        'META_TOTAL': META_TOTAL,
        'J1_PCT_META': J1_PCT_META,
        'J2_PCT_META': J2_PCT_META,
        'J3_PCT_META': J3_PCT_META,
        'J4_PCT_META': J4_PCT_META,
        'J5_PCT_META': J5_PCT_META,
        'TOTAL_PCT_META': TOTAL_PCT_META,
        'JANELAS_DIA': janela_dia,
        'PROD_MOTORISTA': prod_tabela,
        'BAIRRO_ANALISE': bairro_tabela,
        'VEICULO_ANALISE': veic_tabela,
    }


# -------------------------------------------------------------------
# Montagem do HTML – RELATÓRIO DIÁRIO
# -------------------------------------------------------------------
def montar_html_relatorio_diario(dados):
    """Monta o HTML do relatório DIÁRIO."""
    logo_src = img_to_data_uri("Logo Oliveira Sem Fundo.png")
    oliver_src = img_to_data_uri("Oliver_RomaneioSF.png")

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
    <p>
      O volume total {'superou' if dados['TOTAL'] >= 100 else 'ficou abaixo da'} meta de 100 entregas,
      atingindo {dados['TOTAL_PCT']}% da capacidade planejada.
    </p>
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
    <p><strong>Entregas na 1ª janela:</strong> {dados['J1']} &nbsp;|&nbsp; <strong>Meta:</strong> 30 &nbsp;|&nbsp;
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
    <p><strong>Entregas na 4ª janela:</strong> {dados['J4']} &nbsp;|&nbsp; <strong>Meta:</strong> 30 &nbsp;|&nbsp;
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
# Montagem do HTML – RELATÓRIO DE PERÍODO
# -------------------------------------------------------------------
def montar_html_relatorio_periodo(dados):
    """Monta o HTML do relatório de acompanhamento de PERÍODO."""

    logo_src = img_to_data_uri("Logo Oliveira Sem Fundo.png")
    oliver_src = img_to_data_uri("Oliver_RomaneioSF.png")

    tabela_periodo = gerar_html_tabela(
        dados['JANELAS_DIA'],
        ['Data', 'Total', 'J1', 'J2', 'J3', 'J4', 'J5', 'TD% Dia']
    )

    tabela_prod = gerar_html_tabela(
        dados['PROD_MOTORISTA'],
        ['Motorista', 'Entregas', 'Devoluções', 'Total', 'TD%', 'Média/Dia']
    )

    tabela_bairro = gerar_html_tabela(
        dados['BAIRRO_ANALISE'],
        ['Bairro', 'Entregas', 'Devoluções', 'Total', 'TD%']
    )

    tabela_veic = gerar_html_tabela(
        dados['VEICULO_ANALISE'],
        ['Tipo de Veículo', 'Entregas', 'Devoluções', 'Total', 'TD%']
    )

    html_content = f"""
<html>
<head>
  <meta charset="UTF-8" />
  <title>Relatório de Período - Oliveira Materiais de Construção</title>
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
  <div class="header">
    <div class="logo-container">
      <img src="{logo_src}" alt="Logo Oliveira" />
    </div>
    <div class="mascote-container">
      <img src="{oliver_src}" alt="Mascote Romaneio" />
    </div>
  </div>

  <div class="title-block">
    RELATÓRIO DE ACOMPANHAMENTO – PERÍODO
  </div>

  <div class="subtitle">
    Período: {dados['data_ini']} a {dados['data_fim']} – {dados['N_DIAS']} dia(s) com movimento
  </div>

  <div class="section-title">1. RESUMO EXECUTIVO DO PERÍODO</div>
  <div class="content">
    <p><strong>Total de entregas no período:</strong> {dados['TOTAL']}</p>
    <p><strong>Entregue:</strong> {dados['ENTREGUE']} &nbsp;|&nbsp;
       <strong>Devolvido:</strong> {dados['DEVOLVIDO']} &nbsp;|&nbsp;
       <strong>Em Entrega:</strong> {dados['EM_ENTREGA']}</p>
    <p><strong>TD% médio do período:</strong> {dados['TD_PCT']}%</p>
  </div>

  <div class="section-title">2. JANELAS – META x REAL NO PERÍODO</div>
  <div class="content">
    <table>
      <thead>
        <tr>
          <th>Janela</th>
          <th>Meta no período</th>
          <th>Real</th>
          <th>% da Meta</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>1ª (saídas antes de 09:00)</td>
          <td>{dados['META_J1']}</td>
          <td>{dados['J1']}</td>
          <td>{dados['J1_PCT_META']}%</td>
        </tr>
        <tr>
          <td>2ª (09:00–10:30)</td>
          <td>{dados['META_J2']}</td>
          <td>{dados['J2']}</td>
          <td>{dados['J2_PCT_META']}%</td>
        </tr>
        <tr>
          <td>3ª (10:30–12:00)</td>
          <td>{dados['META_J3']}</td>
          <td>{dados['J3']}</td>
          <td>{dados['J3_PCT_META']}%</td>
        </tr>
        <tr>
          <td>4ª (12:00–14:30)</td>
          <td>{dados['META_J4']}</td>
          <td>{dados['J4']}</td>
          <td>{dados['J4_PCT_META']}%</td>
        </tr>
        <tr>
          <td>5ª (≥ 14:30)</td>
          <td>{dados['META_J5']}</td>
          <td>{dados['J5']}</td>
          <td>{dados['J5_PCT_META']}%</td>
        </tr>
        <tr style="font-weight: bold; background-color: #e8f5e9;">
          <td>TOTAL</td>
          <td>{dados['META_TOTAL']}</td>
          <td>{dados['TOTAL']}</td>
          <td>{dados['TOTAL_PCT_META']}%</td>
        </tr>
      </tbody>
    </table>
  </div>

  <div class="section-title">3. DISTRIBUIÇÃO POR DIA E JANELA</div>
  <div class="content">
    <p>Tabela com entregas por dia, total e por janela (incluindo TD% diário):</p>
    {tabela_periodo}
  </div>

  <div class="section-title">4. PRODUTIVIDADE POR MOTORISTA (PERÍODO)</div>
  <div class="content">
    <p>Tabela consolidada de entregas e devoluções por motorista no período.</p>
    {tabela_prod}
  </div>

  <div class="section-title">5. ANÁLISE POR BAIRRO (PERÍODO)</div>
  <div class="content">
    <p>Distribuição de entregas e devoluções por bairro no período.</p>
    {tabela_bairro}
  </div>

  <div class="section-title">6. ANÁLISE POR TIPO DE VEÍCULO (PERÍODO)</div>
  <div class="content">
    <p>Distribuição de entregas e devoluções por tipo de veículo no período.</p>
    {tabela_veic}
  </div>

</body>
</html>
"""
    return html_content


# -------------------------------------------------------------------
# Rotas Flask
# -------------------------------------------------------------------
INDEX_HTML = """
<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <title>Gerador de Relatórios - Oliveira Logística</title>
  <style>
    body {
      font-family: Arial, Helvetica, sans-serif;
      margin: 40px auto;
      max-width: 900px;
      line-height: 1.4;
    }
    h1 {
      color: #008000;
      text-align: center;
    }
    fieldset {
      border: 1px solid #ccc;
      margin-top: 20px;
      padding: 15px;
    }
    legend {
      font-weight: bold;
      color: #008000;
    }
    label {
      display: block;
      margin-top: 10px;
    }
    input[type="file"],
    input[type="date"] {
      margin-top: 5px;
      padding: 4px;
      width: 100%;
      box-sizing: border-box;
    }
    .radio-group {
      margin-top: 10px;
    }
    button {
      margin-top: 20px;
      padding: 10px 20px;
      background-color: #008000;
      color: #fff;
      border: none;
      cursor: pointer;
      font-size: 14px;
      border-radius: 4px;
    }
    button:hover {
      background-color: #006600;
    }
    .alert {
      color: red;
      margin-top: 10px;
    }
  </style>
</head>
<body>
  <h1>Gerador de Relatórios - Oliveira Materiais de Construção</h1>

  {% if error %}
    <div class="alert">{{ error }}</div>
  {% endif %}

  <form method="post" action="/gerar" enctype="multipart/form-data">
    <fieldset>
      <legend>Planilha de Entregas</legend>
      <label>Arquivo Excel (.xlsx):
        <input type="file" name="planilha" accept=".xlsx,.xls" required>
      </label>
    </fieldset>

    <fieldset>
      <legend>Tipo de Relatório</legend>
      <div class="radio-group">
        <label>
          <input type="radio" name="tipo_relatorio" value="diario" checked>
          Relatório Diário (1 dia)
        </label>
        <label>
          <input type="radio" name="tipo_relatorio" value="periodo">
          Relatório de Período (início e fim)
        </label>
      </div>

      <div id="div_diario">
        <label>Data de referência:
          <input type="date" name="data_referencia">
        </label>
      </div>

      <div id="div_periodo" style="margin-top:10px;">
        <label>Data inicial:
          <input type="date" name="data_ini">
        </label>
        <label>Data final:
          <input type="date" name="data_fim">
        </label>
      </div>
    </fieldset>

    <button type="submit">Gerar PDF</button>
  </form>

  <script>
    const radioDiario = document.querySelector('input[value="diario"]');
    const radioPeriodo = document.querySelector('input[value="periodo"]');
    const divDiario = document.getElementById('div_diario');
    const divPeriodo = document.getElementById('div_periodo');

    function atualizarVisibilidade() {
      if (radioDiario.checked) {
        divDiario.style.opacity = 1;
        divDiario.style.pointerEvents = 'auto';
        divPeriodo.style.opacity = 0.4;
        divPeriodo.style.pointerEvents = 'none';
      } else {
        divDiario.style.opacity = 0.4;
        divDiario.style.pointerEvents = 'none';
        divPeriodo.style.opacity = 1;
        divPeriodo.style.pointerEvents = 'auto';
      }
    }

    radioDiario.addEventListener('change', atualizarVisibilidade);
    radioPeriodo.addEventListener('change', atualizarVisibilidade);
    atualizarVisibilidade();
  </script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML)


@app.route("/gerar", methods=["POST"])
def gerar():
    try:
        if "planilha" not in request.files:
            return render_template_string(INDEX_HTML, error="Envie a planilha de entregas.")

        file = request.files["planilha"]
        if file.filename == "":
            return render_template_string(INDEX_HTML, error="Selecione um arquivo Excel válido.")

        # Lê o Excel em memória
        df = pd.read_excel(file)
        df = preparar_dataframe(df)

        tipo = request.form.get("tipo_relatorio", "diario")

        if tipo == "diario":
            data_ref = request.form.get("data_referencia")
            if not data_ref:
                return render_template_string(INDEX_HTML, error="Informe a data de referência para o relatório diário.")
            dados = processar_dados_diarios(df, data_ref)
            html = montar_html_relatorio_diario(dados)
            nome_pdf = f"relatorio_diario_{data_ref}.pdf"
        else:
            data_ini = request.form.get("data_ini")
            data_fim = request.form.get("data_fim")
            if not data_ini or not data_fim:
                return render_template_string(INDEX_HTML, error="Informe data inicial e final para o relatório de período.")
            dados = processar_dados_periodo(df, data_ini, data_fim)
            html = montar_html_relatorio_periodo(dados)
            nome_pdf = f"relatorio_periodo_{data_ini}_a_{data_fim}.pdf"

        # Gera PDF em memória
        pdf_io = io.BytesIO()
        HTML(string=html).write_pdf(pdf_io)
        pdf_io.seek(0)

        return send_file(
            pdf_io,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=nome_pdf
        )

    except Exception as e:
        return render_template_string(INDEX_HTML, error=f"Erro ao gerar relatório: {e}")


# -------------------------------------------------------------------
# Execução
# -------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
