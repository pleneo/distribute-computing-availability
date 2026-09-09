#!/usr/bin/env python3
"""
Computação Distribuída - Trabalho 1 (Exercício 1.2)
Prof. Nabor C. Mendonça - UNIFOR

Este módulo implementa:
1. O cálculo analítico da disponibilidade de serviço replicado em n servidores com mínimo k ativos.
2. O simulador estocástico de Monte Carlo para validação experimental.
3. Geração de tabelas comparativas (CSV) e gráficos 2D (PNG).
"""

import os
import math
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==============================================================================
# 1. CÁLCULO ANALÍTICO (FÓRMULAS DEDUZIDAS NO EXERCÍCIO 1.1)
# ==============================================================================

def disponibilidade_analitica(n: int, k: int, p: float) -> float:
    """
    Calcula a disponibilidade teórica A(n, k, p) de um cluster com n servidores,
    onde no mínimo k servidores precisam estar disponíveis, cada um com prob p.

    Fórmula Geral:
        A(n, k, p) = sum_{i=k}^n comb(n, i) * (p**i) * ((1 - p)**(n - i))

    Casos Extremos Otimizados:
        - k = 1 (Consulta): A(n, 1, p) = 1 - (1 - p)**n
        - k = n (Atualização): A(n, n, p) = p**n
    """
    if not (0 < k <= n):
        raise ValueError(f"Parâmetros inválidos: requer 0 < k <= n (recebido n={n}, k={k})")
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"Probabilidade p deve estar no intervalo [0, 1] (recebido p={p})")

    # Limites triviais
    if p == 0.0:
        return 0.0
    if p == 1.0:
        return 1.0

    # Otimização caso k = 1 (Consulta)
    if k == 1:
        return 1.0 - (1.0 - p) ** n

    # Otimização caso k = n (Atualização)
    if k == n:
        return p ** n

    # Caso geral: somatório da distribuição binomial acumulada
    prob_total = 0.0
    for i in range(k, n + 1):
        termo = math.comb(n, i) * (p ** i) * ((1.0 - p) ** (n - i))
        prob_total += termo

    return min(max(prob_total, 0.0), 1.0)


# ==============================================================================
# 2. SIMULADOR ESTOCÁSTICO (MONTE CARLO)
# ==============================================================================

def simulador_monte_carlo(n: int, k: int, p: float, rodadas: int = 50000, seed: int = None) -> float:
    """
    Simula estocasticamente a disponibilidade de um serviço replicado.

    Para cada rodada:
        1. Sorteia o estado de cada um dos n servidores (1 se rand <= p, senão 0).
        2. Conta o número de servidores disponíveis.
        3. Verifica se a condição operacional foi atingida (ativos >= k).

    Retorna a frequência experimental: (rodadas com sucesso) / rodadas.
    """
    if rodadas <= 0:
        raise ValueError("O número de rodadas deve ser positivo.")

    # Gerador de números aleatórios moderno do NumPy
    rng = np.random.default_rng(seed)

    # Matriz booleana de dimensões (rodadas, n): True se servidor ativo (rand <= p)
    servidores_ativos = rng.random(size=(rodadas, n)) <= p

    # Contagem de servidores ativos por rodada (soma na dimensão das colunas)
    total_ativos_por_rodada = np.sum(servidores_ativos, axis=1)

    # Contagem de rodadas bem-sucedidas (total_ativos >= k)
    sucessos = np.count_nonzero(total_ativos_por_rodada >= k)

    return float(sucessos / rodadas)


# ==============================================================================
# 3. EXPERIMENTAÇÃO E TABULAÇÃO DE DADOS
# ==============================================================================

def executar_experimentos(
    lista_n: list[int] = [3, 5, 7],
    passos_p: int = 21,
    rodadas_mc: int = 50000,
    seed: int = 42
) -> pd.DataFrame:
    """
    Executa a bateria completa de comparações para os cenários pedidos:
    - Casos específicos: k = 1, k = ceil(n/2), k = n
    - Faixa de probabilidade p: [0.0, 1.0]
    """
    valores_p = np.linspace(0.0, 1.0, passos_p)
    registros = []

    print(f"[*] Iniciando experimentos (Rodadas Monte Carlo por ponto: {rodadas_mc:,})...")

    for n in lista_n:
        casos_k = {
            "Consulta (k = 1)": 1,
            f"Maioria Simples (k = {math.ceil(n / 2)})": math.ceil(n / 2),
            f"Atualização (k = {n})": n,
        }

        for tipo_rotulo, k in casos_k.items():
            for p in valores_p:
                p_float = float(p)
                disp_analitica = disponibilidade_analitica(n, k, p_float)
                disp_simulada = simulador_monte_carlo(n, k, p_float, rodadas=rodadas_mc, seed=seed)
                erro_abs = abs(disp_analitica - disp_simulada)

                registros.append({
                    "n": n,
                    "k": k,
                    "operacao": tipo_rotulo,
                    "p": round(p_float, 4),
                    "disponibilidade_analitica": round(disp_analitica, 5),
                    "disponibilidade_simulada": round(disp_simulada, 5),
                    "erro_absoluto": round(erro_abs, 5),
                })

    df = pd.DataFrame(registros)
    return df


# ==============================================================================
# 4. GERAÇÃO DE GRÁFICOS 2D
# ==============================================================================

def gerar_graficos(df: pd.DataFrame, output_dir: Path):
    """
    Gera gráficos 2D para cada valor de n e um gráfico panorâmico geral.
    - Linha contínua: Cálculo Analítico
    - Marcadores pontuais: Simulação Estocástica
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    lista_n = df["n"].unique()

    # Cores e estilos consistentes
    estilos = {
        "Consulta": {"cor": "#2b8a3e", "marker": "o", "label_prefix": "k=1 (Consulta)"},
        "Maioria": {"cor": "#1971c2", "marker": "s", "label_prefix": "k=maioria"},
        "Atualizacao": {"cor": "#e03131", "marker": "^", "label_prefix": "k=n (Atualização)"},
    }

    # 1. Gráficos individuais por n
    for n in lista_n:
        df_n = df[df["n"] == n]
        fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)

        for operacao in df_n["operacao"].unique():
            df_op = df_n[df_n["operacao"] == operacao]

            if "k = 1" in operacao:
                cfg = estilos["Consulta"]
            elif "Atualização" in operacao:
                cfg = estilos["Atualizacao"]
            else:
                cfg = estilos["Maioria"]

            label_nome = operacao

            # Linha analítica contínua
            ax.plot(
                df_op["p"],
                df_op["disponibilidade_analitica"],
                label=f"Analítico: {label_nome}",
                color=cfg["cor"],
                linewidth=2.2,
                alpha=0.85
            )

            # Marcadores discretos da simulação estocástica
            ax.scatter(
                df_op["p"],
                df_op["disponibilidade_simulada"],
                label=f"Simulado: {label_nome}",
                color=cfg["cor"],
                marker=cfg["marker"],
                s=36,
                edgecolors="black",
                linewidths=0.5,
                zorder=4
            )

        ax.set_title(f"Disponibilidade do Serviço vs. Confiabilidade Individual (n = {n} servidores)", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Probabilidade de cada servidor estar ativo (p)", fontsize=11, labelpad=8)
        ax.set_ylabel("Disponibilidade do Serviço A(n, k, p)", fontsize=11, labelpad=8)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

        caminho_grafico = output_dir / f"grafico_disponibilidade_n{n}.png"
        fig.savefig(caminho_grafico, bbox_inches="tight")
        plt.close(fig)
        print(f"  [+] Gráfico salvo: {caminho_grafico}")

    # 2. Gráfico comparativo geral de escalabilidade: n=3 vs n=7 para k=1 e k=n
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    cores_comp = {
        (3, 1): ("#2b8a3e", "-"),
        (7, 1): ("#099268", "--"),
        (3, 3): ("#e03131", "-"),
        (7, 7): ("#c92a2a", "--"),
    }

    for (n_val, k_val), (cor, estilo_linha) in cores_comp.items():
        df_sub = df[(df["n"] == n_val) & (df["k"] == k_val)]
        if not df_sub.empty:
            tipo_txt = "Consulta" if k_val == 1 else "Atualização"
            ax.plot(
                df_sub["p"],
                df_sub["disponibilidade_analitica"],
                label=f"Analítico n={n_val}, k={k_val} ({tipo_txt})",
                color=cor,
                linestyle=estilo_linha,
                linewidth=2
            )
            ax.scatter(
                df_sub["p"],
                df_sub["disponibilidade_simulada"],
                label=f"Simulado n={n_val}, k={k_val}",
                color=cor,
                s=28,
                alpha=0.7
            )

    ax.set_title("Efeito da Escala (n=3 vs n=7) nos Extremos de Consulta e Atualização", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Probabilidade de cada servidor estar ativo (p)", fontsize=11)
    ax.set_ylabel("Disponibilidade do Serviço", fontsize=11)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)

    caminho_comp = output_dir / "grafico_comparativo_escala.png"
    fig.savefig(caminho_comp, bbox_inches="tight")
    plt.close(fig)
    print(f"  [+] Gráfico comparativo salvo: {caminho_comp}")


# ==============================================================================
# 5. EXECUÇÃO PRINCIPAL
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Simulador de Disponibilidade - Trabalho 1 (Comp. Distribuída)")
    parser.add_argument("--rodadas", type=int, default=30000, help="Número de rodadas por simulação de Monte Carlo (padrão: 30000)")
    parser.add_argument("--passos", type=int, default=21, help="Quantidade de divisões no intervalo p in [0, 1] (padrão: 21)")
    args = parser.parse_args()

    diretorio_base = Path(__file__).resolve().parent
    diretorio_resultados = diretorio_base / "resultados"
    diretorio_resultados.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("COMPUTACAO DISTRIBUIDA - SIMULADOR ANALITICO E ESTOCASTICO")
    print("Prof. Nabor C. Mendonca - UNIFOR")
    print("=" * 70)

    # Executa bateria de testes para n = 3, n = 5 e n = 7
    df_resultados = executar_experimentos(
        lista_n=[3, 5, 7],
        passos_p=args.passos,
        rodadas_mc=args.rodadas,
        seed=42
    )

    # Salva tabela CSV
    caminho_csv = diretorio_resultados / "tabela_comparativa.csv"
    df_resultados.to_csv(caminho_csv, index=False)
    print(f"\n[OK] Tabela comparativa salva com sucesso em:\n     -> {caminho_csv}")

    # Exibe amostra da tabela no terminal
    print("\nAmostra dos Resultados (Primeiras 10 linhas):")
    print(df_resultados[["n", "k", "operacao", "p", "disponibilidade_analitica", "disponibilidade_simulada", "erro_absoluto"]].head(10).to_string(index=False))

    # Erro médio absoluto entre teoria e prática
    erro_medio = df_resultados["erro_absoluto"].mean()
    erro_max = df_resultados["erro_absoluto"].max()
    print(f"\nMetricas de Precisao (Lei dos Grandes Numeros):")
    print(f"    - Erro Absoluto Medio: {erro_medio:.5f}")
    print(f"    - Erro Absoluto Maximo: {erro_max:.5f}")

    # Gera gráficos 2D
    print("\nGerando graficos 2D...")
    gerar_graficos(df_resultados, diretorio_resultados)

    print("\n" + "=" * 70)
    print("EXECUCAO CONCLUIDA COM SUCESSO")
    print(f"Todos os artefatos foram gerados na pasta: {diretorio_resultados}")
    print("=" * 70)


if __name__ == "__main__":
    main()
