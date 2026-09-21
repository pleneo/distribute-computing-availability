#!/usr/bin/env python3
"""
Computação Distribuída - Disponibilidade em Sistemas Replicados
Professor Nabor C. Mendonça - UNIFOR

Simulador teórico e estocástico (Monte Carlo) de alta performance
para sistemas distribuídos em larga escala (N de 1 a 10.000).
"""

import math
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def calcular_disponibilidade(n: int, k: int, p: float) -> float:
    """
    Calcula a disponibilidade teórica A(n, k, p) de um sistema com n servidores,
    onde no mínimo k servidores precisam estar ativos simultaneamente, cada um com confiabilidade p.

    Fórmula Geral:
        A(n, k, p) = sum_{i=k}^n comb(n, i) * (p**i) * ((1 - p)**(n - i))

    Para evitar overflow em clusters grandes (ex.: n=10000), utiliza a função de sobrevivência
    (Survival Function) da distribuição binomial com suporte a ponto flutuante de alta precisão.
    """
    if not (0 < k <= n):
        raise ValueError(f"Parâmetros inválidos: requer 0 < k <= n (recebido n={n}, k={k})")
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"Probabilidade p deve estar no intervalo [0, 1] (recebido p={p})")

    if p == 0.0:
        return 0.0
    if p == 1.0:
        return 1.0

    if k == 1:
        return float(1.0 - (1.0 - p) ** n)
    if k == n:
        return float(p ** n)

    # stats.binom.sf(k - 1, n, p) calcula P(X >= k)
    val = float(stats.binom.sf(k - 1, n, p))
    return min(max(val, 0.0), 1.0)


def simulador_monte_carlo(n: int, k: int, p: float, rodadas: int = 30000, seed: int = None) -> float:
    """
    Simulação estocástica de Monte Carlo de alta performance.
    
    A contagem de nós ativos em cada rodada segue a distribuição Binomial(n, p).
    A amostragem direta vetorizada opera em O(rodadas), consumindo pouca memória
    e executando em milissegundos mesmo para n=10000.
    """
    if rodadas <= 0:
        raise ValueError("O número de rodadas deve ser positivo.")
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0

    rng = np.random.default_rng(seed)
    ativos_por_rodada = rng.binomial(n=n, p=p, size=rodadas)
    sucessos = np.count_nonzero(ativos_por_rodada >= k)

    return float(sucessos / rodadas)


# ==============================================================================
# BATERIAS DE EXPERIMENTOS
# ==============================================================================

def executar_operacoes_n_fixo(
    n: int = 1000,
    passos_p: int = 41,
    rodadas_mc: int = 30000,
    seed: int = 42
) -> pd.DataFrame:
    """
    Experimento 1: N fixado (N=1000) avaliando os 3 requisitos clássicos de servidores mínimos:
    - Consulta: k = 1
    - Maioria Simples: k = ceil(n / 2)
    - Atualização: k = n
    """
    valores_p_base = np.linspace(0.0, 1.0, passos_p)
    p_denso_maioria = np.linspace(0.46, 0.54, 17)
    p_denso_escrita = np.array([0.990, 0.993, 0.995, 0.997, 0.998, 0.999, 0.9995])
    valores_p = np.unique(np.sort(np.concatenate([valores_p_base, p_denso_maioria, p_denso_escrita])))

    k_maioria = math.ceil(n / 2)
    operacoes = {
        "1 Servidor Ativo (k = 1)": 1,
        "Metade dos Servidores (k = n/2)": k_maioria,
        "Todos os Servidores (k = n)": n,
    }

    registros = []
    print(f"[*] Executando Experimento 1: N={n} fixado com k=1, k=n/2 e k=n...")

    for label_op, k in operacoes.items():
        for p in valores_p:
            p_float = float(p)
            disp_teo = calcular_disponibilidade(n, k, p_float)
            disp_sim = simulador_monte_carlo(n, k, p_float, rodadas=rodadas_mc, seed=seed)
            registros.append({
                "n": n,
                "k": k,
                "operacao": label_op,
                "p": round(p_float, 5),
                "disponibilidade_teorica": round(disp_teo, 6),
                "disponibilidade_simulada": round(disp_sim, 6),
                "erro_absoluto": round(abs(disp_teo - disp_sim), 6)
            })

    return pd.DataFrame(registros)


def executar_escala_tres_operacoes(
    lista_n: list[int] = [1, 5, 100, 1000, 10000],
    passos_p: int = 41,
    rodadas_mc: int = 30000,
    seed: int = 42
) -> pd.DataFrame:
    """
    Experimento 2: Escala de N variando em valores grandes (1, 5, 100, 1000, 10000)
    para as três configurações padronizadas:
    - 1 Servidor Ativo (k = 1)
    - Metade dos Servidores (k = n/2)
    - Todos os Servidores (k = n)
    """
    valores_p = np.linspace(0.0, 1.0, passos_p)
    # Pontos adicionais próximos a zero e a um para curvas com transições fortes
    valores_p_refinados = np.unique(np.sort(np.concatenate([
        valores_p,
        np.array([0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.48, 0.49, 0.50, 0.51, 0.52, 0.995, 0.999])
    ])))

    registros = []
    print("[*] Executando Experimento 2: Escala de N (1 a 10.000) para k=1, k=n/2 e k=n...")

    for n in lista_n:
        casos = {
            "1 Servidor Ativo (k = 1)": 1,
            "Metade dos Servidores (k = n/2)": math.ceil(n / 2),
            "Todos os Servidores (k = n)": n,
        }

        for tipo_op, k in casos.items():
            for p in valores_p_refinados:
                p_float = float(p)
                disp_teo = calcular_disponibilidade(n, k, p_float)
                disp_sim = simulador_monte_carlo(n, k, p_float, rodadas=rodadas_mc, seed=seed)
                registros.append({
                    "n": n,
                    "k": k,
                    "tipo_operacao": tipo_op,
                    "p": round(p_float, 5),
                    "disponibilidade_teorica": round(disp_teo, 6),
                    "disponibilidade_simulada": round(disp_sim, 6),
                    "erro_absoluto": round(abs(disp_teo - disp_sim), 6)
                })

    return pd.DataFrame(registros)


def executar_variando_k(
    n: int = 1000,
    lista_p: list[float] = [0.2, 0.4, 0.6, 0.8, 0.95],
    passos_k: int = 50,
    rodadas_mc: int = 30000,
    seed: int = 42
) -> pd.DataFrame:
    """
    Experimento 3: N fixado (N=1000) e k variando de 1 a N.
    """
    k_base = np.unique(np.linspace(1, n, passos_k, dtype=int))
    registros = []
    print(f"[*] Executando Experimento 3: N={n} fixado e k variando de 1 a {n}...")

    for p in lista_p:
        k_corte = int(round(n * p))
        k_redor = [max(1, k_corte - 5), max(1, k_corte - 2), k_corte, min(n, k_corte + 2), min(n, k_corte + 5)]
        k_amostrados = np.unique(np.sort(np.concatenate([k_base, k_redor])))

        for k in k_amostrados:
            k_int = int(k)
            disp_teo = calcular_disponibilidade(n, k_int, p)
            disp_sim = simulador_monte_carlo(n, k_int, p, rodadas=rodadas_mc, seed=seed)
            registros.append({
                "n": n,
                "p": p,
                "k": k_int,
                "k_sobre_n": round(k_int / n, 4),
                "disponibilidade_teorica": round(disp_teo, 6),
                "disponibilidade_simulada": round(disp_sim, 6),
                "erro_absoluto": round(abs(disp_teo - disp_sim), 6)
            })

    return pd.DataFrame(registros)


# ==============================================================================
# GERAÇÃO DE GRÁFICOS
# ==============================================================================

def gerar_graficos(
    df_operacoes: pd.DataFrame,
    df_escala: pd.DataFrame,
    df_var_k: pd.DataFrame,
    output_dir: Path
):
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9.5,
        "figure.titlesize": 13
    })

    # --------------------------------------------------------------------------
    # 1. Gráfico: N = 1000 Fixado com as 3 Operações
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.8), dpi=180)
    n_val = df_operacoes["n"].iloc[0]

    estilos = {
        "1 Servidor Ativo (k = 1)": {"cor": "#2b8a3e", "marker": "o", "label": "1 Servidor Ativo (k = 1)"},
        "Metade dos Servidores (k = n/2)": {"cor": "#1971c2", "marker": "s", "label": f"Metade dos Servidores (k = {math.ceil(n_val/2)})"},
        "Todos os Servidores (k = n)": {"cor": "#e03131", "marker": "^", "label": f"Todos os Servidores (k = {n_val})"},
    }

    for operacao in df_operacoes["operacao"].unique():
        sub = df_operacoes[df_operacoes["operacao"] == operacao].sort_values("p")
        cfg = estilos.get(operacao, {"cor": "#333333", "marker": "o", "label": operacao})

        # Linha contínua do modelo teórico
        ax.plot(sub["p"], sub["disponibilidade_teorica"], label=cfg["label"], color=cfg["cor"], linewidth=2.4)
        
        # Marcadores da simulação estocástica (amostrados para evitar poluição visual)
        sub_amostra = sub.iloc[::max(1, len(sub) // 20)]
        ax.scatter(sub_amostra["p"], sub_amostra["disponibilidade_simulada"], color=cfg["cor"], marker=cfg["marker"], s=36, edgecolors="black", linewidths=0.5, zorder=4)

    # Linha vertical discreta de referência em p = 0.5
    ax.axvline(0.5, color="#868e96", linestyle=":", alpha=0.7)

    ax.set_title(f"Disponibilidade do Serviço para N = {n_val} Servidores", fontweight="bold", pad=12)
    ax.set_xlabel("Confiabilidade de cada servidor (p)", labelpad=8)
    ax.set_ylabel("Disponibilidade do Serviço A(N, k, p)", labelpad=8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Legenda limpa e unificada
    ax.legend(title="Requisito Mínimo (Linha: Teórico | Pontos: Simulação)", loc="center right", framealpha=0.92)

    caminho1 = output_dir / "grafico_operacoes_n1000.png"
    fig.tight_layout()
    fig.savefig(caminho1, bbox_inches="tight")
    plt.close(fig)
    print(f"  [+] Gráfico salvo: {caminho1}")

    # --------------------------------------------------------------------------
    # 2. Gráfico: Escala de N (1 a 10.000) para k=1, k=n/2 e k=n
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=180)
    fig.suptitle("Efeito da Escala de Servidores (N de 1 a 10.000)", fontweight="bold", y=1.02)

    cores_n = {
        1: "#868e96",
        5: "#f59f00",
        100: "#20c997",
        1000: "#1971c2",
        10000: "#7950f2"
    }

    painéis = [
        ("1 Servidor Ativo (k = 1)", axes[0], "1 Servidor Ativo (k = 1)"),
        ("Metade dos Servidores (k = n/2)", axes[1], "Metade dos Servidores (k = n/2)"),
        ("Todos os Servidores (k = n)", axes[2], "Todos os Servidores (k = n)"),
    ]

    for tipo_op, ax, titulo in painéis:
        df_sub_op = df_escala[df_escala["tipo_operacao"] == tipo_op]
        for n_val in sorted(df_sub_op["n"].unique()):
            sub = df_sub_op[df_sub_op["n"] == n_val].sort_values("p")
            cor = cores_n.get(n_val, "#333333")
            ax.plot(sub["p"], sub["disponibilidade_teorica"], label=f"N = {n_val:,}", color=cor, linewidth=2.2)
            sub_amostra = sub.iloc[::max(1, len(sub) // 14)]
            ax.scatter(sub_amostra["p"], sub_amostra["disponibilidade_simulada"], color=cor, s=24, edgecolors="black", linewidths=0.4, zorder=4)

        if "Metade" in tipo_op:
            ax.axvline(0.5, color="#868e96", linestyle=":", alpha=0.7)

        ax.set_title(titulo, fontweight="bold", fontsize=11, pad=10)
        ax.set_xlabel("Confiabilidade de cada servidor (p)", labelpad=7)
        ax.set_ylabel("Disponibilidade do Serviço", labelpad=7)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(title="Tamanho (N)", loc="best", framealpha=0.9)

    caminho2 = output_dir / "grafico_escala_n_tres_operacoes.png"
    fig.tight_layout()
    fig.savefig(caminho2, bbox_inches="tight")
    plt.close(fig)
    print(f"  [+] Gráfico salvo: {caminho2}")

    # --------------------------------------------------------------------------
    # 3. Gráfico: N = 1000 Fixado e k Variando de 1 a N
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 6), dpi=180)
    n_var = df_var_k["n"].iloc[0]

    cores_p = {
        0.2: "#e03131",
        0.4: "#f76707",
        0.6: "#1971c2",
        0.8: "#2b8a3e",
        0.95: "#7950f2"
    }

    for p_val in sorted(df_var_k["p"].unique()):
        sub = df_var_k[df_var_k["p"] == p_val].sort_values("k")
        cor = cores_p.get(p_val, "#333333")
        # Sem rótulo redundante "(Teórico)"
        ax.plot(sub["k"], sub["disponibilidade_teorica"], label=f"p = {p_val}", color=cor, linewidth=2.3)
        sub_amostra = sub.iloc[::max(1, len(sub) // 14)]
        ax.scatter(sub_amostra["k"], sub_amostra["disponibilidade_simulada"], color=cor, s=26, alpha=0.8, edgecolors="black", linewidths=0.4, zorder=4)

        # Ponto de transição em k = N * p
        k_crit = int(round(n_var * p_val))
        ax.axvline(k_crit, color=cor, linestyle=":", alpha=0.55)

    ax.set_title(f"Disponibilidade vs. Número Mínimo de Servidores Ativos (k) para N = {n_var}", fontweight="bold", pad=12)
    ax.set_xlabel(f"Servidores Mínimos Requeridos k (de 1 a {n_var})", labelpad=8)
    ax.set_ylabel(f"Disponibilidade do Serviço A({n_var}, k, p)", labelpad=8)
    ax.set_xlim(0, n_var + 15)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Legenda limpa com título solicitado: "Confiabilidade (p)"
    ax.legend(title="Confiabilidade (p)", loc="upper right", framealpha=0.92)

    caminho3 = output_dir / "grafico_n1000_variando_k.png"
    fig.tight_layout()
    fig.savefig(caminho3, bbox_inches="tight")
    plt.close(fig)
    print(f"  [+] Gráfico salvo: {caminho3}")


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Simulador de Disponibilidade - Computação Distribuída")
    parser.add_argument("--rodadas", type=int, default=30000, help="Rodadas Monte Carlo por ponto (padrão: 30000)")
    parser.add_argument("--passos", type=int, default=35, help="Divisões no intervalo p in [0, 1] (padrão: 35)")
    args = parser.parse_args()

    diretorio_base = Path(__file__).resolve().parent
    diretorio_resultados = diretorio_base / "resultados"
    diretorio_resultados.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("COMPUTAÇÃO DISTRIBUÍDA - SIMULADOR DE DISPONIBILIDADE EM LARGA ESCALA")
    print("Prof. Nabor C. Mendonça - UNIFOR")
    print("=" * 75)
    print(f"Parâmetros: {args.rodadas:,} rodadas Monte Carlo | {args.passos} divisões de p")

    # 1. Experimento: N=1000 fixo com Consulta (k=1), Maioria (k=n/2) e Atualização (k=n)
    df_operacoes = executar_operacoes_n_fixo(
        n=1000,
        passos_p=args.passos,
        rodadas_mc=args.rodadas,
        seed=42
    )
    df_operacoes.to_csv(diretorio_resultados / "tabela_operacoes_n1000.csv", index=False)
    df_operacoes.to_csv(diretorio_resultados / "tabela_comparativa.csv", index=False)

    # 2. Experimento: Escala de N variando (1, 5, 100, 1000, 10000) para Consulta, Maioria e Atualização
    df_escala = executar_escala_tres_operacoes(
        lista_n=[1, 5, 100, 1000, 10000],
        passos_p=args.passos,
        rodadas_mc=args.rodadas,
        seed=42
    )
    df_escala.to_csv(diretorio_resultados / "tabela_escala_n.csv", index=False)

    # 3. Experimento: N=1000 fixo e k variando de 1 a N
    df_var_k = executar_variando_k(
        n=1000,
        lista_p=[0.2, 0.4, 0.6, 0.8, 0.95],
        passos_k=45,
        rodadas_mc=args.rodadas,
        seed=42
    )
    df_var_k.to_csv(diretorio_resultados / "tabela_variando_k_n1000.csv", index=False)

    print("\n[OK] Tabelas CSV exportadas com sucesso.")

    # Gera os gráficos
    print("\nGerando gráficos limpos e padronizados...")
    gerar_graficos(df_operacoes, df_escala, df_var_k, diretorio_resultados)

    # Métricas consolidadas
    erros = np.concatenate([
        df_operacoes["erro_absoluto"].values,
        df_escala["erro_absoluto"].values,
        df_var_k["erro_absoluto"].values
    ])
    print("\nMétricas de Precisão Geral (Simulado vs. Teórico):")
    print(f"  - Erro Absoluto Médio:  {np.mean(erros):.6f}")
    print(f"  - Erro Absoluto Máximo: {np.max(erros):.6f}")

    print("\n" + "=" * 75)
    print("SIMULAÇÃO CONCLUÍDA COM SUCESSO!")
    print(f"Resultados disponíveis em: {diretorio_resultados}")
    print("=" * 75)


if __name__ == "__main__":
    main()
