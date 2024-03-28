from argparse import ArgumentParser

import pandas as pd
from eulerlagrange import EulerLagrangeSimulation
from hamilton import HamiltonSimulation
from dataclasses import asdict

ARGS = [
    {
        "flags": ["--m1"],
        "type": float,
        "default": 0.1,
        "help": "Massa do pêndulo 1 (kg)",
    },
    {
        "flags": ["--m2"],
        "type": float,
        "default": 0.005,
        "help": "Massa do pêndulo 2 (kg)",
    },
    {
        "flags": ["--l1"],
        "type": float,
        "default": 0.5,
        "help": "Tamanho do pêndulo 1 (m)",
    },
    {
        "flags": ["--l2"],
        "type": float,
        "default": 0.2,
        "help": "Tamanho do pêndulo 2 (m)",
    },
    {"flags": ["--h"], "type": float, "default": 0.01, "help": "Passo temporal (s)"},
    {
        "flags": ["--tf"],
        "type": float,
        "default": 100,
        "help": "Duração da simulação (s)",
    },
    {
        "flags": ["--theta1"],
        "type": float,
        "default": 0,
        "help": "Ângulo inicial do pêndulo 1 (rad)",
    },
    {
        "flags": ["--p1"],
        "type": float,
        "default": 0.09752,
        "help": "Momento angular inicial do pêndulo 1 (kg m²/s)",
    },
    {
        "flags": ["--theta2"],
        "type": float,
        "default": 1.5708,
        "help": "Ângulo inicial do pêndulo 2 (rad)",
    },
    {
        "flags": ["--p2"],
        "type": float,
        "default": 0,
        "help": "Momento angular inicial do pêndulo 2 (kg m²/s)",
    },
    {
        "flags": ["--output"],
        "type": str,
        "default": "results.csv",
        "help": "Nome do arquivo de saída",
    },
    {
        "flags": ["--method"],
        "type": str,
        "default": "eulerlagrange",
        "help": "Método de simulação (eulerlagrange ou hamilton)",
    },
]

methods = {
    "eulerlagrange": EulerLagrangeSimulation,
    "hamilton": HamiltonSimulation,
}

if __name__ == "__main__":
    parser = ArgumentParser(description="Simulação de um pêndulo duplo")
    for arg in ARGS:
        flags = arg.pop("flags")
        parser.add_argument(*flags, **arg)
    # Adiciona o
    args = parser.parse_args()

    simulation = methods[args.method](m1=args.m1, m2=args.m2, l1=args.l1, l2=args.l2)
    s1, s2, energy = simulation.simulate(
        h=args.h,
        tf=args.tf,
        theta1=args.theta1,
        p1=args.p1,
        theta2=args.theta2,
        p2=args.p2,
    )

    # Salva os resultados em um arquivo CSV
    s1 = [asdict(s) for s in s1]
    s2 = [asdict(s) for s in s2]
    df1 = pd.DataFrame(s1).rename(columns={"theta": "theta1", "p": "p1"})
    df2 = pd.DataFrame(s2).rename(columns={"theta": "theta2", "p": "p2"})
    dfe = pd.DataFrame(energy, columns=["energy"])
    df = pd.concat([df1, df2, dfe], axis=1)
    df.to_csv(args.output, index=False)
