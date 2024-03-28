from abc import ABC, abstractmethod
from argparse import ArgumentParser
from dataclasses import asdict, dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass
class PendulumState:
    """Define o estado de um pêndulo."""

    theta: float
    p: float


@dataclass
class Pendulum:
    """Define as propriedades de um pêndulo."""

    m: float
    l: float


class RK2Simulation(ABC):
    """Classe abstrata para implementação de uma simulação do pêndulo duplo.

    Parameters
    ----------
    m1 : float
        Massa do pêndulo 1.
    m2 : float
        Massa do pêndulo 2.
    l1 : float
        Tamanho do pêndulo 1.
    l2 : float
        Tamanho do pêndulo 2.
    """

    p1: Pendulum
    p2: Pendulum
    g: float = 10

    def __init__(
        self,
        m1: float,
        m2: float,
        l1: float,
        l2: float,
    ) -> None:
        self.p1 = Pendulum(m=m1, l=l1)
        self.p2 = Pendulum(m=m2, l=l2)

    @abstractmethod
    def eval_derivatives(
        p1: Pendulum, p2: Pendulum, s1: PendulumState, s2: PendulumState
    ) -> Tuple[float, float, float, float]:
        """Calculo das derivadas para um pêndulo duplo.

        Parameters
        ----------
        p1 : Pendulum
            Atributos do pêndulo 1.
        p2 : Pendulum
            Atributos do pêndulo 2.
        s1 : PendulumState
            Estado do pêndulo 1.
        s2 : PendulumState
            Estado do pêndulo 2.

        Returns
        -------
        Tuple[float, float, float, float]
        """
        pass

    def eval_energy(
        self, p1: Pendulum, p2: Pendulum, s1: PendulumState, s2: PendulumState
    ) -> float:
        """Calcula a energia total do sistema.

        Parameters
        ----------
        p1 : Pendulum
            Atributos do pêndulo 1.
        p2 : Pendulum
            Atributos do pêndulo 2.
        s1 : PendulumState
            Estado do pêndulo 1.
        s2 : PendulumState
            Estado do pêndulo 2.

        Returns
        -------
        float
            Energia total do sistema.
        """
        # Energia cinética
        T1 = s1.p**2 / (2 * p1.m * p1.l**2)
        T2 = s2.p**2 / (2 * p2.m * p2.l**2)

        # Energia potencial gravitacional
        G1 = p1.m * self.g * p1.l * (1 - np.cos(s1.theta))
        G2 = p2.m * self.g * p2.l * (1 - np.cos(s2.theta))
        U = (s1.p**2 * s2.p**2 * np.cos(s1.theta - s2.theta)) / (p1.m * p1.l * p2.l)

        return T1 + T2 + G1 + G2 - U

    def simulate(
        self, h: float, tf: float, theta1: float, p1: float, theta2: float, p2: float
    ) -> Tuple[List[PendulumState], List[PendulumState], List[float]]:
        # Inicializa a lista de estados
        states1 = [PendulumState(theta=theta1, p=p1)]
        states2 = [PendulumState(theta=theta2, p=p2)]
        energy = [self.eval_energy(self.p1, self.p2, states1[-1], states2[-1])]

        # Quantidade de iterações a serem realizadas
        n_iterations = len(np.arange(0, tf + h, h)) - 1

        for _ in range(n_iterations):
            # Calculando as derivadas
            dt1, dp1, dt2, dp2 = self.eval_derivatives(
                p1=self.p1, p2=self.p2, s1=states1[-1], s2=states2[-1]
            )

            # Estados ao meio passo
            state1_half = PendulumState(
                theta=self.normalize_angle(states1[-1].theta + dt1 * h * 0.5),
                p=states1[-1].p + dp1 * h * 0.5,
            )
            state2_half = PendulumState(
                theta=self.normalize_angle(states2[-1].theta + dt2 * h * 0.5),
                p=states2[-1].p + dp2 * h * 0.5,
            )

            # Calculando as derivadas novamente
            dt1, dp1, dt2, dp2 = self.eval_derivatives(
                p1=self.p1, p2=self.p2, s1=state1_half, s2=state2_half
            )

            # Adicionando os novos estados
            states1.append(
                PendulumState(
                    theta=self.normalize_angle(states1[-1].theta + dt1 * h),
                    p=states1[-1].p + dp1 * h,
                )
            )
            states2.append(
                PendulumState(
                    theta=self.normalize_angle(states2[-1].theta + dt2 * h),
                    p=states2[-1].p + dp2 * h,
                )
            )
            energy.append(self.eval_energy(self.p1, self.p2, states1[-1], states2[-1]))

        return states1, states2, energy

    @staticmethod
    def normalize_angle(angle: float) -> float:
        """Normaliza um ângulo para que fique entre -pi e pi.

        Parameters
        ----------
        angle : float

        Returns
        -------
        float
            Ângulo normalizado.
        """
        if angle > np.pi:
            return angle - 2 * np.pi
        elif angle < -np.pi:
            return angle + 2 * np.pi
        else:
            return angle


class EulerLagrangeSimulation(RK2Simulation):
    def eval_derivatives(
        self, p1: Pendulum, p2: Pendulum, s1: PendulumState, s2: PendulumState
    ) -> Tuple[float, float, float, float]:
        """Calculo das derivadas para um pêndulo duplo.

        Parameters
        ----------
        p1 : Pendulum
            Atributos do pêndulo 1.
        p2 : Pendulum
            Atributos do pêndulo 2.
        s1 : PendulumState
            Estado do pêndulo 1.
        s2 : PendulumState
            Estado do pêndulo 2.

        Returns
        -------
        Tuple[float, float, float, float]
        """
        # Derivadas do ângulo
        dt1 = s1.p / (p1.m * p1.l**2) - (s2.p * np.cos(s1.theta - s2.theta)) / (
            p1.m * p1.l * p2.l
        )
        dt2 = s2.p / (p2.m * p2.l**2) - (s1.p * np.cos(s1.theta - s2.theta)) / (
            p1.m * p1.l * p2.l
        )

        # Derivadas do momento angular
        dp1 = -p1.m * self.g * p1.l * np.sin(s1.theta) - (
            s1.p * s2.p * np.sin(s1.theta - s2.theta)
        ) / (p1.m * p1.l * p2.l)
        dp2 = -p2.m * self.g * p2.l * np.sin(s2.theta) + (
            s1.p * s2.p * np.sin(s1.theta - s2.theta)
        ) / (p1.m * p1.l * p2.l)

        return dt1, dp1, dt2, dp2


class HamiltonSimulation(RK2Simulation):
    def eval_derivatives(
        self, p1: Pendulum, p2: Pendulum, s1: PendulumState, s2: PendulumState
    ) -> Tuple[float, float, float, float]:
        """Calculo das derivadas para um pêndulo duplo.

        Parameters
        ----------
        p1 : Pendulum
            Atributos do pêndulo 1.
        p2 : Pendulum
            Atributos do pêndulo 2.
        s1 : PendulumState
            Estado do pêndulo 1.
        s2 : PendulumState
            Estado do pêndulo 2.

        Returns
        -------
        Tuple[float, float, float, float]
        """
        # Derivadas do ângulo
        dt1 = (p2.l * s1.p - p1.l * s2.p * np.cos(s1.theta - s2.theta)) / (
            p1.l**2 * p2.l * (p1.m + p2.m * np.sin(s1.theta - s2.theta) ** 2)
        )
        dt2 = (
            -p2.l * s1.p * np.cos(s1.theta - s2.theta) + p1.l * (1 + p1.m / p2.m) * s2.p
        ) / (p1.l * p2.l**2 * (p1.m + p2.m * np.sin(s1.theta - s2.theta) ** 2))

        # Derivadas do momento angular
        A = (s1.p * s2.p * np.sin(s1.theta - s2.theta)) / (
            p1.l * p2.l * (p1.m + p2.m * np.sin(s1.theta - s2.theta) ** 2)
        )
        B = (
            p2.l**2 * p2.m * s1.p**2
            + p1.l**2 * s2.p**2 * (p1.m + p2.m)
            - p1.l * p2.l * p2.m * s1.p * s2.p * np.cos(s1.theta - s2.theta)
        ) / (
            2
            * p1.l**2
            * p2.l**2
            * (p1.m + p2.m * np.sin(s1.theta - s2.theta) ** 2) ** 2
        )
        dp1 = -(p1.m + p2.m) * self.g * p1.l * np.sin(s1.theta) - A + B
        dp2 = -p2.m * self.g * p2.l * np.sin(s2.theta) + A - B

        return dt1, dp1, dt2, dp2


# Argumentos para a simulação
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

# Métodos de cálculo disponíveis
METHODS = {
    "eulerlagrange": EulerLagrangeSimulation,
    "hamilton": HamiltonSimulation,
}


if __name__ == "__main__":
    parser = ArgumentParser(description="Simulação de um pêndulo duplo")
    for arg in ARGS:
        flags = arg.pop("flags")
        parser.add_argument(*flags, **arg)
    args = parser.parse_args()

    simulation = METHODS[args.method](m1=args.m1, m2=args.m2, l1=args.l1, l2=args.l2)
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
