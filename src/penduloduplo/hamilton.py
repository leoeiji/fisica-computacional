from typing import Tuple

import numpy as np
from simulation import Pendulum, PendulumState, RK2Simulation


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
