from typing import Tuple

import numpy as np
from simulation import Pendulum, PendulumState, RK2Simulation


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
