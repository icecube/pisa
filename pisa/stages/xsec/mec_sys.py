"""
Systematic uncertainty model for Meson Exchange Current (MEC) events.
"""

from pisa import ureg
from pisa.core.param import Param, ParamSet
from pisa.core.stage import Stage
from pisa.utils.profiler import profile

__all__ = ["mec_sys", "init_test"]


class mec_sys(Stage):  # pylint: disable=invalid-name
    """
    Stage to apply MEC event systematic uncertainties

    This is currently just a normalization parameter.

    Parameters
    ----------

    params : ParamSet
        Must have parameters::
            mec_scale : quantity (dimensionless)
                A normalization to be applied to all MEC events

    Notes
    -----

    Expected container keys are::

        "mec" - a boolean flag indicating if the event is an MEC interaction
        "weights" - the event weight that will be modified by this stage
    """

    def __init__(
        self,
        **std_kwargs,
    ):
        expected_params = ("mec_scale",)

        expected_container_keys = (
            "weights",
            "mec",
        )

        # init base class
        super().__init__(
            expected_params=expected_params,
            expected_container_keys=expected_container_keys,
            **std_kwargs,
        )

    @profile
    def apply_function(self):

        # Get the normalization parameter, and use it to scale the weights of all events that
        # are marked by the "mec" flag.

        mec_scale = self.params.mec_scale.m_as("dimensionless")

        for container in self.data:
            mec = container["mec"].astype(bool)
            if mec.sum() > 0:
                container["weights"][mec] *= mec_scale
                container.mark_changed("weights")


def init_test(**param_kwargs):
    """Instantiation example"""
    param_set = ParamSet([Param(name="mec_scale", value=1.0, **param_kwargs)])

    return mec_sys(params=param_set)
