"""
Standard RB analysis class.
"""
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple, Union, Optional, TYPE_CHECKING

import lmfit
from qiskit.exceptions import QiskitError

import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.exceptions import AnalysisError
from qiskit_experiments.framework import AnalysisResultData, ExperimentData
from qiskit_experiments.framework.analysis_result import AnalysisResult

if TYPE_CHECKING:
    from uncertainties import UFloat

# A dictionary key of qubit aware quantum instruction; type alias for better readability
QubitGateTuple = Tuple[Tuple[int, ...], str]


class SimplerSingleQubitRBAnalysis(curve.CurveAnalysis):
    r"""A class to analyze randomized benchmarking experiments.

    # section: overview
        This analysis takes only single series.
        This series is fit by the exponential decay function.
        From the fit :math:`\alpha` value this analysis estimates the error per Clifford (EPC).

        When analysis option ``gate_error_ratio`` is provided, this analysis also estimates
        errors of individual gates assembling a Clifford gate.
        In computation of two-qubit EPC, this analysis can also decompose
        the contribution from the underlying single qubit depolarizing channels when
        ``epg_1_qubit`` analysis option is provided [1].

    # section: fit_model
        .. math::

            F(x) = a \alpha^x + b

    # section: fit_parameters
        defpar a:
            desc: Height of decay curve.
            init_guess: Determined by :math:`1 - b`.
            bounds: [0, 1]
        defpar b:
            desc: Base line.
            init_guess: Determined by :math:`(1/2)^n` where :math:`n` is number of qubit.
            bounds: [0, 1]
        defpar \alpha:
            desc: Depolarizing parameter.
            init_guess: Determined by :func:`~.guess.rb_decay`.
            bounds: [0, 1]

    # section: reference
        .. ref_arxiv:: 1 1712.06550

    """

    def __init__(self):
        super().__init__(
            models=[
                lmfit.models.ExpressionModel(
                    expr="a * alpha ** x + b",
                    name="rb_decay",
                )
            ]
        )
    

    @classmethod
    def _default_options(cls):
        """Default analysis options.

        Analysis Options:
            gate_error_ratio (Optional[Dict[str, float]]): A dictionary with gate name keys
                and error ratio values used when calculating EPG from the estimated EPC.
                The default value will use standard gate error ratios.
                If you don't know accurate error ratio between your basis gates,
                you can skip analysis of EPGs by setting this options to ``None``.
            epg_1_qubit (List[AnalysisResult]): Analysis results from previous RB experiments
                for individual single qubit gates. If this is provided, EPC of
                2Q RB is corrected to exclude the depolarization of underlying 1Q channels.
        """
        default_options = super()._default_options()
        default_options.plotter.set_figure_options(
            xlabel="Clifford Length",
            ylabel="P(0)",
        )
        default_options.plot_raw_data = True
        default_options.result_parameters = ["alpha"]
        default_options.average_method = "sample"
        default_options.outcome = "0"

        return default_options

    def _generate_fit_guesses(
            self,
            user_opt: curve.FitOptions,
            curve_data: curve.CurveData,
        ) -> Union[curve.FitOptions, List[curve.FitOptions]]:
            """Create algorithmic initial fit guess from analysis options and curve data.

            Args:
                user_opt: Fit options filled with user provided guess and bounds.
                curve_data: Formatted data collection to fit.

            Returns:
                List of fit options that are passed to the fitter function.
            """
            user_opt.bounds.set_if_empty(
                a=(0, 1),
                alpha=(0, 1),
                b=(0, 1),
            )

            b_guess = 1 / 2
            # TODO: use SUBSET of the data. (use get_subset method)
            # use the not-flipped subset to get the alpha and a guess.
            # ASSUME that we can reuse the guess for the other subset. 
            alpha_guess = curve.guess.rb_decay(curve_data.x, curve_data.y, b=b_guess)
            a_guess = (curve_data.y[0] - b_guess) / (alpha_guess ** curve_data.x[0])

            user_opt.p0.set_if_empty(  # 
                b=b_guess,
                a=a_guess,
                # a_flip = dsfasdf
                alpha=alpha_guess,
            )

            return user_opt
    
    def _create_analysis_results(
        self,
        fit_data: curve.CurveFitResult,
        quality: str,
        **metadata,
    ) -> List[AnalysisResultData]:
        """Create analysis results for important fit parameters.

        Args:
            fit_data: Fit outcome.
            quality: Quality of fit outcome.

        Returns:
            List of analysis result data.
        """
        outcomes = super()._create_analysis_results(fit_data, quality, **metadata)
        num_qubits = 1

        # Calculate EPC
        alpha = fit_data.ufloat_params["alpha"]
        scale = (2**num_qubits - 1) / (2**num_qubits)
        epc = scale * (1 - alpha)

        outcomes.append(
            AnalysisResultData(
                name="EPC",
                value=epc,
                chisq=fit_data.reduced_chisq,
                quality=quality,
                extra=metadata,
            )
        )

        return outcomes  # outcome of the measurement