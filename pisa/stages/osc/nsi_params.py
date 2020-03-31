# author: T. Ehrhardt
# date:   Nov 8, 2018
"""
NSIParams: Characterize non-standard neutrino interaction coupling strengths

merged in by Elisa Lohfink (ellohfin; elohfink@icecube.wisc.edu)
to include NSI changes made by Thomas Ehrhardt on his branch:
original version can be found in thehrh/pisa nsi_reparameterisation branch
"""

from __future__ import absolute_import, division

import numpy as np

from pisa import CTYPE, FTYPE
from pisa.utils.comparisons import ALLCLOSE_KW, isscalar, recursiveEquality
from pisa.utils.log import Levels, set_verbosity, logging

__all__ = ['NSIParams', 'StdNSIParams', 'VacuumLikeNSIParams']


ARY2STR_KW = dict(
    precision=np.finfo(FTYPE).precision + 2,
    floatmode="fixed",
    sign=" ",
    max_line_width=200,
    separator=", ",
)


def _set_magnitude_phase(magn_phase_tuple):
    try:
        magnitude, phase = magn_phase_tuple
    except:
        raise ValueError(
            'Pass an iterable with two items (magnitude and phase)!'
        )
    if not isscalar(magnitude) or not isscalar(phase):
        raise TypeError(
            'Only scalar values for magnitude and phase accepted!'
        )
    if magnitude < 0.0 and phase != 0.0:
        raise ValueError(
            'Only accepting negative values with a zero phase (real coupling)!'
        )
    return magnitude, phase


class NSIParams(object):
    """
    Holds non-standard neutrino interaction parameters of neutral current type
    for propagating neutrinos, interacting with 1st generation Standard Model
    background quarks (u or d) or electrons in the Earth.


    Attributes
    ----------
    eps_matrix : 2d float array of shape (3, 3)
        Hermitian NSI matrix holding the effective epsilon parameters describing
        strengths of NSI transitions between the two specified neutrino flavors,
        via NC-type interaction with 1st generation quarks or electrons in the
        Earth.
        Flavour-preserving (diagonal) ones are real, while the flavour-changing
        (off-diagonal) ones are complex.
        Note that these parameters are not the Lagrangian-level couplings but
        rather the sums over these weighted by the respective relative number
        densities (approx. constant) of each possible scattering partner
        in the Earth.

    """

    def __init__(self):
        """Set NSI parameters."""
        self._eps_matrix = np.zeros((3, 3), dtype=CTYPE)


class StdNSIParams(NSIParams):
    """
    NSI parameters in the standard parameterization.

    Attributes
    ----------
    eps_matrix : 2d float array of shape (3, 3)
        Effective NSI coupling matrix.

    eps_ee, eps_emu, eps_etau, eps_mumu, eps_mutau, eps_tautau : float or complex
        Effective NSI coupling parameters.

    """

    def __init__(self):
        super().__init__()
        self._eps_ee = 0.
        self._eps_emu = 0.
        self._eps_etau = 0.
        self._eps_mumu = 0.
        self._eps_mutau = 0.
        self._eps_tautau = 0.

    # --- NSI epsilons ---
    @property
    def eps_ee(self):
        """effective nue-nue NSI coupling parameter"""
        return self.eps_matrix[0, 0].real

    @eps_ee.setter
    def eps_ee(self, value):
        if isinstance(value, complex) or not isscalar(value):
            raise TypeError("eps_ee must be a real number!")
        self._eps_matrix[0, 0] = value + 1.j * self._eps_matrix[0, 0].imag

    @property
    def eps_emu(self):
        """effective nue-numu NSI coupling parameter"""
        return self.eps_matrix[0, 1]

    @eps_emu.setter
    def eps_emu(self, value):
        magnitude, phase = _set_magnitude_phase(value)
        self._eps_matrix[0, 1] = magnitude * (np.cos(phase) + 1.j * np.sin(phase))
        self._eps_matrix[1, 0] = np.conjugate(self._eps_matrix[0, 1])

    @property
    def eps_etau(self):
        """effective nue-nutau NSI coupling parameter"""
        return self.eps_matrix[0, 2]

    @eps_etau.setter
    def eps_etau(self, value):
        magnitude, phase = _set_magnitude_phase(value)
        self._eps_matrix[0, 2] = magnitude * (np.cos(phase) + 1.j * np.sin(phase))
        self._eps_matrix[2, 0] = np.conjugate(self._eps_matrix[0, 2])

    @property
    def eps_mumu(self):
        """effective numu-numu NSI coupling parameter"""
        return self.eps_matrix[1, 1].real

    @eps_mumu.setter
    def eps_mumu(self, value):
        if isinstance(value, complex) or not isscalar(value):
            raise TypeError("eps_mumu must be a real number!")
        self._eps_matrix[1, 1] = value + 1.j * self._eps_matrix[1, 1].imag

    @property
    def eps_mutau(self):
        """effective numu-nutau NSI coupling parameter"""
        return self.eps_matrix[1, 2]

    @eps_mutau.setter
    def eps_mutau(self, value):
        magnitude, phase = _set_magnitude_phase(value)
        self._eps_matrix[1, 2] = magnitude * (np.cos(phase) + 1.j * np.sin(phase))
        self._eps_matrix[2, 1] = np.conjugate(self._eps_matrix[1, 2])

    @property
    def eps_tautau(self):
        """effective nutau-nutau NSI coupling parameter"""
        return self.eps_matrix[2, 2].real

    @eps_tautau.setter
    def eps_tautau(self, value):
        if isinstance(value, complex) or not isscalar(value):
            raise TypeError("eps_tautau must be a real number!")
        self._eps_matrix[2, 2] = value + 1.j * self._eps_matrix[2, 2].imag

    @property
    def eps_matrix(self):
        nsi_eps = self._eps_matrix
        # subtract mumu entry from diagonal entries (trace irrelevant)
        nsi_eps = nsi_eps - nsi_eps[1, 1] * np.eye(3, dtype=FTYPE)
        # explicitly nullify imaginary parts of diagonal entries which
        # are only there due to numerical inaccuracies
        for i in range(3):
            nsi_eps[i, i] = nsi_eps[i, i].real + 0 * 1.j

        # make sure this is a valid Hermitian potential matrix
        # before returning anything
        assert np.allclose(nsi_eps, nsi_eps.conj().T, **ALLCLOSE_KW)

        return nsi_eps


class VacuumLikeNSIParams(NSIParams):
    """
    NSI parameters using a vacuum Hamiltonian-like parameterization.

    """
    # pylint: disable=invalid-name
    def __init__(self):
        super().__init__()
        self._eps_scale = 1.
        self._eps_prime = 0.
        self._phi12 = 0.
        self._phi13 = 0.
        self._phi23 = 0.
        self._alpha1 = 0.
        self._alpha2 = 0.
        self._deltansi = 0.

    # --- overall matter potential strength ---
    @property
    def eps_scale(self):
        """Generalised matter potential strength scale"""
        return self._eps_scale

    @eps_scale.setter
    def eps_scale(self, value):
        if isinstance(value, complex) or not isscalar(value):
            raise TypeError("eps_scale must be a real number!")
        self._eps_scale = value

    @property
    def eps_prime(self):
        """Second Hmat eigenvalue (beside eps_scale)"""
        return self._eps_prime

    @eps_prime.setter
    def eps_prime(self, value):
        if isinstance(value, complex) or not isscalar(value):
            raise TypeError("eps_prime must be a real number!")
        self._eps_prime = value

    # --- projection phases ---
    # --- phi12 ---
    @property
    def phi12(self):
        """1-2 angle"""
        return self._phi12

    @phi12.setter
    def phi12(self, value):
        assert -np.pi <= value <= np.pi
        self._phi12 = value

    # --- phi13 ---
    @property
    def phi13(self):
        """1-3 angle"""
        return self._phi13

    @phi13.setter
    def phi13(self, value):
        assert -np.pi <= value <= np.pi
        self._phi13 = value

    # --- phi23 ---
    @property
    def phi23(self):
        """2-3 angle"""
        return self._phi23

    @phi23.setter
    def phi23(self, value):
        assert -np.pi <= value <= np.pi
        self._phi23 = value

    # --- vacuum-matter relative phases ---
    # --- alpha1 ---
    @property
    def alpha1(self):
        """1-phase"""
        return self._alpha1

    @alpha1.setter
    def alpha1(self, value):
        assert 0. <= value <= 2*np.pi
        self._alpha1 = value

    # --- alpha2 ---
    @property
    def alpha2(self):
        """2-phase"""
        return self._alpha2

    @alpha2.setter
    def alpha2(self, value):
        assert 0. <= value <= 2*np.pi
        self._alpha2 = value

    # --- nsi phase ---
    @property
    def deltansi(self):
        """NSI phase"""
        return self._deltansi

    @deltansi.setter
    def deltansi(self, value):
        assert 0. <= value <= 2*np.pi
        self._deltansi = value

    # getters for the std. coupling parameters
    # which are just the coupling matrix entries
    @property
    def eps_ee(self):
        """effective nue-nue NSI coupling parameter"""
        return self.eps_matrix[0, 0].real

    @property
    def eps_emu(self):
        """effective nue-numu NSI coupling parameter"""
        return self.eps_matrix[0, 1]

    @property
    def eps_etau(self):
        """effective nue-nutau NSI coupling parameter"""
        return self.eps_matrix[0, 2]

    @property
    def eps_mumu(self):
        """effective numu-numu NSI coupling parameter"""
        return self.eps_matrix[1, 1].real

    @property
    def eps_mutau(self):
        """effective numu-nutau NSI coupling parameter"""
        return self.eps_matrix[1, 2]

    @property
    def eps_tautau(self):
        """effective nutau-nutau NSI coupling parameter"""
        return self.eps_matrix[2, 2].real


    @property
    def eps_matrix(self):
        """Effective NSI coupling matrix."""
        # numerical calculation for now...
        # relative matter-nsi phases
        Qrel = (
            np.array([
                complex(np.cos(self.alpha1), np.sin(self.alpha1)),
                complex(np.cos(self.alpha2), np.sin(self.alpha2)),
                complex(np.cos(-(self.alpha1 + self.alpha2)), np.sin(-(self.alpha1 + self.alpha2)))
            ]) * np.eye(3, dtype=FTYPE)
        )
        # rotation matrices (signs as for PMNS matrix,
        # also reproduce NSI global fit paper relations)
        R12 = np.array(
            [[np.cos(self.phi12), np.sin(self.phi12), 0],
            [-np.sin(self.phi12), np.cos(self.phi12), 0],
            [0, 0, 1]],
            dtype=FTYPE
        )
        R13 = np.array(
            [[np.cos(self.phi13), 0, np.sin(self.phi13)],
            [0, 1, 0],
            [-np.sin(self.phi13), 0, np.cos(self.phi13)]],
            dtype=FTYPE
        )
        R23_complex = np.array(
            [[1, 0, 0],
            [0, np.cos(self.phi23), np.sin(self.phi23) * complex(np.cos(-self.deltansi), np.sin(-self.deltansi))],
            [0, -np.sin(self.phi23) * complex(np.cos(self.deltansi), np.sin(self.deltansi)), np.cos(self.phi23)]],
        )
        # "matter mixing matrix"
        Umat = np.matmul(R12, np.matmul(R13, R23_complex))
        # Hmat eigenvalues
        Dmat = np.array([self.eps_scale, self.eps_prime, 0], dtype=FTYPE) * np.eye(3, dtype=FTYPE)
        # start from the innermost product, work your way outwards
        mat_pot = np.matmul(
            Qrel,
            np.matmul(Umat,
                np.matmul(Dmat,
                    np.matmul(Umat.conj().T, Qrel.conj().T)
                )
            )
        )
        # subtract mumu entry from diagonal entries (trace irrelevant)
        mat_pot = mat_pot - mat_pot[1, 1] * np.eye(3, dtype=FTYPE)
        # subtract standard matter potential entry for CC coherent
        # forward-scattering
        mat_pot[0, 0] = mat_pot[0, 0] - 1.
        # this is now the actual nsi coupling matrix
        nsi_eps = mat_pot
        # explicitly nullify imaginary parts of diagonal entries which
        # are only there due to numerical inaccuracies
        for i in range(3):
            nsi_eps[i, i] = nsi_eps[i, i].real + 0 * 1.j

        # make sure this is a valid Hermitian potential matrix
        # before returning anything
        assert np.allclose(nsi_eps, nsi_eps.conj().T, **ALLCLOSE_KW)

        return nsi_eps

    @property
    def eps_matrix_analytical(self):
        """Effective NSI coupling matrix calculated analytically."""
        # Analytical relations. These are wrong right now! #FIXME
        nsi_eps = np.zeros((3, 3, 2), dtype=FTYPE)

        sp12 = np.sin(self.phi12)
        sp13 = np.sin(self.phi13)
        sp23 = np.sin(self.phi23)
        cp12 = np.sqrt(1. - sp12**2)
        cp13 = np.sqrt(1. - sp13**2)
        cp23 = np.sqrt(1. - sp23**2)

        sdnsi = np.sin(self.deltansi)
        cdnsi = np.cos(self.deltansi)

        # eps_ee - eps_mumu (real)
        nsi_eps[0, 0, 0] = (
            self.eps_scale * cp13**2 * (cp12**2 - sp12**2) +
            self.eps_prime * (
                (cp12**2 - sp12**2) * (sp13**2 * sp23**2 - cp23**2) -
                4 * cp12 * sp12 * sp13 * cp23 * sp23 * cdnsi
            )
        ) - 1
        nsi_eps[0, 0, 1] = 0.
        # eps_emu (complex)
        nsi_eps[0, 1, 0] = (
            self.eps_scale * cp12 * sp12 * cp13**2 * np.cos(self.alpha1 - self.alpha2) +
            self.eps_prime * (
                (
                    cp12 * sp12 * (sp13**2 * sp23**2 - cp23**2) +
                    sp13 * cp23 * sp23 * cdnsi * (cp12**2 - sp12**2)
                ) * np.cos(self.alpha1 - self.alpha2) -
                (
                    sp13 * cp23 * sp23 * sdnsi
                ) * np.sin(self.alpha1 - self.alpha2)
            )
        )
        nsi_eps[0, 1, 1] = (
            self.eps_scale * cp12 * sp12 * cp13**2 * np.sin(self.alpha1 - self.alpha2) +
            self.eps_prime * (
                (
                    cp12 * sp12 * (sp13**2 * sp23**2 - cp23**2) +
                    sp13 * cp23* sp23 * cdnsi * (cp12**2 - sp12**2)
                ) * np.sin(self.alpha1 - self.alpha2) +
                (
                    sp13 * cp23 * sp23 * sdnsi
                ) * np.cos(self.alpha1 - self.alpha2)
            )
        )
        # eps_etau (complex)
        nsi_eps[0, 2, 0] = (
            -self.eps_scale * cp12 * sp13 * cp13 * np.cos(2 * self.alpha1 + self.alpha2) +
            self.eps_prime * (
                (
                    cp13 * sp23 * (cp12 * sp13 * sp23 - sp12 * cp23 * cdnsi)
                ) * np.cos(2 * self.alpha1  + self.alpha2) -
                (
                    cp13 * sp12 * cp23 * sp23 * sdnsi
                ) * np.sin(2 * self.alpha1 + self.alpha2)
            )
        )
        nsi_eps[0, 2, 1] = (
            -self.eps_scale * cp12* sp13 * cp13 * np.sin(2 * self.alpha1 + self.alpha2) +
            self.eps_prime * (
                (
                    cp13 * sp23 * (cp12 * sp13 * sp23 - sp12 * cp23 * cdnsi)
                ) * np.sin(2 * self.alpha1 + self.alpha2) +
                (
                    cp13 * sp23 * sp12 * cp23 * sdnsi
                ) * np.cos(2 * self.alpha1 + self.alpha2)
            )
        )
        # eps_emu* (complex)
        nsi_eps[1, 0, 0] = nsi_eps[0, 1, 0]
        nsi_eps[1, 0, 1] = -nsi_eps[0, 1, 1]
        # eps_etau* (complex)
        nsi_eps[2, 0, 0] = nsi_eps[0, 2, 0]
        nsi_eps[2, 0, 1] = -nsi_eps[0, 2, 1]
        # eps_mumu - eps_mumu (0 by definition)
        nsi_eps[1, 1, 0] = 0.
        nsi_eps[1, 1, 1] = 0.
        # eps_mutau (complex)
        nsi_eps[1, 2, 0] = (
            -self.eps_scale * sp12 * cp13 * sp13 * np.cos(self.alpha1 + 2 * self.alpha2) +
            self.eps_prime * (
                (
                    cp13 * sp23 * (sp12 * sp13 * sp23 + cp12 * cp23 * cdnsi)
                ) * np.cos(self.alpha1 + 2 * self.alpha2) +
                (
                    cp12 * cp13 * cp23 * sp23 * sdnsi
                ) * np.sin(self.alpha1 + 2 * self.alpha2)
            )
        )
        nsi_eps[1, 2, 1] = (
            -self.eps_scale * sp12 * cp13 * sp13 * np.sin(self.alpha1 + 2 * self.alpha2) +
            self.eps_prime * (
                (
                    -cp12 * cp13 * cp23 * sp23 * sdnsi
                ) * np.cos(self.alpha1 + 2 * self.alpha2) +
                (
                    cp13 * sp23 * (sp12 * sp13 * sp23 + cp12 * cp23 * cdnsi)
                ) * np.sin(self.alpha1 + 2 * self.alpha2)
            )
        )
        # eps_mutau* (complex)
        nsi_eps[2, 1, 0] = nsi_eps[1, 2, 0]
        nsi_eps[2, 1, 1] = -nsi_eps[1, 2, 1]
        # eps_tautau - eps_mumu (real)
        nsi_eps[2, 2, 0] = (
            self.eps_scale * (sp13**2 - cp13**2 * sp12**2) +
            self.eps_prime *(
                sp23**2 * (cp13**2 - sp12**2 * sp13**2) -
                2 * cp12 * sp12 * sp13 * cp23 * sp23 * cdnsi -
                cp12**2 * cp23**2
            )
        )
        nsi_eps[2, 2, 1] = 0.

        # make this into a complex 2d array
        nsi_eps = nsi_eps[:, :, 0] + nsi_eps[:, :, 1] * 1.j
        # make sure this is a valid Hermitian potential matrix
        # before returning anything
        assert np.allclose(nsi_eps, nsi_eps.conj().T, **ALLCLOSE_KW)

        return nsi_eps

def test_nsi_params():
    """Unit tests for subclasses of `NSIParams`."""
    # TODO: these have to be extended
    rand = np.random.RandomState(0)
    std_nsi = StdNSIParams()
    try:
        # cannot accept a sequence
        std_nsi.eps_ee = [rand.rand()]
    except TypeError:
        pass

    try:
        # must be real
        std_nsi.eps_ee = rand.rand() * 1.j
    except TypeError:
        pass

    try:
        # unphysical negative magnitude for nonzero phase
        std_nsi.eps_mutau = ((rand.rand() - 1.0), 0.1)
    except ValueError:
        pass

    std_nsi.eps_ee = 0.5
    std_nsi.eps_mumu = 0.5
    std_nsi.eps_tautau = 0.5
    if not np.allclose(
        std_nsi.eps_matrix, np.zeros((3, 3), dtype=CTYPE), **ALLCLOSE_KW
    ):
        raise ValueError("NSI coupling matrix should be identically zero!")

    vac_like_nsi = VacuumLikeNSIParams()
    vac_like_nsi.eps_scale = rand.rand() * 10.
    assert recursiveEquality(vac_like_nsi.eps_ee, vac_like_nsi.eps_scale - 1.0)

def test_nsi_parameterization():
    """Unit test for Hvac-like NSI parameterization."""
    rand = np.random.RandomState(0)
    alpha1, alpha2, deltansi = rand.rand(3) * 2. * np.pi
    phi12, phi13, phi23 = rand.rand(3) * 2*np.pi - np.pi
    eps_max_abs = 10.0
    eps_scale, eps_prime = rand.rand(2) * 2 * eps_max_abs - eps_max_abs
    nsi_params = VacuumLikeNSIParams()
    nsi_params.eps_scale = eps_scale
    nsi_params.eps_prime = eps_prime
    nsi_params.phi12 = phi12
    nsi_params.phi13 = phi13
    nsi_params.phi23 = phi23
    nsi_params.alpha1 = alpha1
    nsi_params.alpha2 = alpha2
    nsi_params.deltansi = deltansi

    logging.trace('Checking agreement between numerical & analytical NSI matrix...')

    eps_mat_numerical = nsi_params.eps_matrix
    eps_mat_analytical = nsi_params.eps_matrix_analytical

    try:
        close = np.isclose(eps_mat_numerical, eps_mat_analytical, **ALLCLOSE_KW)
        if not np.all(close):
            logging.debug(
                "Numerical NSI matrix:\n%s",
                np.array2string(eps_mat_numerical, **ARY2STR_KW)
            )
            logging.debug(
                "Analytical expansion (by hand):\n%s",
                np.array2string(eps_mat_analytical, **ARY2STR_KW)
            )
            raise ValueError(
                'Evaluating analytical expressions for NSI matrix elements'
                ' does not give agreement with numerical calculation!'
                ' Elementwise agreement:\n%s'
                % close
            )
    except ValueError as err:
        logging.warning(
            "%s\nThis is expected."
            " Going ahead with numerical calculation for now.", err
        )

    logging.trace('Now checking agreement with sympy calculation...')

    eps_mat_sympy = nsi_sympy_mat_mult(
        eps_scale_val=eps_scale,
        eps_prime_val=eps_prime,
        phi12_val=phi12,
        phi13_val=phi13,
        phi23_val=phi23,
        alpha1_val=alpha1,
        alpha2_val=alpha2,
        deltansi_val=deltansi
    )

    logging.trace('ALLCLOSE_KW = {}'.format(ALLCLOSE_KW))
    close = np.isclose(eps_mat_numerical, eps_mat_sympy, **ALLCLOSE_KW)
    if not np.all(close):
        logging.error(
            'Numerical NSI matrix:\n%s',
            np.array2string(eps_mat_numerical, **ARY2STR_KW)
        )
        logging.error(
            'Sympy NSI matrix:\n%s', np.array2string(eps_mat_sympy, **ARY2STR_KW)
        )
        raise ValueError(
            'Sympy and numerical calculations disagree! Elementwise agreement:\n'
            '%s' % close
        )

def nsi_sympy_mat_mult(
    eps_scale_val,
    eps_prime_val,
    phi12_val,
    phi13_val,
    phi23_val,
    alpha1_val,
    alpha2_val,
    deltansi_val,
):
    """Sympy calculation of generalised matter Hamiltonian."""
    # pylint: disable=invalid-name
    from sympy import (
        cos, sin,
        Matrix, eye,
        I, re, im,
        Symbol, symbols,
        simplify,
        init_printing
    )
    from sympy.physics.quantum.dagger import Dagger
    init_printing(use_unicode=True)
    phi12, phi13, phi23 = symbols('phi12 phi13 phi23', real=True)
    alpha1, alpha2 = symbols('alpha1 alpha2', real=True)
    eps_scale, eps_prime = symbols('eps_scale eps_prime', real=True)
    deltansi = Symbol('deltansi', real=True)

    Dmat = Matrix(
        [[eps_scale, 0, 0], [0, eps_prime, 0], [0, 0, 0]]
    )
    Qrel = Matrix(
        [[cos(alpha1) + I * sin(alpha1), 0, 0],
        [0, cos(alpha2) + I * sin(alpha2), 0],
        [0, 0, cos(-(alpha1 + alpha2)) + I * sin(-(alpha1 + alpha2))]]
    )
    R12 = Matrix(
        [[cos(phi12), sin(phi12), 0],
        [-sin(phi12), cos(phi12), 0],
        [0, 0, 1]]
    )
    R13 = Matrix(
        [[cos(phi13), 0, sin(phi13)],
        [0, 1, 0],
        [-sin(phi13), 0, cos(phi13)]]
    )
    R23_complex = Matrix(
        [[1, 0, 0],
        [0, cos(phi23), sin(phi23) * (cos(deltansi) + I * sin(-deltansi))],
        [0, -sin(phi23) * (cos(deltansi) + I * sin(deltansi)), cos(phi23)]]
    )

    Umat = R12 * R13 * R23_complex
    tmp = Dagger(Umat) * Dagger(Qrel)
    tmp2 = Dmat * tmp
    tmp3 = Umat * tmp2
    Hmat_sympy = Qrel * tmp3
    # subtract constant * id
    Hmat_sympy_minus_mumu = Hmat_sympy - Hmat_sympy[1, 1] * eye(3)
    Hmat_sympy_minus_mumu[0, 0] = Hmat_sympy_minus_mumu[0, 0] - 1
    eps_mat_sympy = Hmat_sympy_minus_mumu
    # simplify
    eps_mat_sympy_simpl = simplify(eps_mat_sympy)
    # evaluate
    eps_mat_sympy_eval = eps_mat_sympy_simpl.subs(
        [(eps_scale, eps_scale_val), (eps_prime, eps_prime_val),
         (phi12, phi12_val), (phi13, phi13_val), (phi23, phi23_val),
         (alpha1, alpha1_val), (alpha2, alpha2_val),
         (deltansi, deltansi_val)]
    )
    # real part
    eps_mat_sympy_eval_re = re(eps_mat_sympy_eval)
    # imaginary part
    eps_mat_sympy_eval_im = im(eps_mat_sympy_eval)

    # complex numpy array
    return (
        np.array(eps_mat_sympy_eval_re) + np.array(eps_mat_sympy_eval_im) * 1.j
    ).astype(CTYPE)


if __name__ == '__main__':
    set_verbosity(Levels.INFO)
    test_nsi_params()
    test_nsi_parameterization()
