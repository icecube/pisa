"""
Utilities for performing some not-so-common matrix tasks.
"""

import numpy as np
import scipy.linalg as lin

__all__ = [
    'is_psd',
    'fronebius_nearest_psd',
]

__author__ = 'A. Trettin'

__license__ = '''Copyright (c) 2020, The IceCube Collaboration

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.'''

def is_psd(A):
    """Test whether a matrix is positive semi-definite.

    Test is done via attempted Cholesky decomposition as suggested in [1]_.

    Parameters
    ----------
    A : numpy.ndarray
        Symmetric matrix
    
    Returns
    -------
    bool
        True if `A` is positive semi-definite, else False

    References
    ----------
    ..  [1] N.J. Higham, "Computing a nearest symmetric positive semidefinite
        matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """
    # pylint: disable=invalid-name
    try:
        _ = np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False

def fronebius_nearest_psd(A, return_distance=False):
    """Find the positive semi-definite matrix closest to `A`.

    The closeness to `A` is measured by the Fronebius norm. The matrix closest to `A`
    by that measure is uniquely defined in [3]_.

    Parameters
    ----------
    A : numpy.ndarray
        Symmetric matrix
    return_distance : bool, optional
        Return distance of the input matrix to the approximation as given in
        theorem 2.1 in [3]_.
        This can be compared to the actual Frobenius norm between the
        input and output to verify the calculation.

    Returns
    -------
    X : numpy.ndarray
        Positive semi-definite matrix approximating `A`.

    Notes
    -----
    This function is a modification of [1]_, which is a Python adaption of [2]_, which
    credits [3]_.

    References
    ----------
    ..  [1] https://gist.github.com/fasiha/fdb5cec2054e6f1c6ae35476045a0bbd
    ..  [2] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    ..  [3] N.J. Higham, "Computing a nearest symmetric positive semidefinite
        matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """
    # pylint: disable=invalid-name
    assert A.ndim == 2, "input is not a 2D matrix"
    B = (A + A.T)/2.
    _, H = lin.polar(B)
    X = (B + H)/2.
    # small numerical errors can make matrices that are not exactly
    # symmetric, fix that
    X = (X + X.T)/2.
    # due to numerics, it's possible that the matrix is _still_ not psd.
    # We can fix that iteratively by adding small increments of the identity matrix.
    # This part comes from [1].
    if not is_psd(X):
        spacing = np.spacing(lin.norm(X))
        I = np.eye(X.shape[0])
        k = 1
        while not is_psd(X):
            mineig = np.min(np.real(lin.eigvals(X)))
            X += I * (-mineig * k**2 + spacing)
            k += 1
    if return_distance:
        C = (A - A.T)/2.
        lam = lin.eigvalsh(B)
        # pylint doesn't know that numpy.sum takes the "where" argument
        # pylint: disable=unexpected-keyword-arg
        dist = np.sqrt(np.sum(lam**2, where=lam < 0.) + lin.norm(C, ord='fro')**2)
        return X, dist
    return X

def test_frob_psd(A):
    """Test approximation of Frobenius-closest PSD on given matrix
    
    Parameters
    ----------
    A : numpy.ndarray
        Symmetric matrix
    """
    # pylint: disable=invalid-name
    X, xdist = fronebius_nearest_psd(A, return_distance=True)
    is_psd_after = is_psd(X)
    actual_dist = lin.norm(A - X, ord='fro')
    assert is_psd_after, "did not produce PSD matrix"
    assert np.isclose(xdist, actual_dist), "actual distance differs from expectation"

if __name__ == '__main__':
    m_test = np.array([[1, -1], [2, 4]])
    test_frob_psd(m_test)
    print('matrix before:')
    print(m_test)
    print('matrix after:')
    print(fronebius_nearest_psd(m_test))
    print('The result matrix is psd and the Frobenius norm matches the expectation!')
    print('testing random matrices...')
    for i in range(100):
        m_test = np.random.randn(3, 3)
        test_frob_psd(m_test)
    print('Test passed!')
