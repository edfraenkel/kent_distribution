#!/usr/bin/env python
"""
The algorithms here are partially based on methods described in:
[The Fisher-Bingham Distribution on the Sphere, John T. Kent
Journal of the Royal Statistical Society. Series B (Methodological)
Vol. 44, No. 1 (1982), pp. 71-80 Published by: Wiley
Article Stable URL: http://www.jstor.org/stable/2984712]

The example code in kent_example.py serves not only as an example
but also as a test. It performs some higher level tests but it also 
generates example plots if called directly from the shell.
"""

from numpy import *
from scipy.optimize import fmin_bfgs
from scipy.special import gamma as gamma_fun
from scipy.special import iv as modified_bessel_2ndkind
from scipy.special import ivp as modified_bessel_2ndkind_derivative
from scipy.stats import uniform
# to avoid confusion with the norm of a vector we give the normal distribution a less confusing name here
from scipy.stats import norm as gauss 
from scipy.linalg import eig
import sys
import warnings


#helper function
def MMul(A, B):
  return inner(A, transpose(B))

#helper function to compute the L2 norm. scipy.linalg.norm is not used because this function does not allow to choose an axis
def norm(x, axis=None):
  if isinstance(x, list) or isinstance(x, tuple):
    x = array(x)
  return sqrt(sum(x*x, axis=axis))
  
def kent(theta, phi, psi, kappa, beta):
  """
  Generates the Kent distribution based on the spherical coordinates theta, phi, psi
  with the concentration parameter kappa and the ovalness beta
  """
  gamma1, gamma2, gamma3 = KentDistribution.spherical_coordinates_to_gammas(theta, phi, psi)
  k = KentDistribution(gamma1, gamma2, gamma3, kappa, beta)
  return k

def kent2(gamma1, gamma2, gamma3, kappa, beta):
  """
  Generates the Kent distribution using the orthonormal vectors gamma1, 
  gamma2 and gamma3, with the concentration parameter kappa and the ovalness beta
  """
  assert abs(inner(gamma1, gamma2)) < 1E-10
  assert abs(inner(gamma2, gamma3)) < 1E-10
  assert abs(inner(gamma3, gamma1)) < 1E-10
  return KentDistribution(gamma1, gamma2, gamma3, kappa, beta)
    
def kent3(A, B):
  """
  Generates the Kent distribution using the orthogonal vectors A and B
  where A = gamma1*kappa and B = gamma2*beta (gamma3 is inferred)
  A may have not have length zero but may be arbitrarily close to zero
  B may have length zero however. If so, then an arbitrary value for gamma2
  (orthogonal to gamma1) is chosen
  """
  kappa = norm(A)
  beta = norm(B)
  gamma1 = A/kappa
  if beta == 0.0:
    gamma2 = __generate_arbitrary_orthogonal_unit_vector(gamma1)
  else:
    gamma2 = B/beta
  theta, phi, psi = KentDistribution.gammas_to_spherical_coordinates(gamma1, gamma2)
  gamma1, gamma2, gamma3 = KentDistribution.spherical_coordinates_to_gammas(theta, phi, psi)
  return KentDistribution(gamma1, gamma2, gamma3, kappa, beta)
  
def kent4(Gamma, kappa, beta):
  """
  Generates the kent distribution
  """
  gamma1 = Gamma[:,0]
  gamma2 = Gamma[:,1]
  gamma3 = Gamma[:,2]
  return kent2(gamma1, gamma2, gamma3, kappa, beta)

def __generate_arbitrary_orthogonal_unit_vector(x):
  v1 = cross(x, array([1.0, 0.0, 0.0]))
  v2 = cross(x, array([0.0, 1.0, 0.0]))
  v3 = cross(x, array([0.0, 0.0, 1.0]))
  v1n = norm(v1)
  v2n = norm(v2)
  v3n = norm(v3)
  v = [v1, v2, v3][argmax([v1n, v2n, v3n])]
  return v/norm(v)
  
class KentDistribution(object):
  minimum_value_for_kappa = 1E-6
  @staticmethod
  def create_matrix_H(theta, phi):
    return array([
      [cos(theta),          -sin(theta),         0.0      ],
      [sin(theta)*cos(phi), cos(theta)*cos(phi), -sin(phi)],
      [sin(theta)*sin(phi), cos(theta)*sin(phi), cos(phi) ]
    ])

  @staticmethod
  def create_matrix_Ht(theta, phi):
    return transpose(KentDistribution.create_matrix_H(theta, phi))

  @staticmethod
  def create_matrix_K(psi):
    return array([
      [1.0, 0.0,      0.0      ],
      [0.0, cos(psi), -sin(psi)],
      [0.0, sin(psi), cos(psi) ]
    ])  
  
  @staticmethod
  def create_matrix_Kt(psi):
    return transpose(KentDistribution.create_matrix_K(psi)) 
  
  @staticmethod
  def create_matrix_Gamma(theta, phi, psi):
    H = KentDistribution.create_matrix_H(theta, phi)
    K = KentDistribution.create_matrix_K(psi)
    return MMul(H, K)
  
  @staticmethod
  def create_matrix_Gammat(theta, phi, psi):
    return transpose(KentDistribution.create_matrix_Gamma(theta, phi, psi))
  
  @staticmethod
  def spherical_coordinates_to_gammas(theta, phi, psi):
    Gamma = KentDistribution.create_matrix_Gamma(theta, phi, psi)
    gamma1 = Gamma[:,0]
    gamma2 = Gamma[:,1]
    gamma3 = Gamma[:,2]    
    return gamma1, gamma2, gamma3

  @staticmethod
  def gamma1_to_spherical_coordinates(gamma1):
    theta = arccos(gamma1[0])
    phi = arctan2(gamma1[2], gamma1[1])
    return theta, phi

  @staticmethod
  def gammas_to_spherical_coordinates(gamma1, gamma2):
    theta, phi = KentDistribution.gamma1_to_spherical_coordinates(gamma1)
    Ht = KentDistribution.create_matrix_Ht(theta, phi)
    u = MMul(Ht, reshape(gamma2, (3, 1)))
    psi = arctan2(u[2][0], u[1][0])
    return theta, phi, psi

  
  def __init__(self, gamma1, gamma2, gamma3, kappa, beta):
    self.gamma1 = array(gamma1, dtype=float64)
    self.gamma2 = array(gamma2, dtype=float64)
    self.gamma3 = array(gamma3, dtype=float64)
    self.kappa = float(kappa)
    self.beta = float(beta)
    
    self.theta, self.phi, self.psi = KentDistribution.gammas_to_spherical_coordinates(self.gamma1, self.gamma2)
    
    for gamma in gamma1, gamma2, gamma3:
      assert len(gamma) == 3

    self._cached_rvs = array([], dtype=float64)
    self._cached_rvs.shape = (0, 3)
  
  @property
  def Gamma(self):
    return self.create_matrix_Gamma(self.theta, self.phi, self.psi)
  
  def normalize(self, cache=dict(), return_num_iterations=False):
    """
    Returns the normalization constant of the Kent distribution.
    The proportional error may be expected not to be greater than 
    1E-11.
    
    >>> gamma1 = array([1.0, 0.0, 0.0])
    >>> gamma2 = array([0.0, 1.0, 0.0])
    >>> gamma3 = array([0.0, 0.0, 1.0])
    >>> tiny = KentDistribution.minimum_value_for_kappa
    >>> abs(kent2(gamma1, gamma2, gamma3, tiny, 0.0).normalize() - 4*pi) < 4*pi*1E-12
    True
    >>> for kappa in [0.01, 0.1, 0.2, 0.5, 2, 4, 8, 16]:
    ...     print abs(kent2(gamma1, gamma2, gamma3, kappa, 0.0).normalize() - 4*pi*sinh(kappa)/kappa) < 1E-15*4*pi*sinh(kappa)/kappa,
    ... 
    True True True True True True True True
    """
    k, b = self.kappa, self.beta
    if not (k, b) in cache:
      G = gamma_fun
      I = modified_bessel_2ndkind
      result = 0.0
      j = 0
      if b == 0.0:
        result = (
          ( (0.5*k)**(-2*j-0.5) )*
          ( I(2*j+0.5, k) )
        )
        result /= G(j+1)
        result *= G(j+0.5)
        
      else:
        while True:
          a = (
            exp(
              log(b)*2*j +
              log(0.5*k)*(-2*j-0.5)
            )*I(2*j+0.5, k)
          )
          a /= G(j+1)
          a *= G(j+0.5)
          result += a
          
          j += 1
          if abs(a) < abs(result)*1E-12 and j > 5:
            break
        
        # print "c", result, "(", k, b
      
      cache[k, b] = 2*pi*result
    if return_num_iterations:
      return cache[k, b], j
    else:
      return cache[k, b]

  def log_normalize(self, return_num_iterations=False):
    """
    Returns the logarithm of the normalization constant.
    """
    if return_num_iterations:
      normalize, num_iter = self.normalize(return_num_iterations=True)
      return log(normalize), num_iter
    else:
      return log(self.normalize())
      
  
  def pdf_max(self, normalize=True):
    return exp(self.log_pdf_max(normalize))

  def log_pdf_max(self, normalize=True):
    """
    Returns the maximum value of the log(pdf)
    """
    if self.beta == 0.0:
      x = 1
    else:
      x = self.kappa*1.0/(2*self.beta)
    if x > 1.0:
      x = 1
    fmax = self.kappa*x + self.beta*(1-x**2)
    if normalize:
      return fmax - self.log_normalize()
    else:
      return fmax
    
  def pdf(self, xs, normalize=True):
    """
    Returns the pdf of the kent distribution for 3D vectors that
    are stored in xs which must be an array of N x 3 or N x M x 3
    N x M x P x 3 etc.
    
    The code below shows how points in the pdf can be evaluated. An integral is
    calculated using random points on the sphere to determine wether the pdf is
    properly normalized.
    
    >>> from numpy.random import seed
    >>> from scipy.stats import norm as gauss
    >>> seed(666)
    >>> num_samples = 400000
    >>> xs = gauss(0, 1).rvs((num_samples, 3))
    >>> xs = divide(xs, reshape(norm(xs, 1), (num_samples, 1)))
    >>> assert abs(4*pi*average(kent(1.0, 1.0, 1.0, 4.0,  2.0).pdf(xs)) - 1.0) < 0.01
    >>> assert abs(4*pi*average(kent(1.0, 2.0, 3.0, 4.0,  2.0).pdf(xs)) - 1.0) < 0.01
    >>> assert abs(4*pi*average(kent(1.0, 2.0, 3.0, 4.0,  8.0).pdf(xs)) - 1.0) < 0.01
    >>> assert abs(4*pi*average(kent(1.0, 2.0, 3.0, 16.0, 8.0).pdf(xs)) - 1.0) < 0.01
    """
    return exp(self.log_pdf(xs, normalize))
  
  
  def log_pdf(self, xs, normalize=True):
    """
    Returns the log(pdf) of the kent distribution.
    """
    axis = len(shape(xs))-1
    g1x = sum(self.gamma1*xs, axis)
    g2x = sum(self.gamma2*xs, axis)
    g3x = sum(self.gamma3*xs, axis)
    k, b = self.kappa, self.beta

    f = k*g1x + b*(g2x**2 - g3x**2)
    if normalize:
      return f - self.log_normalize()
    else:
      return f
      
  def pdf_prime(self, xs, normalize=True):
    """
    Returns the derivative of the pdf with respect to kappa and beta. 
    """
    return self.pdf(xs, normalize)*self.log_pdf_prime(xs, normalize)
    
  def log_pdf_prime(self, xs, normalize=True):
    """
    Returns the derivative of the log(pdf) with respect to kappa and beta.
    """
    axis = len(shape(xs))-1
    g1x = sum(self.gamma1*xs, axis)
    g2x = sum(self.gamma2*xs, axis)
    g3x = sum(self.gamma3*xs, axis)
    k, b = self.kappa, self.beta

    dfdk = g1x
    dfdb = g2x**2 - g3x**2
    df = array([dfdk, dfdb])
    if normalize:
      return transpose(transpose(df) - self.log_normalize_prime())
    else:
      return df
          
  def normalize_prime(self, cache=dict(), return_num_iterations=False):
    """
    Returns the derivative of the normalization factor with respect to kappa and beta.
    """
    k, b = self.kappa, self.beta
    if not (k, b) in cache:
      G = gamma_fun
      I = modified_bessel_2ndkind
      dIdk = lambda v, z : modified_bessel_2ndkind_derivative(v, z, 1)
      dcdk, dcdb = 0.0, 0.0
      j = 0
      if b == 0:
        dcdk = (
          ( G(j+0.5)/G(j+1) )*
          ( (-0.5*j-0.125)*(k)**(-2*j-1.5) )*
          ( I(2*j+0.5, k) )
        )
        dcdk += (
          ( G(j+0.5)/G(j+1) )*
          ( (0.5*k)**(-2*j-0.5) )*
          ( dIdk(2*j+0.5, k) )
        )   

        dcdb = 0.0 
      else:
        while True:
          dk = (
            (-1*j-0.25)*exp(
              log(b)*2*j + 
              log(0.5*k)*(-2*j-1.5)
            )*I(2*j+0.5, k)
          )
          dk += (
            exp(
              log(b)*2*j +
              log(0.5*k)*(-2*j-0.5)
            )*dIdk(2*j+0.5, k)
          )        
          dk /= G(j+1)
          dk *= G(j+0.5)                        

          db = (
            2*j*exp(
              log(b)*(2*j-1) +
              log(0.5*k)*(-2*j-0.5)
            ) * I(2*j+0.5, k)
          )
          db /= G(j+1)
          db *= G(j+0.5)                     
          dcdk += dk
          dcdb += db
        
          j += 1
          if abs(dk) < abs(dcdk)*1E-12 and abs(db) < abs(dcdb)*1E-12  and j > 5:
            break
      
        # print "dc", dcdk, dcdb, "(", k, b
      
      cache[k, b] = 2*pi*array([dcdk, dcdb])
    if return_num_iterations:
      return cache[k, b], j
    else:
      return cache[k, b]
    
  def log_normalize_prime(self, return_num_iterations=False):
    """
    Returns the derivative of the logarithm of the normalization factor.
    """
    if return_num_iterations:
      normalize_prime, num_iter = self.normalize_prime(return_num_iterations=True)
      return normalize_prime/self.normalize(), num_iter
    else:
      return self.normalize_prime()/self.normalize()

  def log_likelihood(self, xs):
    """
    Returns the log likelihood for xs.
    """
    retval = self.log_pdf(xs)
    return sum(retval, len(shape(retval)) - 1)
    
  def log_likelihood_prime(self, xs):
    """
    Returns the derivative with respect to kappa and beta of the log likelihood for xs.
    """
    retval = self.log_pdf_prime(xs)
    if len(shape(retval)) == 1:
      return retval
    else:
      return sum(retval, len(shape(retval)) - 1)
    
  def _rvs_helper(self):
    num_samples = 10000
    xs = gauss(0, 1).rvs((num_samples, 3))
    xs = divide(xs, reshape(norm(xs, 1), (num_samples, 1)))
    pvalues = self.pdf(xs, normalize=False)
    fmax = self.pdf_max(normalize=False)
    return xs[uniform(0, fmax).rvs(num_samples) < pvalues]
  
  def rvs(self, shape=None):
    """
    Returns random samples from the Kent distribution by rejection sampling. 
    May become inefficient for large kappas.
    """
    if shape == None:
      num_samples = 1
    else:
      if hasattr(shape, '__iter__'):
        num_samples = 1
        for s in shape:
          num_samples *= s
      else:
        num_samples = shape
    rvs = self._cached_rvs
    while len(rvs) < num_samples:
      new_rvs = self._rvs_helper()
      rvs = concatenate([rvs, new_rvs])
    if shape == None:
      self._cached_rvs = rvs[1:]
      return rvs[0]
    else:
      self._cached_rvs = rvs[num_samples:]
      retval = rvs[:num_samples]
      if hasattr(shape, '__iter__'):
        retval.shape = shape
      return retval
      
  def __repr__(self):
    return "kent(%s, %s, %s, %s, %s)" % (self.theta, self.phi, self.psi, self.kappa, self.beta)
      
def kent_me(xs):
  """Generates and returns a KentDistribution based on a moment estimation."""
  lenxs = len(xs)
  xbar = average(xs, 0) # average direction of samples from origin
  S = average(xs.reshape((lenxs, 3, 1))*xs.reshape((lenxs, 1, 3)), 0) # dispersion (or covariance) matrix around origin
  gamma1 = xbar/norm(xbar) # has unit length and is in the same direction and parallel to xbar
  theta, phi = KentDistribution.gamma1_to_spherical_coordinates(gamma1)
  
  H = KentDistribution.create_matrix_H(theta, phi)
  Ht = KentDistribution.create_matrix_Ht(theta, phi)
  B = MMul(Ht, MMul(S, H))
  
  eigvals, eigvects = eig(B[1:,1:])
  eigvals = real(eigvals)
  if eigvals[0] < eigvals[1]:
    eigvals[0], eigvals[1] = eigvals[1], eigvals[0]
    eigvects = eigvects[:,::-1]
  K = diag([1.0, 1.0, 1.0])
  K[1:,1:] = eigvects
  
  G = MMul(H, K)
  Gt = transpose(G)
  T = MMul(Gt, MMul(S, G))
  
  r1 = norm(xbar)
  t22, t33 = T[1, 1], T[2, 2]
  r2 = t22 - t33
  
  # kappa and beta can be estimated but may not lie outside their permitted ranges
  min_kappa = KentDistribution.minimum_value_for_kappa
  kappa = max( min_kappa, 1.0/(2.0-2.0*r1-r2) + 1.0/(2.0-2.0*r1+r2)  )
  beta  = 0.5*(1.0/(2.0-2.0*r1-r2) - 1.0/(2.0-2.0*r1+r2))
  
  return kent4(G, kappa, beta)  

def __kent_mle_output1(k_me, callback):
  print
  print "******** Maximum Likelihood Estimation ********"
  print "Initial moment estimates are:"
  print "theta =", k_me.theta
  print "phi   =", k_me.phi
  print "psi   =", k_me.psi
  print "kappa =", k_me.kappa
  print "beta  =", k_me.beta
  print "******** Starting the Gradient Descent ********"
  print "[iteration]   kappa        beta        -L"

def __kent_mle_output2(x, minusL, output_count, verbose):
  interval = verbose if isinstance(verbose, int) else 1
  str_values = list()
  for value in (tuple(x) + (minusL,)):
    str_value = "%- 8g" % value
    while len(str_value) < 12:
      str_value += " "
    str_values.append(str_value)
  if output_count[0] % interval == 0:
    print ("[%3i]       " + " %s" * 3) % tuple(output_count + str_values)
  output_count[0] = output_count[0] + 1

def kent_mle(xs, verbose=False, return_intermediate_values=False, return_bfgs_values=False, bfgs_kwargs=dict(), warning='warn'):
  """
  Generates a KentDistribution fitted to xs using maximum likelihood estimation
  For a first approximation kent_me() is used. The function 
  -k.log_likelihood(xs)/len(xs) (where k is an instance of KentDistribution) is 
  minimized.
  
  Input:
    - xs: values on the sphere to be fitted by MLE
    - verbose: if True, output is given for every step
    - return_intermediate_values: if true the values of all intermediate steps
      are returned as well
    - return_bfgs_values: if true the values from the bfgs_min algorithm are 
      returned as well
    - bfgs_args: extra arguments that can be passed to min_bfgs: not all arguments may
      be overwritten. Default value of 'disp' is 0 but may be set to 1 'full_output'
      'gtol' is chosen to be 1E-7 but may be set to other values.
      is 1 (can't be overwritten), 'callback' can't be overwritten and the first 
      three arguments of min_bfgs can't be overwritten. 
    - warning: choices are 
      - "warn": issues any warning via warning.warn
      - a file object: which results in any warning message being written to a file 
        (e.g. stdout) 
      - "none": or any other value for this argument results in no warnings to be issued
  Output:
    - an instance of the fitted KentDistribution
  Extra output:
    - if return_intermediate_values and/or return_bfgs_values is specified then
    a tuple is returned with the KentDistribution argument as the first element
    and containing the extra requested values in the rest of the elements.
  """
  # first get estimated moments
  if 'disp' not in bfgs_kwargs:
    bfgs_kwargs['disp'] = 0
  if 'gtol' not in bfgs_kwargs:
    bfgs_kwargs['gtol'] = 1E-7
  k_me = kent_me(xs)
  gamma1, gamma2, gamma3, kappa, beta = k_me.gamma1, k_me.gamma2, k_me.gamma3, k_me.kappa, k_me.beta
  min_kappa = KentDistribution.minimum_value_for_kappa
  
  # method that generates an instance of KentDistribution
  def generate_k(fudge_kappa, fudge_beta):
    # small value is added to kappa = min_kappa + abs(fudge_kappa) > min_kappa
    return kent2(gamma1, gamma2, gamma3, min_kappa + abs(fudge_kappa), abs(fudge_beta))

  # method that generates the minus L to be minimized
  def minus_log_likelihood(x):
    return -generate_k(*x).log_likelihood(xs)/len(xs)
    
  def minus_log_likelihood_prime(x):
    return -generate_k(*x).log_likelihood_prime(xs)/len(xs)
  
  # callback for keeping track of the values
  intermediate_values = list()
  def callback(x, output_count=[0]):
    minusL = -generate_k(*x).log_likelihood(xs)
    fudge_kappa, fudge_beta = x
    kappa, beta = min_kappa + abs(fudge_kappa), abs(fudge_beta)
    imv = intermediate_values
    imv.append((kappa, beta, minusL))
        
  # starting parameters (small value is subtracted from kappa and add in generatke k)
  x_start = array([kappa - min_kappa, beta])
  if verbose:
    __kent_mle_output1(k_me, callback)

  
  # here the mle is done
  all_values = fmin_bfgs(minus_log_likelihood, x_start, minus_log_likelihood_prime,
    callback=callback, full_output=1, **bfgs_kwargs)

  x_opt = all_values[0]
  warnflag = all_values[6]
  if warnflag:
    warning_message = "Unknownw warning %s" % warnflag
    if warnflag == 2:
      warning_message = "Desired error not necessarily achieved due to precision loss."
    if warnflag == 1:
      warning_message = "Maximum number of iterations has been exceeded."
    if warning == "warn":
      warnings.warn(warning_message, RuntimeWarning)
    if hasattr(warning, "write"):
      warning.write("Warning: "+warning_message+"\n")
  
  k = (generate_k(*x_opt),)
  if return_intermediate_values:
    k += (intermediate_values,)
  if  return_bfgs_values:
    k += (all_values,)
  if len(k) == 1:
    k = k[0]
  return k
  
if __name__ == "__main__":
  __doc__ += """
  >>> from kent_example import test_example_normalization, test_example_mle, test_example_mle2
  >>> from numpy.random import seed
  >>> test_example_normalization(gridsize=10)
  Calculating the watrix M_ij of values that can be calculated: kappa=100.0*i+1, beta=100.0+j*1
  Calculating normalization factor for combinations of kappa and beta: 
  Iterations necessary to calculate normalize(kappa, beta):
    8   x   x   x   x   x   x   x   x   x
    6  81 150 172 172   x   x   x   x   x
    6  51 134 172 172 172 172 172 172   x
    6  27 108 172 172 172 172 172 172   x
    6  19  71 159 172 172 172 172   x   x
    6  15  42 126 172 172 172   x   x   x
    6  13  29  86 172 172   x   x   x   x
    6  12  23  55 141   x   x   x   x   x
    x   x   x   x   x   x   x   x   x   x
    x   x   x   x   x   x   x   x   x   x
  Iterations necessary to calculate the gradient of normalize(kappa, beta):
    9   x   x   x   x   x   x   x   x   x
    6  82 150 172 172   x   x   x   x   x
    6  54 135 172 172 172 172 172 172   x
    6  31 109 172 172 172 172 172 172   x
    6  22  75 160 172 172 172 172   x   x
    6  18  47 128 172 172 172   x   x   x
    6  15  34  91 172 172   x   x   x   x
    6  14  27  61 143   x   x   x   x   x
    x   x   x   x   x   x   x   x   x   x
    x   x   x   x   x   x   x   x   x   x
  >>> test_example_mle()
  Original Distribution: k = kent(0.0, 0.0, 0.0, 1.0, 0.0)
  Drawing 10000 samples from k
  Moment estimation:  k_me = kent(0.023855003559, 0.214069389101, -1.3132066017, 1.46267336487, 0.00126428116096)
  Fitted with MLE:   k_mle = kent(0.023855003559, 0.214069389101, -1.3132066017, 1.01193032035, 0.0097247745197)
  Original Distribution: k = kent(0.75, 2.39159265359, 2.39159265359, 20.0, 0.0)
  Drawing 10000 samples from k
  Moment estimation:  k_me = kent(0.749919453671, 2.38852872459, -1.88413666083, 19.9741138811, 0.084525921375)
  Fitted with MLE:   k_mle = kent(0.749919453671, 2.38852872459, -1.88413666083, 19.974220064, 0.0985914019577)
  Original Distribution: k = kent(0.785398163397, 2.35619449019, -2.82743338823, 20.0, 2.0)
  Drawing 10000 samples from k
  Moment estimation:  k_me = kent(0.786399024085, 2.3579915613, 0.279826232127, 19.9901236404, 1.74323732912)
  Fitted with MLE:   k_mle = kent(0.786399024085, 2.3579915613, 0.279826232127, 20.0399896422, 2.05207823993)
  Original Distribution: k = kent(0.785398163397, 2.35619449019, -2.94524311274, 20.0, 5.0)
  Drawing 10000 samples from k
  Moment estimation:  k_me = kent(0.781987502626, 2.35443398261, 0.198219177812, 19.3583928323, 3.99437892323)
  Fitted with MLE:   k_mle = kent(0.781987502626, 2.35443398261, 0.198219177812, 19.8021238599, 5.0054799622)
  Original Distribution: k = kent(1.09955742876, 2.35619449019, -3.04341788317, 50.0, 25.0)
  Drawing 10000 samples from k
  Moment estimation:  k_me = kent(1.09977161299, 2.35530110258, 0.100204406833, 37.2827008959, 14.8149016955)
  Fitted with MLE:   k_mle = kent(1.09977161299, 2.35530110258, 0.100204406833, 50.5020652363, 25.3106110745)
  Original Distribution: k = kent(0.0, 0.0, 0.0981747704247, 50.0, 25.0)
  Drawing 10000 samples from k
  Moment estimation:  k_me = kent(0.00316353001387, -2.71047863482, -0.332583051616, 37.0898399888, 14.7385957253)
  Fitted with MLE:   k_mle = kent(0.00316353001387, -2.71047863482, -0.332583051616, 50.3265157358, 25.2489782311)
  >>> seed(2323)
  >>> assert test_example_mle2(300)
  Testing various combinations of kappa and beta for 300 samples.
  MSE of MLE is higher than 0.7 times the moment estimate for beta/kappa <= 0.2
  MSE of MLE is higher than moment estimate for beta/kappa >= 0.3
  MSE of MLE is five times higher than moment estimates for beta/kappa >= 0.5

  A test to ensure that the vectors gamma1 ... gamma3 are orthonormal
  >>> for k in [
  ...   kent(0.0,      0.0,      0.0,    20.0, 0.0),
  ...   kent(-0.25*pi, -0.25*pi, 0.0,    20.0, 0.0),
  ...   kent(-0.25*pi, -0.25*pi, 0.0,    20.0, 5.0),
  ...   kent(0.0,      0.0,      0.5*pi, 10.0, 7.0),
  ...   kent(0.0,      0.0,      0.5*pi, 0.1,  0.0),
  ...   kent(0.0,      0.0,      0.5*pi, 0.1,  0.1),
  ...   kent(0.0,      0.0,      0.5*pi, 0.1,  8.0),
  ... ]:
  ...   assert(abs(sum(k.gamma1 * k.gamma2)) < 1E-14)
  ...   assert(abs(sum(k.gamma1 * k.gamma3)) < 1E-14)
  ...   assert(abs(sum(k.gamma3 * k.gamma2)) < 1E-14)
  ...   assert(abs(sum(k.gamma1 * k.gamma1) - 1.0) < 1E-14)
  ...   assert(abs(sum(k.gamma2 * k.gamma2) - 1.0) < 1E-14)
  ...   assert(abs(sum(k.gamma3 * k.gamma3) - 1.0) < 1E-14)

  A test to ensure that the pdf() and the pdf_max() are calculated
  correctly.
  >>> from numpy.random import seed
  >>> from scipy.stats import norm as gauss
  >>> seed(666)
  >>> for k, pdf_value in [
  ...   (kent(0.0,      0.0,      0.0,    20.0, 0.0), 3.18309886184),
  ...   (kent(-0.25*pi, -0.25*pi, 0.0,    20.0, 0.0), 0.00909519370),
  ...   (kent(-0.25*pi, -0.25*pi, 0.0,    20.0, 5.0), 0.09865564569),
  ...   (kent(0.0,      0.0,      0.5*pi, 10.0, 7.0), 0.59668931662),
  ...   (kent(0.0,      0.0,      0.5*pi, 0.1,  0.0), 0.08780030026),
  ...   (kent(0.0,      0.0,      0.5*pi, 0.1,  0.1), 0.08768344462),
  ...   (kent(0.0,      0.0,      0.5*pi, 0.1,  8.0), 0.00063128997),
  ... ]:
  ...   assert abs(k.pdf(array([1.0, 0.0, 0.0])) - pdf_value) < 1E-8
  ...   assert abs(k.log_pdf(array([1.0, 0.0, 0.0])) - log(pdf_value)) < 1E-8
  ...   num_samples = 100000
  ...   xs = gauss(0, 1).rvs((num_samples, 3))
  ...   xs = divide(xs, reshape(norm(xs, 1), (num_samples, 1)))
  ...   values = k.pdf(xs, normalize=False)
  ...   fmax = k.pdf_max(normalize=False)
  ...   assert all(values <= fmax)
  ...   assert any(values > fmax*0.999)
  ...   values = k.pdf(xs)
  ...   fmax = k.pdf_max()
  ...   assert all(values <= fmax)
  ...   assert any(values > fmax*0.999)

  These are tests to ensure that the coordinate transformations are done correctly
  that the functions that generate instances of KentDistribution are consistent and
  that the derivatives are calculated correctly. In addition some more orthogonality 
  testing is done.

  >>> from kent_distribution import *
  >>> from numpy.random import seed
  >>> from scipy.stats import uniform 
  >>> def test_orth(k):
  ...   # a bit more orthonormality testing for good measure 
  ...   assert(abs(sum(k.gamma1 * k.gamma2)) < 1E-14)
  ...   assert(abs(sum(k.gamma1 * k.gamma3)) < 1E-14)
  ...   assert(abs(sum(k.gamma3 * k.gamma2)) < 1E-14)
  ...   assert(abs(sum(k.gamma1 * k.gamma1) - 1.0) < 1E-14)
  ...   assert(abs(sum(k.gamma2 * k.gamma2) - 1.0) < 1E-14)
  ...   assert(abs(sum(k.gamma3 * k.gamma3) - 1.0) < 1E-14)
  ... 
  >>> # generating some specific boundary values and some random values
  >>> seed(666)
  >>> upi, u2pi = uniform(0, pi), uniform(-pi, 2*pi)
  >>> thetas, phis, psis = list(upi.rvs(925)), list(u2pi.rvs(925)), list(u2pi.rvs(925))
  >>> for a in (0.0, 0.5*pi, pi):
  ...   for b in (-pi, -0.5*pi, 0, 0.5*pi, pi):
  ...     for c in (-pi, -0.5*pi, 0, 0.5*pi, pi):
  ...       thetas.append(a)
  ...       phis.append(b)
  ...       psis.append(c)
  ... 
  >>> # testing consintency of angles (specifically kent())
  >>> for theta, phi, psi in zip(thetas, phis, psis):
  ...   k = kent(theta, phi, psi, 1.0, 1.0)
  ...   assert abs(theta - k.theta) < 1E-12
  ...   a = abs(phi - k.phi)
  ...   b = abs(psi - k.psi)
  ...   if theta != 0 and theta != pi:
  ...     assert a < 1E-12 or abs(a-2*pi) < 1E-12
  ...     assert b < 1E-12 or abs(b-2*pi) < 1E-12
  ...   test_orth(k)
  ... 
  >>> # testing consistency of gammas and consistency of back and forth
  >>> # calculations between gammas and angles (specifically kent2(), kent3() and kent4())
  >>> kappas = gauss(0, 2).rvs(1000)**2
  >>> betas = gauss(0, 2).rvs(1000)**2
  >>> for theta, phi, psi, kappa, beta in zip(thetas, phis, psis, kappas, betas):
  ...   gamma1, gamma2, gamma3 = KentDistribution.spherical_coordinates_to_gammas(theta, phi, psi)
  ...   theta, phi, psi = KentDistribution.gammas_to_spherical_coordinates(gamma1, gamma2)
  ...   gamma1a, gamma2a, gamma3a = KentDistribution.spherical_coordinates_to_gammas(theta, phi, psi)
  ...   assert all(abs(gamma1a-gamma1) < 1E-12) 
  ...   assert all(abs(gamma2a-gamma2) < 1E-12) 
  ...   assert all(abs(gamma3a-gamma3) < 1E-12) 
  ...   k2 = kent2(gamma1, gamma2, gamma3, kappa, beta)
  ...   assert all(abs(gamma1 - k2.gamma1) < 1E-12) 
  ...   assert all(abs(gamma2 - k2.gamma2) < 1E-12) 
  ...   assert all(abs(gamma3 - k2.gamma3) < 1E-12)
  ...   A = gamma1*kappa
  ...   B = gamma2*beta
  ...   k3 = kent3(A, B)
  ...   assert all(abs(gamma1 - k3.gamma1) < 1E-12) 
  ...   assert all(abs(gamma2 - k3.gamma2) < 1E-12) 
  ...   assert all(abs(gamma3 - k3.gamma3) < 1E-12)
  ...   test_orth(k)
  ...   gamma = array([
  ...     [gamma1[0], gamma2[0], gamma3[0]], 
  ...     [gamma1[1], gamma2[1], gamma3[1]], 
  ...     [gamma1[2], gamma2[2], gamma3[2]], 
  ...   ])
  ...   k4 = kent4(gamma, kappa, beta)
  ...   assert all(k2.gamma1 == k4.gamma1) 
  ...   assert all(k2.gamma2 == k4.gamma2) 
  ...   assert all(k2.gamma3 == k4.gamma3) 
  ... 
  >>> # testing special case for B with zero length (kent3())
  >>> for theta, phi, psi, kappa, beta in zip(thetas, phis, psis, kappas, betas):
  ...   gamma1, gamma2, gamma3 = KentDistribution.spherical_coordinates_to_gammas(theta, phi, psi)
  ...   A = gamma1*kappa
  ...   B = gamma2*0.0
  ...   k = kent3(A, B)
  ...   assert all(abs(gamma1 - k.gamma1) < 1E-12) 
  ...   test_orth(k)
  ... 
  >>> # testing the derivatives
  >>> for theta, phi, psi, kappa, beta in zip(thetas, phis, psis, kappas, betas):
  ...   k0 = kent(theta, phi, psi, kappa, beta)
  ...   eps = 1E-7
  ...   kk = kent(theta, phi, psi, kappa+eps, beta)
  ...   kb = kent(theta, phi, psi, kappa, beta+eps)
  ...   num_samples = 101
  ...   xs = gauss(0, 1).rvs((num_samples, 3))
  ...   xs = divide(xs, reshape(norm(xs, 1), (num_samples, 1)))
  ...   for ys in [xs[0], xs[1:], k0.rvs(), k0.rvs(100)]:
  ...     f = False
  ...     for name, f0, fk, fb, fprime in (
  ...       ("k0.pdf(ys),            ", k0.pdf(ys),            kk.pdf(ys),            kb.pdf(ys),            k0.pdf_prime(ys)           ),
  ...       ("k0.pdf(ys, f),         ", k0.pdf(ys, f),         kk.pdf(ys, f),         kb.pdf(ys, f),         k0.pdf_prime(ys, f)        ),
  ...       ("k0.log_pdf(ys),        ", k0.log_pdf(ys),        kk.log_pdf(ys),        kb.log_pdf(ys),        k0.log_pdf_prime(ys)       ),
  ...       ("k0.log_pdf(ys, f),     ", k0.log_pdf(ys, f),     kk.log_pdf(ys, f),     kb.log_pdf(ys, f),     k0.log_pdf_prime(ys, f)    ),
  ...       ("k0.normalize(),        ", k0.normalize(),        kk.normalize(),        kb.normalize(),        k0.normalize_prime()       ),
  ...       ("k0.log_normalize(),    ", k0.log_normalize(),    kk.log_normalize(),    kb.log_normalize(),    k0.log_normalize_prime()   ),
  ...       ("k0.log_likelihood(ys), ", k0.log_likelihood(ys), kk.log_likelihood(ys), kb.log_likelihood(ys), k0.log_likelihood_prime(ys)),
  ...     ):
  ...       fprime_approx = array([(fk-f0)/eps, (fb-f0)/eps])
  ...       assert all(abs(fprime_approx - fprime) < 1E-2*(abs(fprime_approx)+abs(fprime)))
  ...       assert sum(abs(fprime_approx - fprime) > 1E-4*(abs(fprime_approx)+abs(fprime))) < 5
  ...  
  """

  import doctest
  doctest.testmod(optionflags=doctest.ELLIPSIS)

