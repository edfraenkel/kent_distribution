#!/usr/bin/env python
"""
The example code below shows various examples of the Kent distribution
some random samples are drawn from the distribution and new estimates
are generated using a moment estimation and a maximum likelihood fit.
If this script is called with the --make-plots option then plots of
the results will be shown.
"""

from kent_distribution import *
from numpy.random import seed, uniform
import warnings
import sys

seed(2323)

def test_example_normalization(showplots=False, verbose=False, gridsize=100, print_grid=True):
  scale = (1000.0/gridsize)
  print "Calculating the matrix M_ij of values that can be calculated: kappa=%.1f*i+1, beta=%.1f+j*1" % (scale, scale)
  with warnings.catch_warnings():
    warnings.simplefilter("error")
    c_grid, dck_grid, dcb_grid = [zeros((gridsize, gridsize))-1.0 for z in xrange(3)]
    cnum_grid, dcnum_grid = [zeros((gridsize, gridsize), dtype=int32)-1 for z in xrange(2)]
    sys.stdout.write("Calculating normalization factor for combinations of kappa and beta: ")
    for i in xrange(gridsize):
      if verbose:
        sys.stdout.write("%s/%s " % (i*gridsize, gridsize*gridsize))
        sys.stdout.flush()
      kappa = scale*i + 1.0
      for j in xrange(gridsize):
        beta = scale*j + 1.0
        f = kent(0.0, 0.0, 0.0, kappa, beta) 
        try:
          c, cnum = f.log_normalize(return_num_iterations=True)
          c_grid[i, j] = c
          cnum_grid[i, j] = cnum
        except (OverflowError, RuntimeWarning):
          pass
        try:
          (dck, dcb), cnum = f.log_normalize_prime(return_num_iterations=True)
          dck_grid[i, j] = dck
          dcb_grid[i, j] = dcb
          dcnum_grid[i, j] = cnum  
        except (OverflowError, RuntimeWarning):
          pass
    if showplots:
      from pylab import figure, show
      for name, grid in zip(
        [
          r"$\mathrm{Calculated\ values\ of\ }c(\kappa,\beta)$", 
          r"$\mathrm{Calculated\ values\ of\ }{\frac{\partial}{\partial \kappa}}c(\kappa,\beta)$",
          r"$\mathrm{Calculated\ values\ of\ }{\frac{\partial}{\partial \beta}}c(\kappa,\beta)$",
          r"$\mathrm{Iterations\ necessary\ to\ calculate\ }c(\kappa,\beta)$",
          r"$\mathrm{Iterations\ necessary\ to\ calculate\ }{\nabla}c(\kappa,\beta)$",
        ],
        [ c_grid,   dck_grid,   dcb_grid,   cnum_grid,   dcnum_grid]
      ):
        f = figure()
        ax = f.add_subplot(111)
        cb = ax.imshow(grid, interpolation="nearest")
        f.colorbar(cb)
        ax.set_title(name+" $(-1=\mathrm{overflow}$)")
        ax.set_xticklabels([str(int(t*scale+1)) for t in ax.get_xticks()])
        ax.set_yticklabels([str(int(t*scale+1)) for t in ax.get_yticks()])
        ax.set_ylabel(r"$\kappa$")
        ax.set_xlabel(r"$\beta$")
  print
  if print_grid:
    for message, grid in [
      ("Iterations necessary to calculate normalize(kappa, beta):", cnum_grid),
      ("Iterations necessary to calculate the gradient of normalize(kappa, beta):", dcnum_grid)
    ]:
      print message
      for i, line in enumerate(grid):
        print " ".join(['  x' if n == -1.0 else '%3i' % n for n in line])
    
  
def test_example_mle(showplots=False):
  for k in [
    kent(0.0,       0.0,     0.0,    1.0,  0.0),
    kent(-0.75,    -0.75,   -0.75,   20.0, 0.0),
    kent(-0.25*pi, -0.25*pi, pi/10,  20.0, 2.0),
    kent(-0.25*pi, -0.25*pi, pi/16,  20.0, 5.0),
    kent(-0.35*pi, -0.25*pi, pi/32,  50.0, 25.0),
    kent(0.0, 0.0, pi/32,  50.0, 25.0),
  ]:
    print "Original Distribution: k =", k
    gridsize = 200
    u = linspace(0, 2 * pi, gridsize)
    v = linspace(0, pi, gridsize)

    x = outer(cos(u), sin(v))
    y = outer(sin(u), sin(v))
    z = outer(ones(size(u)), cos(v))

    
    keys = list()
    points = list()
    for i in range(gridsize):
      for j in range(gridsize):
        points.append([x[i, j], y[i, j], z[i, j]])
        keys.append((i, j))
    points = array(points)

    print "Drawing 10000 samples from k"
    xs = k.rvs(10000)
    k_me = kent_me(xs)
    print "Moment estimation:  k_me =", k_me
    k_mle = kent_mle(xs, warning=sys.stdout)
    print "Fitted with MLE:   k_mle =", k_mle
    assert k_me.log_likelihood(xs) < k_mle.log_likelihood(xs)

    value_for_color = k_mle.pdf(points)
    value_for_color /= max(value_for_color)  
    colors = empty((gridsize, gridsize), dtype=tuple)
    for (i, j), v in zip(keys, value_for_color):
      colors[i, j] = (1.0-v, 1.0-v, 1.0, 1.0)

    if showplots:
      from mpl_toolkits.mplot3d import Axes3D
      import matplotlib.pyplot as plt

      f = plt.figure()
      ax = f.add_subplot(111, projection='3d')
      xx, yy, zz = zip(*xs[:100]) # plot only a portion of these values
      ax.scatter(1.05*array(xx), 1.05*array(yy), 1.05*array(zz), c='b')
      ax.plot_surface(x, y, z, rstride=4, cstride=4, facecolors=colors, linewidth=0)
      values_t = r"$\theta=%.2f^\circ,\ \phi=%.2f^\circ,\ \psi=%.2f^\circ,\ \kappa=%.3f,\ \beta=%.3f$" 
      f.text(0.12, 0.99-0.025, "$\mathrm{Original\ Values:}$")
      f.text(0.12, 0.99-0.055, "$\mathrm{Moment\ estimates:}$")
      f.text(0.12, 0.99-0.080, "$\mathrm{MLE\ (shown):}$")
      f.text(0.30, 0.99-0.025, values_t % (k.theta*180/pi,     k.phi*180/pi,     k.psi*180/pi,     k.kappa,     k.beta))
      f.text(0.30, 0.99-0.055, values_t % (k_me.theta*180/pi,  k_me.phi*180/pi,  k_me.psi*180/pi,  k_me.kappa,  k_me.beta))
      f.text(0.30, 0.99-0.080, values_t % (k_mle.theta*180/pi, k_mle.phi*180/pi, k_mle.psi*180/pi, k_mle.kappa, k_mle.beta))
      ax.set_xlabel(r"$Q\rightarrow$")
      ax.set_ylabel(r"$U\rightarrow$")
      ax.set_zlabel(r"$V\rightarrow$")      

def calculate_bias_var_and_mse(x, y):
  bias = average(x) - average(y)
  variance = var(x-y)
  mse = average((x-y)**2)
  assert abs((bias**2 + variance) - mse) < 1E-12*(bias**2 + variance + mse)
  return bias, variance, mse

def test_example_mle2(num_samples, showplots=False, verbose=False, stepsize=1.0):
  max_kappa = 50.0
  real_kappas = arange(1.0, max_kappa, stepsize)
  print "Testing various combinations of kappa and beta for", num_samples, "samples."
  bias_var_mse_kappa_me, bias_var_mse_kappa_mle, bias_var_mse_beta_me, bias_var_mse_beta_mle = [list() for i in xrange(4)]
  beta_ratios = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)
  for beta_ratio in beta_ratios:
    real_betas = beta_ratio*real_kappas
    kappas_me, kappas_mle, betas_me, betas_mle = [list() for i in xrange(4)]
    if verbose:
      print "beta (max 2.0) = %s*kappa : kappa (max %.1f) = " % (beta_ratio, max_kappa),
    for kappa in real_kappas:
      if verbose:
        print "%.1f" % kappa,
        sys.stdout.flush()
      beta = kappa*beta_ratio
      k = kent(uniform(0, pi), uniform(0, 2*pi), uniform(0, 2*pi), kappa, beta)
      samples = k.rvs(num_samples)
      k_me = kent_me(samples)
      k_mle = kent_mle(samples, warning=sys.stdout)
      assert k_me.log_likelihood(samples) < k_mle.log_likelihood(samples)
      kappas_me.append(k_me.kappa)
      betas_me.append(k_me.beta)
      kappas_mle.append(k_mle.kappa)
      betas_mle.append(k_mle.beta)
    bias_var_mse_kappa_me.append(calculate_bias_var_and_mse(real_kappas, kappas_me))
    bias_var_mse_beta_me.append(calculate_bias_var_and_mse(real_betas, betas_me))
    bias_var_mse_kappa_mle.append(calculate_bias_var_and_mse(real_kappas, kappas_mle))
    bias_var_mse_beta_mle.append(calculate_bias_var_and_mse(real_betas, betas_mle))
    if verbose:
      print
    if showplots:
      from pylab import figure, show
      f = figure(figsize=(12.0, 5))
      ax = f.add_subplot(121)
      ax.plot(real_kappas, kappas_me, label=r"$\kappa\ \mathrm{Moment\ Est.}$")
      ax.plot(real_kappas, kappas_mle, label=r"$\kappa\ \mathrm{MLE}$")
      ax.plot(real_kappas, real_kappas, '-.k')  
      ax.set_xlabel(r"$\mathrm{Real}\ \kappa$")
      ax.set_xlabel(r"$\mathrm{Fitted}\ \kappa$")
      ax.set_title(r"$\mathrm{Number of Samples = %s}" % num_samples)
      ax.legend()  
      ax = f.add_subplot(122)
      ax.plot(real_betas, betas_me, label=r"$\beta\ \mathrm{Moment\ Est.}$")
      ax.plot(real_betas, betas_mle, label=r"$\beta\ \mathrm{MLE}$")
      ax.plot(real_betas, real_betas, '-.k')  
      ax.set_xlabel(r"$\mathrm{Real}\ \beta$")
      ax.set_xlabel(r"$\mathrm{Fitted}\ \beta$")
      ax.set_title(r"$\beta = %.1f\kappa$" % beta_ratio)
      ax.legend()
  if showplots:
    for (name, bias_var_mse_mle, bias_var_mse_me) in [
      ("kappa", bias_var_mse_kappa_mle, bias_var_mse_kappa_me),
      ("beta", bias_var_mse_beta_mle, bias_var_mse_beta_me),
    ]:
      f = figure()
      ax = f.add_subplot(111)
      bias_me, var_me, mse_me = zip(*bias_var_mse_me)
      bias_mle, var_mle, mse_mle = zip(*bias_var_mse_mle)
      ax.plot(bias_me, bias_mle, label='bias')
      ax.plot(var_me, var_mle, label='var')
      ax.plot(mse_me, mse_mle, label='mse')
      for me, mle in [
        (bias_me, bias_mle),
        (var_me, var_mle),
        (mse_me, mse_mle),
      ]:
        for x, y, br in zip(me, mle, beta_ratios):
          ax.text(x, y, "%.2f" % br)
      ax.set_xlabel("Moment Estimate")
      ax.set_ylabel("MLE")
      xl = ax.get_xlim()
      yl = ax.get_ylim()
      mxl = (min(xl[0], yl[0]), max(xl[1], yl[1]))
      ax.plot(mxl, mxl, '-.k')
      ax.set_xlim(mxl)
      ax.set_ylim(mxl)
      ax.set_title(r"$\%s$" % name)
      ax.legend()
  
  for (name, bias_var_mse_mle, bias_var_mse_me, beta_ratio) in [
    ("kappa", bias_var_mse_kappa_mle, bias_var_mse_kappa_me, beta_ratios),
    ("beta", bias_var_mse_beta_mle, bias_var_mse_beta_me, beta_ratios),
  ]:
    biass_me, vars_me, mses_me = zip(*bias_var_mse_me)
    biass_mle, vars_mle, mses_mle = zip(*bias_var_mse_mle)
    for mse_me, mse_mle, beta_ratio in zip(mses_me, mses_mle, beta_ratios):
      if mse_me < mse_mle*0.7:
        print "MSE of MLE is lower than 0.7 times the moment estimate for %s" % name
        return False
      if beta_ratio >= 0.3:
        if mse_me < mse_mle:
          print "MSE of MLE is lower than moment estimate for %s with beta/kappa >= 0.3" % name
          return False
      if beta_ratio > 0.5:
        if mse_me < 5*mse_mle:
          print "MSE of MLE is not lower than five times the moment estimate %s with beta/kappa >= 0.5" % name
          return False
      
  print "MSE of MLE is higher than 0.7 times the moment estimate for beta/kappa <= 0.2"  
  print "MSE of MLE is higher than moment estimate for beta/kappa >= 0.3"
  print "MSE of MLE is five times higher than moment estimates for beta/kappa >= 0.5"
  return True

if __name__ == "__main__":
  from sys import argv
  showplots = False if len(argv) > 1 and '--no-plots' in argv[1:] else True
  # this should go to the bottom

  if not (len(argv) > 1 and '--no-normalization' in argv[1:]):
    test_example_normalization(showplots=showplots, verbose=True, print_grid=False)
  if not (len(argv) > 1 and '--no-mle' in argv[1:]):
    test_example_mle(showplots=showplots)
  if not (len(argv) > 1 and '--no-mle2' in argv[1:]):
    test_example_mle2(300, showplots=showplots, verbose=True)
    test_example_mle2(10000, showplots=showplots, verbose=True)
  if showplots:
    from pylab import show
    show()
  exit()
  
