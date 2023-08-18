import numpy as np
import scipy.linalg as spla


def _linesearch(f, delta_xt, minusbt, xt, C, fxprime):
    """A robust non-local linesearch method."""
    evals = min(20, C//2)
    alpha = 1.2
    outdx = []
    outb = []
    # evaluate the function along both search directions
    for i in range(evals):
        outdx.append(f(xt + delta_xt * alpha**(i-evals//2)))
        outb.append(f(xt + minusbt * alpha**(i-evals//2)))
    index = np.argmin(np.concatenate((np.array(outdx), np.array(outb))))
    # find the best candidate
    if index < evals:
        out = outdx
        delt = delta_xt
    else:
        out = outb
        index = index-evals
        delt = minusbt
    alpha1 = alpha**(index-evals//2)
    fx1 = out[index]
    # if there is improvement
    if fx1 < fxprime:
        return xt+alpha1*delt, fx1, evals*2
    else:
        return xt, fxprime, evals*2


def _scaling(sigma0, sigmat, xdiff):
    """Scaling of the sample points."""
    prec = 1.e-4
    if sigmat < prec:
        sigmat = sigma0
    if np.linalg.norm(xdiff) < prec:
        return sigmat/2.
    elif np.linalg.norm(xdiff) > sigmat*2:
        return np.linalg.norm(xdiff)/2.
    else:
        return sigmat


def _nndir(x0, gf, z, sigma0):
    """Non-locally L2-fits a quadratic model that
    has similar gradients as the objective."""
    G = gf(x0[:,None] + sigma0 * z)
    Z = 2.*sigma0*z
    z_bar = np.mean(Z, axis=1)
    g_bar = np.mean(G, axis=1)
    P = np.matmul((Z-z_bar[:,None]), Z.T)
    V = np.matmul((G-g_bar[:,None]), Z.T)
    A_tilde0 = spla.solve_continuous_lyapunov(P.T, V+V.T)
    A0 = (A_tilde0 + A_tilde0)/4.
    b0 = g_bar - np.matmul(A0 + A0.T, z_bar)
    delta_x0 = np.linalg.solve(2*(A0 + A0.T), -b0)
    return delta_x0, b0, g_bar


def nlqn(f, gf, x0, sigma0, k, C, linesearch=_linesearch, scaling=_scaling, verbose=True):
    """Non-local quasi-Newton method for optimization of
    differentiable functions with many suboptimal local minima.
        
    Non-locally search directions from non-local quadratic approximants
    based on gradients of the objective function.
  
    Args:
        f (function): Continuously differentiable objective function
        mapping |R^n -> |R. Must be vectorized, i.e., accept input of
        shape `(n, m)`, where `m` is a number of evaluation points.

        gf (function): Function returning the gradient of `f`.
        Must be vectorized, i.e., accept input of
        shape `(n, m)`, where `m` is a number of evaluation points.
        Must return output of shape `(n,m)`.

        x0 (np.ndarray): Initial point in |R^n.

        sigma0 (float): Initial scaling in (0, infty).

        k (int): Sample size used to determine the search directions.

        C (int): Total evaluation budget counting gradient and function
        evaluations separate and equally.
    
        linesearch (function): Non-local line search method.
        Defaults to log-linear grid-search.

        scaling (function): Determines the scaling of the sample
        points at each iteration. Defaults to log-linear grid-search
        of scalings.

        verbose (bool): Whether to print optimization progress at each
        iteration.

    Returns:
        np.ndarray:
        A point in |R^n that has the lowest found objective value.
    """
    xt = x0
    sigmat = sigma0
    fxprime = np.inf
    n = len(x0)
    
    while C > k:
        # sample evaluation points
        z = np.random.randn(n, k)
        # compute newton direction and robust gradient
        delta_xt, bt, _ = _nndir(xt, gf, z, sigmat)
        # linesearch along newton direction and robust gradient
        xt1, fxprime, evals = linesearch(f, delta_xt, -bt, xt, C-k, fxprime)
        # rescale sampling
        sigmat = scaling(sigma0, sigmat, xt1-xt)
        # update best guess
        xt = xt1
        # count evaluations
        C -= k + evals

        if verbose:
            print(f"f-val: {fxprime}  //  sigmat: {sigmat} // budget: {C}")

    return xt
