#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import numpy.ma as ma
from scipy.optimize import minimize
import scipy.stats
from warnings import warn
from math import ceil
import GPy
from my_predictive_gradients import my_predictive_gradient

##### Optimization of personalized intervention #####
def personalizedIntervention(individual, X, model, fixed_features = None,
                    cardinality = None, constraint_bounds=None,
                    smoothing_levels = (), quantile = 0.05, l = 2.0, 
                    eta = 1.0, max_iter = 1e4, convergence_thresh = 1e-9,
                    max_l_search = 100):
    '''
    Runs smoothed version of personalized intervention 
    # with a sparsity constraint, using gradient descent + soft-threholding
    # and binary search to find proper lambda regularization.
    # Returns tuple: (optimal_shift, optimal_objective_value).
    # NOTE: only returns SHIFT, does NOT return individual + optimal_shift! need to do this yourself to find desired post-intervention covariate-setting for this individual.
    # model = fitted GP model to training (X, y) pairs.
    # individual = the features of the new individual to intervene upon (1-D array)
    # fixed_features = list of features which cannot be transformed (zero-indexed)
    # cardinality = number of variables that can be intervened upon, None if there is no constraint
    # constraint_bounds should be list of tuples, one for each dimension indicating the min/max SHIFT allowed, None if there is no constraint.  Set = (0,0) to indicate a feature which is fixed and cannot be transformed. Set = (None, None) for a feature which is unconstrained.
    # smoothing_levels = tuple containing different amounts of smoothing to try out, None if there is no smoothing to be performed. Smoothing is used to circumvent nonconvexity in optimization.
    # quantile = which quantile of posterior-gain to use (0.05 by default).
    # l = initial maximal l1-regularization penalty to try.
    # eta, max_iter, convergence_thresh = parameters for gradient method.
    # max_l_search: maximal number of iterations in binary search for desired sparsity-level (don't try more than this many times, simply chose the largest nonzero terms if we still havent achieved target cardinality)
    '''
    d = X.shape[1] # number of features.
    if fixed_features is not None: # set additional constraints:
    	if constraint_bounds is None:
    		constraint_bounds = [(None, None) if (i not in fixed_features) else (0.0,0.0) for i in range(d)]
    	else:
    		constraint_bounds = [(constraint_bounds[i][0], constraint_bounds[i][1]) if (i not in fixed_features) else (0.0,0.0) for i in range(d)]
    
    # First run without penalty (to get initial solns):
    init_diff, obj_val = smoothedPI(X, model, individual, 
    						smoothing_levels = smoothing_levels, 
    						constraint_bounds=constraint_bounds, 
    						l =0.0, quantile = quantile)
    #print("init_diff:",init_diff)
    current_cardinality = d - sum(np.isclose(init_diff, np.zeros(d)))
    if (cardinality is None) or (cardinality >= d) or (current_cardinality <= cardinality):
        return (init_diff, obj_val)
    if cardinality <= 0:
        return (np.zeros(d), 0.0)
    l = l / 2.0 # initial_max
    while (current_cardinality > cardinality): # need more regularization.
        l *= 2.0
        print("Upper l search: Trying l="+ str(l))
        opt_diff, obj_val = smoothedPI(X, model, individual, 
    						smoothing_levels = smoothing_levels, constraint_bounds=constraint_bounds, 
    						l =l, quantile = quantile, initial_diff = init_diff, 
    						eta = eta, max_iter = max_iter, convergence_thresh = convergence_thresh)
        print("diff:",opt_diff)
        current_cardinality = d - sum(np.isclose(opt_diff, np.zeros(d)))
        print(opt_diff)
        print("Current cardinality= " + str(current_cardinality))
        print(np.isclose(current_cardinality, cardinality))
    upper_l = l
    print("Found upper lambda, l=" + str(upper_l))
    # Perform binary search to find right amount of regularization:
    ub = l
    lb = 0
    iteration = 0
    while not np.isclose(current_cardinality, cardinality):
        if iteration >= max_l_search: # simply chose the largest nonzero terms in this case.
            break
        last_size = ub - lb
        l = (lb + ub) / 2
        print("Binary search: Trying l="+ str(l))
        opt_diff, obj_val = smoothedPI(X, model, individual,
    						smoothing_levels = smoothing_levels, constraint_bounds=constraint_bounds, 
    						l =l, quantile = quantile, initial_diff = init_diff, 
    						eta = eta, max_iter = max_iter, convergence_thresh = convergence_thresh)
        current_cardinality = d - sum(np.isclose(opt_diff, np.zeros(d)))
        print("diff:",opt_diff)
        print("Current cardinality= " + str(current_cardinality))
        if current_cardinality > cardinality:
            lb = l
        elif current_cardinality < cardinality:
            ub = l
        if abs(ub - lb - last_size) < 1e-9: # size is not changing much, re-enlarge:
        	lb = max(0.0,lb - 1e-2)
        	ub = min(ub + 1e-2, 2.0*upper_l)
        	init_diff = np.sign(init_diff) * np.maximum(np.abs(init_diff) - 1e-3, 0) # shrink toward 0. 
        iteration += 1
    selected_features = np.where(np.logical_not(np.isclose(opt_diff, np.zeros(d))))[0] # features which are chosen for optimization.
    # print("selected_features:"+str(selected_features)+ "  length="+str(len(selected_features)))
    if not np.isclose(len(selected_features), cardinality):
        print("warning: could not achieve target cardinality")
        ordering = np.argsort(-abs(opt_diff))
        selected_features = ordering[range(cardinality)] # simply choose top entries in this failure case.
        opt_diff[[z for z in range(d) if z not in selected_features]] = 0.0
    # Modify constraint_bounds to ensure not-selected features remain fixed at 0.
    if constraint_bounds is not None:
        print("a selected_features:"+str(selected_features))
        sparse_constraints = [constraint_bounds[i] if (i in selected_features) else (0.0,0.0) for i in range(d)]
    else:
        print("b selected_features:"+str(selected_features))
        sparse_constraints = [(None, None) if (i in selected_features) else (0.0,0.0) for i in range(d)]
    # print(sparse_constraints)
    return(smoothedPI(X, model, individual, 
    			smoothing_levels = smoothing_levels, constraint_bounds=sparse_constraints, 
    			l =0.0, quantile = quantile, initial_diff = None, 
    			eta = eta, max_iter = max_iter, convergence_thresh = convergence_thresh))


def smoothedPI(X, model, individual, smoothing_levels = (),
                constraint_bounds=None, l = 0.0,
                quantile = 0.05, initial_diff = None,
                eta = 1.0, max_iter = 1e4, convergence_thresh = 1e-9):
    # Performs smoothing + optimization.
    # eta, max_iter, convergence_thresh = parameters for gradient method.
    current_guess = initial_diff 
    if (len(smoothing_levels) >= 1):  # Perform Smoothing (only works for standard GP regression with ARD kernel):
        smoothing_levels = sorted(smoothing_levels, reverse = True)
        orig_lengthscale = model.kern.lengthscale.copy()
        orig_variance = model.kern.variance.copy()
        orig_noise_var = model.likelihood.variance.copy()
        if initial_diff is None: # Set initial guess to zero-shift.
            initial_diff = np.zeros(X.shape[1])
        max_features = np.amax(X, axis=0) - individual # need to subtract off individual.
        min_features = np.amin(X, axis=0) - individual
        # Add additional constraints during smoothing to ensure we don't go beyond data range:
        if constraint_bounds is not None:
            upper_bounds = np.array([w[1] if (w[1] is not None) else float('inf') for w in constraint_bounds])
            lower_bounds = np.array([w[0] if (w[0] is not None) else -float('inf') for w in constraint_bounds])
            smoothing_constraints = [(max(min_features[i],lower_bounds[i]), max(min_features[i],max(lower_bounds[i],min(max_features[i],upper_bounds[i])))) for i in range(X.shape[1])]
        else:
            smoothing_constraints = [(min_features[i], max_features[i]) for i in range(X.shape[1])]
        # print(smoothing_constraints)
        for smooth_amt in smoothing_levels:
            if smooth_amt > 1.0:
                # print("Smooth_amt="+str(smooth_amt))
                model.kern.lengthscale = smooth_amt * orig_lengthscale
                model.kern.lengthscale.fix()
                model.optimize() # Recompute variance and noise scale.
                # perform optimization with more aggressive parameter settings:
                current_guess, objval = basePIoptimization(X, model, individual,
                                            constraint_bounds=smoothing_constraints, l = l, 
                                            quantile = quantile,
                                            initial_diff = current_guess, 
                                            eta = eta*5, max_iter = ceil(max_iter/10.0), 
                                            convergence_thresh = convergence_thresh * 10)
                # print(current_guess)
        # Restore original model:
        model.kern.lengthscale = orig_lengthscale
        model.kern.variance = orig_variance
        model.likelihood.variance = orig_noise_var
        # print("Smooth_amt= 1")
    return (basePIoptimization(X, model, individual, 
                constraint_bounds=constraint_bounds, l = l, 
                quantile = quantile, 
                initial_diff = current_guess, 
                eta = eta, max_iter = max_iter, 
                convergence_thresh = convergence_thresh))

def basePIoptimization(x, model, individual,
                       constraint_bounds=None, l = 0.0, quantile = 0.05, 
                       initial_diff = None, 
                       eta = 1.0, max_iter = 1e4, convergence_thresh = 1e-9):
    # Initial guess for optimal shift.
    # Finds local optimum of population intervention objective, 
    # for a given regularizer l and a given level of smoothness specified in the 'model' object.
    # Returns tuple: (optimal_shift, optimal_objective_value)
    # eta, max_iter, convergence_thresh = parameters for gradient method.
    ZSCORE = scipy.stats.norm.ppf(quantile)
    if np.isclose(l, 0.0):
        jac = False
    else: 
        jac = True # To use gradient ascent.
    if initial_diff is None: # Set initial guess to zero-shift.
        initial_diff = np.zeros(x.shape[1])
    kernel = model.kern
    
    def personalizedObj(x_diff): # Only returns objective
        x_star = individual + x_diff
        x = np.vstack((x_star, individual))
        y_mu, y_var = model.predict(x, full_cov=True, include_likelihood=False)
        y_mu = y_mu[0][0] - y_mu[1][0]
        y_var = y_var[0][0] + y_var[1][1] - 2.0 * y_var[1][0]
        if (y_var < 0) or (np.linalg.norm(x_diff) < 1e-15):
            y_var = 0
        yvar_root = np.sqrt(y_var)
        # print(y_mu + ZSCORE * yvar_root)
        return(y_mu + ZSCORE * yvar_root)
    
    def personalizedObjGrad(x_diff): # Returns objective and gradient 
        x_star = individual + x_diff
        x = np.vstack((x_star, individual))
        y_mu, y_var = model.predict(x, full_cov=True, include_likelihood=False)
        y_mu = y_mu[0][0] - y_mu[1][0]
        y_var = y_var[0][0] + y_var[1][1] - 2.0 * y_var[1][0]
        if (y_var < 0) or (np.linalg.norm(x_diff) < 1e-15):
            y_var = 0
        yvar_root = np.sqrt(y_var)
        objval = y_mu + ZSCORE * yvar_root
        dmu_dX, dv_dX = my_predictive_gradient(x_star.reshape(1, x_star.size), model, individual)
        dmu_dX = dmu_dX.reshape((dmu_dX.size,))
        dv_dX = dv_dX[0]
        if yvar_root < 1e-4:
            yvar_root = 1
        if yvar_root < 1e-2:
        	yvar_root *= 10.0
        grad = dmu_dX + ZSCORE * (0.5 / yvar_root) * dv_dX
        return objval, grad
    
    negative_per_obj = lambda z: -personalizedObj(z)
    if not jac:
        if constraint_bounds is None:
            opt_res = minimize(negative_per_obj, initial_diff,
                            jac=False,
                            method = 'SLSQP',
                            options={'disp':False})
            diff_opt = opt_res.x
            objective_val = opt_res.fun
        else:
            opt_res = minimize(negative_per_obj, initial_diff,
                            jac=False,
                            method = 'SLSQP',
                            bounds= constraint_bounds, 
                            options={'disp':False})
            diff_opt = opt_res.x
            print(constraint_bounds)
            print('diff-opt:', diff_opt)
            objective_val = opt_res.fun
    else:
        negate_tuple = lambda tup: tuple(-i for i in tup)
        negative_per_obj_grad = lambda w: negate_tuple(personalizedObjGrad(w))
        diff_opt, objective_val = gradDesSoftThresholdBacktrack(negative_per_obj_grad, negative_per_obj,
                                            center=np.zeros(x.shape[1]), guess= initial_diff, 
                                            l=l, constraint_bounds = constraint_bounds,
                                            eta = eta, max_iter = max_iter, 
                                            convergence_thresh = convergence_thresh)
    if objective_val > 0.0:
        return np.zeros(diff_opt.shape[0]), 0.0
    return diff_opt, -objective_val


##### Gradient method for regularized optimization of objective ######

def gradDesSoftThresholdBacktrack(objective_and_grad, objective_nograd, center, guess, l, 
                                 constraint_bounds = None,
                                 eta = 1.0, max_iter = 1e4, 
                                 convergence_thresh = 1e-9):
    '''
    gradient descent + soft-thresholding with backtracking to choose step-size.
    objective_nograd = just compute objective function (for step-size backtracking).
    center = point toward which to regularize. Should = zeros vector for population-shift.
    eta = initial learning rate
    beta = step-size decrease factor
    max_iter = maximum number of iterations to run.
    convergence_thresh = convergence-criterion (stop once improvement in objective falls below convergence_thresh).
    Returns tuple of: (optimal feature transformation , objective-value).
    '''
    if objective_nograd is None:
        objective_nograd = lambda x: objective_and_grad(x)[0]
    
    if constraint_bounds is not None:
        upper_bounds = np.array([w[1] if (w[1] is not None) else float('inf') for w in constraint_bounds])
        lower_bounds = np.array([w[0] if (w[0] is not None) else -float('inf') for w in constraint_bounds])
    
    prev = float('inf')
    diff = guess - center
    iteration = 0
    prev_grad = 0
    while True:
        o, grad = objective_and_grad(diff + center)
        # print('obj=', - o)
        o_noreg = o # objective w/o regularization penalty.
        o += l * np.linalg.norm(diff, ord=1)
        if (o - prev > -convergence_thresh) or (iteration > max_iter):
            if (iteration > max_iter):
                # warn('gradient descent did not converge')
                print('warning: gradient descent did not converge')
            return (diff + center, o_noreg)
        prev = o
        # Backtracking to select stepsize:
        stepsize = eta
        test_o = objective_nograd(diff - stepsize*grad + center)
        while (test_o - o_noreg > -convergence_thresh):
            stepsize /= 2.0
            if stepsize < convergence_thresh:
                return (diff + center, o_noreg)
            test_o = objective_nograd(diff - stepsize*grad + center)
        # print("stepsize="+str(stepsize))
        diff = diff - stepsize*grad # Take gradient step.
        diff = np.sign(diff) * np.maximum(np.abs(diff) - l * stepsize, 0) # soft-threshold
        if constraint_bounds is not None: # project back into constraint-set:
            upper_violations = np.where(diff > upper_bounds)
            lower_violations = np.where(diff < lower_bounds)
            diff[upper_violations] = upper_bounds[upper_violations]
            diff[lower_violations] = lower_bounds[lower_violations]
        iteration += 1
