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

##### Find optimal Covariate-Fixing populatation intervention #####
def sparsePopulationUniform(X, model, cardinality = None,
                          constraint_bounds=None,
                          smoothing_levels = (), quantile = 0.05, 
                          eta = 1.0, max_iter = 1e4, convergence_thresh = 1e-9):
    '''
    Returns tuple: (optimal_uniform, optimal_objective_value).
    optimal_uniform = vector of masked entries except for those which should be set to the certain specified values.
    model = fitted GP model to training (X, y) pairs.
    cardinality = number of variables that can be intervened upon, None if there is no constraint here.
    constraint_bounds should be list of tuples, one for each dimension indicating the min/max setting allowed, None if there is no constraint.  Set = (0,0) to indicate a feature which is fixed and cannot be transformed. Set = (None, None) for a feature which is unconstrained.
    smoothing_levels = tuple containing different amounts of smoothing to try out, None if there is no smoothing to be performed.
    quantile = which quantile of posterior-gain to use (0.05 by default).
    eta, max_iter, convergence_thresh = parameters for gradient method.
    '''
    d = X.shape[1] # number of features.
    if cardinality is None:
        return(smoothedPopulationUniform(X, model, int_indices=range(d), 
                    smoothing_levels = smoothing_levels, constraint_bounds=constraint_bounds,quantile=quantile))
    int_indices = []
    remaining_features = range(d)
    for iter in range(cardinality): # use greedy forward selection to select intervention indices:
        current_best = 0.0
        for feat in remaining_features:
            test_int_indices = list(int_indices) # new copy
            test_int_indices.append(feat)
            test_int, test_objval = smoothedPopulationUniform(X, model, 
                                        int_indices=test_int_indices, smoothing_levels=smoothing_levels,
                                        constraint_bounds=constraint_bounds, quantile=quantile)
            #print(feat, test_objval, test_int, int_indices)
            if test_objval > current_best:
                current_best = test_objval
                best_feature = feat
        #print(best_feature)
        int_indices.append(best_feature)
        remaining_features.remove(best_feature)
    return(smoothedPopulationUniform(X, model, 
                int_indices=int_indices, smoothing_levels=smoothing_levels,
                            constraint_bounds=constraint_bounds, quantile=quantile))

def smoothedPopulationUniform(X, model, int_indices = None, smoothing_levels = (),
                            constraint_bounds=None, quantile = 0.05, initial_diff = None):
    '''
    Performs smoothing + optimization (continuation approach for dealing with nonconvexity).
    eta, max_iter, convergence_thresh = parameters for gradient method.
    '''
    current_guess = initial_diff
    if (len(smoothing_levels) >= 1):  # Perform Smoothing (only works for standard GP regression with ARD kernel):
        smoothing_levels = sorted(smoothing_levels, reverse = True)
        orig_lengthscale = model.kern.lengthscale.copy()
        orig_variance = model.kern.variance.copy()
        orig_noise_var = model.likelihood.variance.copy()
        if initial_diff is None: # Set initial guess to zero-shift.
            initial_diff = np.zeros(X.shape[1])
        max_features = np.amax(X, axis=0)
        min_features = np.amin(X, axis=0)
        # Add additional constraints during smoothing to ensure we don't go beyond data range:
        if constraint_bounds is not None:
            upper_bounds = np.array([w[1] if (w[1] is not None) else float('inf') for w in constraint_bounds])
            lower_bounds = np.array([w[0] if (w[0] is not None) else -float('inf') for w in constraint_bounds])
            smoothing_constraints = [(max(min_features[i],lower_bounds[i]), max(min_features[i],max(lower_bounds[i],min(max_features[i],upper_bounds[i])))) for i in range(X.shape[1])]
        else:
            smoothing_constraints = [(min_features[i], max_features[i]) for i in range(X.shape[1])]
        for smooth_amt in smoothing_levels:
            if smooth_amt > 1.0:
                # print("Smooth_amt="+str(smooth_amt))
                model.kern.lengthscale = smooth_amt * orig_lengthscale
                model.kern.lengthscale.fix()
                model.optimize() # Recompute variance and noise scale.
                # perform optimization with more aggressive parameter settings:
                current_guess, objval = populationUniformOptimization(X, model,
                                            int_indices = int_indices, constraint_bounds=smoothing_constraints,
                                            quantile=quantile, initial_diff = current_guess)
                # print(current_guess)
        # Restore original model:
        model.kern.lengthscale = orig_lengthscale
        model.kern.variance = orig_variance
        model.likelihood.variance = orig_noise_var
        # print("Smooth_amt= 1")
    return (populationUniformOptimization(X, model, int_indices = int_indices,
                constraint_bounds=constraint_bounds, quantile=quantile, initial_diff = current_guess))

def populationUniformOptimization(x, model, int_indices = None,
                    constraint_bounds=None, quantile = 0.05, initial_diff = None):
    '''
    Initial guess for optimal shift.
    Finds local optimum of population intervention objective,
    int_indices = which dimensions can be set to uniform constants (indices from 0 to d-1)
    Returns tuple: (optimal_shift, optimal_objective_value)
    '''
    n, d = x.shape
    if int_indices is None:
        int_indices = range(d)
    mask = [0 if i in int_indices else 1 for i in range(d)]
    if initial_diff is None: # Set initial guess to be uniform intervention where nothing is set.
        feature_means = np.mean(x, axis = 0)
        initial_diff = ma.masked_array(feature_means, mask=mask)
    ZSCORE = scipy.stats.norm.ppf(quantile)
    kernel = model.kern
    
    def populationMaskedUniformObj(unif_intervention): # Only returns objective (when unif_intervention is masked vector)
        n, d = x.shape
        transformed_pop = x.copy()
        if len(int_indices) > 0:
            transformed_pop[:, int_indices] = unif_intervention[int_indices] # set values according to uniform intervention.
            # print(transformed_pop[range(3),1])
        else:
            return 0.0
        all_pts = np.vstack((x, transformed_pop))
        y_mus, y_cov = model.predict(all_pts, full_cov=True, include_likelihood=False)
        normal_mean = np.sum(y_mus[range(n,2*n)] - y_mus[range(n)]) / n
        normal_var = (np.sum(y_cov[n:(2*n),n:(2*n)]) + np.sum(y_cov[0:n,0:n]) - np.sum(y_cov[n:(2*n),0:n]) - np.sum(y_cov[0:n,n:(2*n)])) / (n**2)
        if (normal_var < 0.0) or (np.linalg.norm(transformed_pop - x) < 1e-14):
            normal_var = 0.0
        obj = normal_mean + ZSCORE * np.sqrt(normal_var)
        return(obj)
    
    negative_pop_obj = lambda z: -populationMaskedUniformObj(z)
    if constraint_bounds is None:
        opt_res = minimize(negative_pop_obj, initial_diff,
                        jac=False,
                        method = 'SLSQP',
                        options={'disp':False})
        diff_opt = opt_res.x
        objective_val = opt_res.fun
    else:
        opt_res = minimize(negative_pop_obj, initial_diff,
                        jac=False,
                        method = 'SLSQP',
                        bounds= constraint_bounds, 
                        options={'disp':False})
        diff_opt = opt_res.x
        objective_val = opt_res.fun
    unif_opt = ma.masked_array(diff_opt, mask=mask)
    # print(unif_opt)
    return unif_opt, abs(objective_val)


##### Find optimal SHIFT populatation intervention ######
def sparsePopulationShift(X, model,
                          cardinality = None,
                          constraint_bounds=None,
                          smoothing_levels = (), 
                          quantile = 0.05, l = 2.0,
                          eta = 1.0, max_iter = 1e4, convergence_thresh = 1e-9):
    '''
    Runs smoothed version of shift population intervention 
    with a sparsity constraint, using gradient descent + soft-threholding
    and binary search to find proper lambda regularization.
    Returns tuple: (optimal_shift, optimal_objective_value).
    model = fitted GP model to training (X, y) pairs.
    cardinality = number of variables that can be intervened upon, None if there is no constraint
    constraint_bounds should be list of tuples, one for each dimension indicating the min/max SHIFT allowed, None if there is no constraint.  Set = (0,0) to indicate a feature which is fixed and cannot be transformed. Set = (None, None) for a feature which is unconstrained.
    smoothing_levels = tuple containing different amounts of smoothing to try out, None if there is no smoothing to be performed.
    quantile = which quantile of posterior-gain to use (0.05 by default).
    l = initial maximal regularization penalty to try.
    eta, max_iter, convergence_thresh = parameters for gradient method.
    d = X.shape[1] # number of features.
    '''
    # First run without penalty (to get initial solns):
    init_diff, obj_val = smoothedPopulationShift(X, model, smoothing_levels=smoothing_levels, 
                            constraint_bounds=constraint_bounds, quantile=quantile, l=0.0)
    print("init_diff:",init_diff)
    current_cardinality = X.shape[1] - sum(np.isclose(init_diff, np.zeros(X.shape[1])))
    if (cardinality is None) or (cardinality >= X.shape[1]) or (current_cardinality <= cardinality):
        return (init_diff, obj_val)
    if cardinality <= 0:
        return (np.zeros(X.shape[1]), 0.0)
    l = l / 2.0 # initial_max
    while (current_cardinality > cardinality): # need more regularization.
        l *= 2.0
        print("Upper l search: Trying l="+ str(l))
        opt_diff, obj_val = smoothedPopulationShift(X, model, smoothing_levels=smoothing_levels,
                                constraint_bounds=constraint_bounds, quantile=quantile,
                                l=l, initial_diff=init_diff, eta = eta, max_iter=max_iter, 
                                convergence_thresh=convergence_thresh)
        print("diff:",opt_diff)
        current_cardinality = X.shape[1] - sum(np.isclose(opt_diff, np.zeros(X.shape[1])))
        print(opt_diff)
        print("Current cardinality= " + str(current_cardinality))
        print(np.isclose(current_cardinality, cardinality))
    print("Found upper lambda, l=" + str(l))
    # Perform binary search to find right amount of regularization:
    ub = l
    lb = 0
    iteration = 0
    while not np.isclose(current_cardinality, cardinality):
        if iteration >= 20: # don't try more than 20 times, simply chose the largest nonzero terms in this case.
            break
        l = (lb + ub) / 2
        print("Binary search: Trying l="+ str(l))
        opt_diff, obj_val = smoothedPopulationShift(X, model, smoothing_levels=smoothing_levels,
                                constraint_bounds=constraint_bounds, quantile=quantile, l=l, initial_diff=init_diff,
                                eta=eta, max_iter=max_iter, convergence_thresh=convergence_thresh)
        current_cardinality = X.shape[1] - sum(np.isclose(opt_diff, np.zeros(X.shape[1])))
        print("diff:",opt_diff)
        print("Current cardinality= " + str(current_cardinality))
        if current_cardinality > cardinality:
            lb = l
        elif current_cardinality < cardinality:
            ub = l
        iteration += 1
    selected_features = np.where(np.logical_not(np.isclose(opt_diff, np.zeros(X.shape[1]))))[0] # features which are chosen for optimization.
    # print("selected_features:"+str(selected_features)+ "  length="+str(len(selected_features)))
    if not np.isclose(len(selected_features), cardinality):
        print("warning: could not achieve target cardinality")
        ordering = np.argsort(-abs(opt_diff))
        selected_features = ordering[range(cardinality)] # simply choose top entries in this failure case.
        opt_diff[[z for z in range(X.shape[1]) if z not in selected_features]] = 0.0
    # Modify constraint_bounds to ensure not-selected features remain fixed at 0.
    if constraint_bounds is not None:
        print("a selected_features:"+str(selected_features))
        sparse_constraints = [constraint_bounds[i] if (i in selected_features) else (0,0) for i in range(X.shape[1])]
    else:
        print("b selected_features:"+str(selected_features))
        sparse_constraints = [(None, None) if (i in selected_features) else (0,0) for i in range(X.shape[1])]
    return(smoothedPopulationShift(X, model, smoothing_levels=smoothing_levels,
                                constraint_bounds=sparse_constraints, quantile=quantile, l=0.0, initial_diff = opt_diff,
                                eta = eta, max_iter = max_iter, convergence_thresh = convergence_thresh))



def smoothedPopulationShift(X, model, smoothing_levels = (),
                            constraint_bounds=None, quantile=0.05, l = 0.0, initial_diff = None,
                            eta = 1.0, max_iter = 1e4, convergence_thresh = 1e-9):
    '''
    Performs smoothing + optimization (continuation approach for dealing with nonconvexity)
    eta, max_iter, convergence_thresh = parameters for gradient method.
    '''
    current_guess = initial_diff 
    if (len(smoothing_levels) >= 1):  # Perform Smoothing (only works for standard GP regression with ARD kernel):
        smoothing_levels = sorted(smoothing_levels, reverse = True)
        orig_lengthscale = model.kern.lengthscale.copy()
        orig_variance = model.kern.variance.copy()
        orig_noise_var = model.likelihood.variance.copy()
        if initial_diff is None: # Set initial guess to zero-shift.
            initial_diff = np.zeros(X.shape[1])
        max_features = np.amax(X, axis=0)
        min_features = np.amin(X, axis=0)
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
                current_guess, objval = populationShiftOptimization(X, model,
                                            constraint_bounds=smoothing_constraints,
                                            quantile=quantile, l=l, initial_diff=current_guess, 
                                            eta = eta*5, max_iter=ceil(max_iter/10.0), 
                                            convergence_thresh=convergence_thresh * 10)
                # print(current_guess)
        # Restore original model:
        model.kern.lengthscale = orig_lengthscale
        model.kern.variance = orig_variance
        model.likelihood.variance = orig_noise_var
        # print("Smooth_amt= 1")
    return (populationShiftOptimization(X, model, constraint_bounds=constraint_bounds, 
                quantile=quantile, l=l, initial_diff=current_guess, eta=eta, max_iter=max_iter, 
                convergence_thresh=convergence_thresh))

def populationShiftOptimization(x, model,
                    constraint_bounds=None, quantile = 0.05, l = 0.0, initial_diff = None, 
                    eta = 1.0, max_iter = 1e4, convergence_thresh = 1e-9):
    '''
    Initial guess for optimal shift.
    Finds local optimum of population intervention objective, 
    for a given regularizer l and a given level of smoothness specified in the 'model' object.
    eta, max_iter, convergence_thresh = parameters for gradient method.
    Returns tuple: (optimal_shift, optimal_objective_value)
    '''
    if np.isclose(l, 0.0):
        jac = False
    else: 
        jac = True # To use gradient ascent.
    if initial_diff is None: # Set initial guess to zero-shift.
        initial_diff = np.zeros(x.shape[1])
    ZSCORE = scipy.stats.norm.ppf(quantile)
    kernel = model.kern
    
    def populationShiftObj(x_diff): # Only returns objective
        n, d = x.shape
        transformed_pop = x + np.tile(x_diff, (len(x), 1))
        all_pts = np.vstack((x, transformed_pop))
        y_mus, y_cov = model.predict(all_pts, full_cov=True, include_likelihood=False)
        normal_mean = np.sum(y_mus[range(n,2*n)] - y_mus[range(n)]) / n
        normal_var = (np.sum(y_cov[n:(2*n),n:(2*n)]) + np.sum(y_cov[0:n,0:n]) - np.sum(y_cov[n:(2*n),0:n]) - np.sum(y_cov[0:n,n:(2*n)])) / (n**2)
        #normal_var = 0.0 # Slow version w loop
        #for i in range(n): 
        #    for j in range(n):
        #        term = y_cov[i+n,j+n] + y_cov[i,j] - y_cov[i+n,j] - y_cov[i,j+n]
        #        normal_var += term
        #normal_var = normal_var / (n ** 2)
        if (normal_var < 0.0) or (np.linalg.norm(x_diff) < 1e-14):
            normal_var = 0.0
        obj = normal_mean + ZSCORE * np.sqrt(normal_var)
        return(obj)
    
    def populationShiftObjGrad(x_diff): # Returns objective and gradient 
        n, d = x.shape
        transformed_pop = x + np.tile(x_diff, (len(x), 1))
        all_pts = np.vstack((x, transformed_pop))
        y_mus, y_cov = model.predict(all_pts, full_cov=True, include_likelihood=False)
        normal_mean = np.sum(y_mus[range(n,2*n)] - y_mus[range(n)]) / n
        normal_var = (np.sum(y_cov[n:(2*n),n:(2*n)]) + np.sum(y_cov[0:n,0:n]) - np.sum(y_cov[n:(2*n),0:n]) - np.sum(y_cov[0:n,n:(2*n)])) / (n**2)
        if (normal_var < 0.0) or (np.linalg.norm(x_diff) < 1e-14):
            normal_var = 0.0
        obj = normal_mean + ZSCORE * np.sqrt(normal_var)
        # Compute gradient (only works for stationary kernel):
        grad = np.zeros(d)
        mean_grad = np.zeros(d)
        mean_jac = np.empty((1,d,model.output_dim))
        for j in range(n):
            # grad_noise = np.random.uniform(low = 0.1, high=0.1, size=d)
            x_new = (x[j] + x_diff).reshape(1, d)
            for i in range(model.output_dim):
                mean_jac[:,:,i] = kernel.gradients_X(model.posterior.woodbury_vector[:,i:i+1].T, x_new, model._predictive_variable)
                # print(mean_grad)
            mean_grad += mean_jac.reshape((mean_jac.size,)) / n
        
        var_grad = np.zeros(d)
        for i in range(n):
            test_point_i = x[i].reshape(1, d)
            x_new_i = test_point_i + x_diff.reshape(1, d)
            for j in range(n):
                test_point_j = x[j].reshape(1, d)
                x_new_j = test_point_j + x_diff.reshape(1, d)
                if (i != j):
                    # Gradient wrt Cov(T(x_i),T(x_j)) (when i != j):
                    #grads wrt 'Schur' part K_{xf}K_{ff}^{-1}K_{fx} :
                    alpha = -np.dot(kernel.K(x_new_i, model._predictive_variable), model.posterior.woodbury_inv)
                    dv_dX = kernel.gradients_X(alpha, x_new_j, model._predictive_variable)
                    var_grad += dv_dX[0]
                    alpha = -np.dot(kernel.K(x_new_j, model._predictive_variable), model.posterior.woodbury_inv)
                    dv_dX = kernel.gradients_X(alpha, x_new_i, model._predictive_variable)
                    var_grad += dv_dX[0]
                    
                    # Gradient wrt Cov(T(x_i),x_j) (when i != j):
                    dC_dX = kernel.gradients_X(x_new_i.shape[0], x_new_i, test_point_j)
                    alpha = -np.dot(kernel.K(test_point_j, model._predictive_variable), model.posterior.woodbury_inv)
                    dC_dX += kernel.gradients_X(alpha, x_new_i, model._predictive_variable)
                    var_grad -= dC_dX[0] 
                    # Gradient wrt Cov(x_i,T(x_j)) (when i != j):
                    dC_dX = kernel.gradients_X(x_new_j.shape[0], x_new_j, test_point_i)
                    alpha = -np.dot(kernel.K(test_point_i, model._predictive_variable), model.posterior.woodbury_inv)
                    dC_dX += kernel.gradients_X(alpha, x_new_j, model._predictive_variable)
                    var_grad -= dC_dX[0]
                else:
                    # gradients wrt Cov(T(x_i),T(x_i)) (when i = j):
                    dv_dX = kernel.gradients_X(np.eye(x_new_i.shape[0]), x_new_i)
                    #grads wrt 'Schur' part K_{xf}K_{ff}^{-1}K_{fx}
                    alpha = -2.*np.dot(kernel.K(x_new_i, model._predictive_variable), model.posterior.woodbury_inv)
                    dv_dX += kernel.gradients_X(alpha, x_new_i, model._predictive_variable)
                    var_grad += dv_dX[0]
                    # Gradient w.r.t. Cov(T(x_i), x_i) (when i = j):
                    dC_dX = kernel.gradients_X(x_new_i.shape[0], x_new_i, test_point_i)
                    alpha = -np.dot(kernel.K(test_point_i, model._predictive_variable), model.posterior.woodbury_inv)
                    dC_dX += kernel.gradients_X(alpha, x_new_i, model._predictive_variable)
                    var_grad -= 2.*dC_dX[0]
        var_grad = var_grad / (n ** 2)
        var_sqrt = np.sqrt(normal_var)
        # Clip the gradient by lower-bounding var_sqrt term:
        if var_sqrt < 1e-6: 
            var_sqrt = 1
        if var_sqrt < 1e-3:
            var_sqrt *= 10
        grad = mean_grad + ZSCORE * (0.5 / var_sqrt) * var_grad
        print("pop-obj=" + str(obj))
        print("x_diff:",x_diff)
        return obj , grad # End of helper functions.
    
    negative_pop_obj = lambda z: -populationShiftObj(z)
    if not jac:
        if constraint_bounds is None:
            opt_res = minimize(negative_pop_obj, initial_diff,
                            jac=False,
                            method = 'SLSQP',
                            options={'disp':False})
            diff_opt = opt_res.x
            objective_val = opt_res.fun
        else:
            opt_res = minimize(negative_pop_obj, initial_diff,
                            jac=False,
                            method = 'SLSQP',
                            bounds= constraint_bounds, 
                            options={'disp':False})
            diff_opt = opt_res.x
            objective_val = opt_res.fun
    else:
        negate_tuple = lambda tup: tuple(-i for i in tup)
        negative_pop_obj_grad = lambda w: negate_tuple(populationShiftObjGrad(w))
        diff_opt, objective_val = gradDesSoftThresholdBacktrack(negative_pop_obj_grad, negative_pop_obj,
                                            center=np.zeros(x.shape[1]), guess= initial_diff, 
                                            l=l, constraint_bounds = constraint_bounds,
                                            eta = eta, max_iter = max_iter, 
                                            convergence_thresh = convergence_thresh)
    return diff_opt, abs(objective_val)


##### Gradient method for regularized optimization ######

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
