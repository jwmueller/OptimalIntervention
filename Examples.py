''' Simple example for identifying personalized and population interventions from data
'''
from PersonalizedIntervention import *
from PopulationIntervention import *
from simulation import *

# Create a dataset with 5 covariates, 30 samples, with a noise-level of 0.1 in the outcome-covariate relationship: 
# Data come from quadratic relationship Y = 1 - X_4^2 - X_5^2 + epsilon
# Outcome only depends on the final two covariates.
d = 5; n = 30; noise = 0.1
simulation_func, simulation_eval, true_opt, correct_dims = paraboloid
X, y = simulation_func(n,d,noise) # the dataset: X = covariates, y = outcomes.
print(X.shape)
print(y.shape)

# Fit GP regression model:
kernel = GPy.kern.RBF(d, ARD=True) # Use Automatic-Relevance-Determination to allow kernel to focus on important covariates.
model = GPy.models.GPRegression(X, y, kernel)
model.optimize(max_iters=1e4) # Find good hyperparameters of ARD kernel by maximizing marginal likelihood.
print(model.kern.lengthscale) # Shows how each covariate is weighted by ARD kernel (lower = more important). 
print(model)

# New individual we wish to identify personalized intervention for:
x_new = np.ones(d)

# Find optimal shift intervention personally tailored for x_new:
constraints = [(-2,2) for i in range(d)] # we are only allowed to shift each covariate by at most 2.
num_intervened = 2 # we are only allowed to intervene on at most 2 covariates.
personal_int =  personalizedIntervention(x_new, X, model, cardinality = num_intervened, constraint_bounds=constraints, smoothing_levels = (4,3,2,1))
print(personal_int) # A tuple (optimal_shift, optimal_objective_value)
# optimal_shift= desired shift in covariates of x_new corresponding to optimal intervention.
# optimal_objective_value= model is confident that intervention will improve expected outcome by at least this much.


# Find optimal shift population intervention:
pop_shift_int = sparsePopulationShift(X, model, cardinality = num_intervened, constraint_bounds=constraints, smoothing_levels = (3,1))
print(pop_shift_int)

# Find optimal covariate-fixing population intervention:
pop_covfix_int =  sparsePopulationUniform(X, model, cardinality = num_intervened, constraint_bounds=constraints, smoothing_levels = (3,1))
print(pop_covfix_int) # Masked array specifies which covariates should be fixed to which values.

