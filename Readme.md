Code for the paper: 

J. Mueller, D. Reshef, G. Du, and T. Jaakkola. 
"Learning Optimal Interventions". AISTATS (2017).

Dependencies: 

Our code requires numpy and scipy. 
You must also install the developer branch of the GPy package: http://github.com/SheffieldML/GPy
(At the time this code was developed, there was a bug in the predictive variances and gradients thereof in the non-developer version of GPy)


The main functions you can use to identify beneficial interventions from data are in: 
PersonalizedIntervention.py (for individually-tailored interventions)
PopulationIntervention.py (for global shift or covariate-fixing policies)

A simple example showing the expected data-format and basic usage is given in: Examples.py