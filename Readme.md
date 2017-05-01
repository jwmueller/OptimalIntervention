Code for the paper: <br/> <br/>
J. Mueller, D. N. Reshef, G. Du, and T. Jaakkola. 
<b><a href="http://proceedings.mlr.press/v54/mueller17a.html">Learning Optimal Interventions</a></b>. <i>AISTATS</i> (2017).

Dependencies: 

Our code requires numpy and scipy. 
You must also install the developer branch of the GPy package: http://github.com/SheffieldML/GPy
(At the time this code was developed, there was a bug in the predictive variances and gradients thereof in the non-developer version of GPy)


The main functions you can use to identify beneficial interventions from data are in: <br/>
[PersonalizedIntervention.py](PersonalizedIntervention.py) (for individually-tailored interventions) <br/>
<i>PopulationIntervention.py</i> (for global shift or covariate-fixing policies)

A simple example showing the expected data-format and basic usage is given in: <i>Examples.py</i>