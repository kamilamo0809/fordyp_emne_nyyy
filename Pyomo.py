import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from gurobipy import *
from Spotprice import *

''' ------------------- Model creation ------------------- '''

# Make model
model = pyo.ConcreteModel()

# Define solver
opt = SolverFactory("gurobi", solver_io="python")

#Enable extraction of dual solution
model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)


''' ------------------- Sets ------------------- '''

# Scenarios
scenarios = list(range(1,4)) # scenario 1, 2 and 3
print(scenarios)
model.scenario = pyo.Set(initialize = scenarios)

# Time
time = list(range(1, 8761)) # each hour in a year
model.time = pyo.Set(initialize = time)

''' --------------------- Parameters --------------------- '''

# Investment cost of power plant
model.invest = pyo.Param(initialize = 85000000) #NOK

# Max power plant capacity
model.plant_cap = pyo.Param(initialize = 100) #MW

# variable cost
model.var_cost = pyo.Param(initialize = 200) #NOK/MWh

# Spot prices for the three scenarios
# Assuming your merged DataFrame looks like this after calling `merge_lists()`
df_spot_prices = merge_lists("spotpriser_21.xlsx", "spotpriser_22.xlsx", "spotpriser_23.xlsx")
spot_prices_dict = {
    (time+1, i + 1): df_spot_prices.iloc[time, i]
    for time in df_spot_prices.index
    for i in range(df_spot_prices.shape[1])
}
# Now pass this dictionary to the initialize function of Pyomo's Param
model.spot = pyo.Param(model.time, model.scenario, initialize=spot_prices_dict)

# probabilities
model.probabilities = pyo.Param(model.scenario, initialize = {1:0.25, 2:0.6, 3:0.15}) # TODO: sett inn riktige sansynligheter

# Demand

df_demand = read_consumption("rye_generation_and_load.csv")
demand_dict = {
    (time+1, i): df_demand.iloc[time]
    for time in df_spot_prices.index
    for i in scenarios
}

model.demand = pyo.Param(model.time, model.scenario, initialize = demand_dict) # TODO: sett inn riktig funksjon

# Line capacity
model.line_cap = pyo.Param(initialize = 10) #MW

# Lifetime
model.lifetime = pyo.Param(initialize = 60) # years

# rate of interest
model.rate = pyo.Param(initialize = 0.05)

''' --------------------- Variables --------------------- '''

# Size of the power plant
model.size = pyo.Var(within = pyo.NonNegativeReals)

# Dispatch
model.dispatch = pyo.Var(model.scenario, model.time, within = pyo.NonNegativeReals)

''' --------------------- Constraints --------------------- '''

# Plant min capacity:
def min_plant_cap(model):
    return model.size >= 0
model.min_plant_cap = pyo.Constraint(rule = min_plant_cap)

# Plant max capacity:
def max_plant_cap(model):
    return model.size <= 100
model.max_plant_cap = pyo.Constraint(rule = max_plant_cap)

# Min production
def min_production(model, w, t):
    return (model.dispatch[w, t] >= 0)
model.min_production = pyo.Constraint(model.scenario, model.time, rule = min_production)

# Max production
def max_production(model, w, t):
    return (model.dispatch[w, t] <= model.size)
model.max_production = pyo.Constraint(model.scenario, model.time, rule = max_production)

# Annuity

# Power balance
def power_balance1(model, w, t):
    return (-model.line_cap <= model.dispatch[w, t] - model.demand[w, t])
model.power_balance1 = pyo.Constraint(model.scenario, model.time, rule = power_balance1)

def power_balance2(model, w, t):
    return (model.line_cap >= model.dispatch[w, t] - model.demand[w, t])
model.power_balance2 = pyo.Constraint(model.scenario, model.tinme, rule = power_balance2)

''' --------------------- Objective function --------------------- '''

# Maximize income
def obj(model):
 outcome = model.invest * model.size * (model.rate/(1-(1+model.rate)**(-model.lifetime)))
 income = sum(model.probability[w] * sum((model.spot[w, t] - model.var_cost) * model.dispatch[w, t] for t in model.time) for w in model.scenario)
 return (income - outcome)

''' --------------------- Solve problem --------------------- '''

results = opt.solve(model, load_solutions = True)
model.display()
model.dual.display()


