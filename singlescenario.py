import matplotlib.pyplot as plt
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from gurobipy import *
from Spotprice import *
import seaborn as sns

''' ------------------- Model creation ------------------- '''

# Make model
model = pyo.ConcreteModel()

# Define solver
opt = SolverFactory("gurobi", solver_io="python")

# Enable extraction of dual solution
model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

''' ------------------- Sets ------------------- '''

# Time
time = list(range(1, 8761))  # each hour in a year
model.time = pyo.Set(initialize=time)

# Scenario (only one scenario now)
single_scenario = 3  # Choose scenario 1 (you can change this to 2 or 3)
model.scenario = pyo.Set(initialize=[single_scenario])

''' --------------------- Parameters --------------------- '''

# Investment cost of power plant
model.invest = pyo.Param(initialize=85000000)  # NOK

# Max power plant capacity
model.plant_cap = pyo.Param(initialize=100)  # MW

# Variable cost
model.var_cost = pyo.Param(initialize=200)  # NOK/MWh

# Spot prices for the single scenario
df_spot_prices = merge_lists("spotpriser_21.xlsx", "spotpriser_22.xlsx", "spotpriser_23.xlsx")
spot_prices_dict = {
    (time + 1): df_spot_prices.iloc[time, single_scenario - 1]  # Extract prices for the selected scenario
    for time in df_spot_prices.index
}
model.spot = pyo.Param(model.time, initialize=spot_prices_dict)

# Demand for the single scenario
df_demand = consumption_3_scenarios("rye_generation_and_load.csv")
demand_dict = {
    (time + 1): df_demand.iloc[time, single_scenario - 1]  # Extract demand for the selected scenario
    for time in df_demand.index
}
model.demand = pyo.Param(model.time, initialize=demand_dict)

# Line capacity
model.line_cap = pyo.Param(initialize=20)  # MW

# Lifetime
model.lifetime = pyo.Param(initialize=60)  # years

# Rate of interest
model.rate = pyo.Param(initialize=0.02)

''' --------------------- Variables --------------------- '''

# Size of the power plant
model.size = pyo.Var(within=pyo.NonNegativeReals)

# Dispatch
model.dispatch = pyo.Var(model.time, within=pyo.NonNegativeReals)

''' --------------------- Constraints --------------------- '''

# Plant min capacity:
def min_plant_cap(model):
    return model.size >= 0
model.min_plant_cap = pyo.Constraint(rule=min_plant_cap)

# Plant max capacity:
def max_plant_cap(model):
    return model.size <= 100
model.max_plant_cap = pyo.Constraint(rule=max_plant_cap)

# Min production
def min_production(model, t):
    return (model.dispatch[t] >= 0)
model.min_production = pyo.Constraint(model.time, rule=min_production)

# Max production
def max_production(model, t):
    return (model.dispatch[t] <= model.size)
model.max_production = pyo.Constraint(model.time, rule=max_production)

# Power balance
def power_balance1(model, t):
    return model.dispatch[t] - model.demand[t] >= -model.line_cap
model.power_balance1 = pyo.Constraint(model.time, rule=power_balance1)

def power_balance2(model, t):
    return model.dispatch[t] - model.demand[t] <= model.line_cap
model.power_balance2 = pyo.Constraint(model.time, rule=power_balance2)

''' --------------------- Objective function --------------------- '''

# Maximize income for the single scenario
def obj(model):
    outcome = model.invest * model.size * (model.rate / (1 - (1 + model.rate) ** (-model.lifetime)))
    income = sum((model.spot[t] - model.var_cost) * model.dispatch[t] for t in model.time)
    return income - outcome
model.obj = pyo.Objective(rule=obj, sense=pyo.maximize)

''' --------------------- Solve problem --------------------- '''

results = opt.solve(model, load_solutions=True)

# Print results
print("Optimal power plant size:", model.size.value)
print("Objective value (income - cost):", model.obj())

# Collect dispatch data for plotting
dispatch_single_scenario = [model.dispatch[i].value for i in range(1, 8761)]
dispatch_single_scenario.sort(reverse=True)

# Plot the dispatch for the single scenario
sns.lineplot(x=list(range(8760)), y=dispatch_single_scenario, label=f'Scenario {single_scenario}')

# Add labels and title
plt.xlabel('Time (hours)')
plt.ylabel('Dispatch (MW)')
plt.title(f'Dispatch Over Time for Scenario {single_scenario} (sorted)')

# Show the legend
plt.legend()
plt.show()
