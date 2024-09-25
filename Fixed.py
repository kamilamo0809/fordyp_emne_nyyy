import matplotlib.pyplot as plt
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from gurobipy import *
from Spotprice import *
import seaborn as sns
'''
• This script solves the problem using the expected value for all scenarios
• To test the script, change the value for: 
        • model.scenario (1, 2 or 3)
        • model.size (a number between 21 and 100)
'''

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
single_scenario = 1  # Choose scenario 1 (you can change this to 2 or 3)
model.scenario = pyo.Set(initialize=[single_scenario])

''' --------------------- Parameters --------------------- '''

# Investment cost of power plant
model.invest = pyo.Param(initialize=44000000)  # NOK

# Fixed power plant capacity
model.plant_cap = pyo.Param(initialize=15)  # Fixed to 50 MW
model.min_plant_cap = pyo.Param(initialize=3.5)  # MW

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
model.line_cap = pyo.Param(initialize=15)  # MW

# Lifetime
model.lifetime = pyo.Param(initialize=60)  # years

# Rate of interest
model.rate = pyo.Param(initialize=0.02)

''' --------------------- Variables --------------------- '''

# Fixed Size of the power plant
model.size = pyo.Param(initialize=3.5)  # NOK

# Dispatch
model.dispatch = pyo.Var(model.time, within=pyo.NonNegativeReals)

''' --------------------- Constraints --------------------- '''

# Min production
def min_production(model, t):
    return (model.dispatch[t] >= 0)
model.min_production = pyo.Constraint(model.time, rule=min_production)

# Max production
def max_production(model, t):
    return (model.dispatch[t] <= model.size)  # Use fixed size
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
#print("Optimal power plant size:", model.size)  # This will always print 50
print("Objective value (income - cost):", model.obj())

# Collect dispatch data for plotting
dispatch_single_scenario = [model.dispatch[i].value for i in range(1, 8761)]
dispatch_single_scenario.sort(reverse=True)

# Plot the dispatch for the single scenario
sns.lineplot(x=list(range(8760)), y=dispatch_single_scenario, label=f'Scenario {single_scenario}', color = "skyblue")

# Add labels and title
plt.xlabel('Time (hours)')
plt.ylabel('Dispatch (MW)')
plt.title(f'Dispatch Over Time for Scenario {single_scenario} (sorted)')

# Show the legend
plt.legend()
plt.show()
