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

#Enable extraction of dual solution
model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)


''' ------------------- Sets ------------------- '''

# Scenarios
scenarios = list(range(1,4)) # scenario 1, 2 and 3
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
model.probabilities = pyo.Param(model.scenario, initialize = {1:0.2, 2:0.7, 3:0.1}) # TODO: sett inn riktige sansynligheter

# Demand
df_demand = consumption_3_scenarios("rye_generation_and_load.csv")

demand_dict = {
    (time + 1, i + 1): df_demand.iloc[time, i]
    for time in df_demand.index
    for i in range(df_demand.shape[1])}

model.demand = pyo.Param(model.time, model.scenario, initialize=demand_dict)

# Line capacity
model.line_cap = pyo.Param(initialize = 50) #MW

# Lifetime
model.lifetime = pyo.Param(initialize = 60) # years

# rate of interest
model.rate = pyo.Param(initialize = 0.02)

''' --------------------- Variables --------------------- '''

# Size of the power plant
model.size = pyo.Var(within = pyo.NonNegativeReals)

# Dispatch
model.dispatch = pyo.Var(model.time, model.scenario, within = pyo.NonNegativeReals)

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
def min_production(model, t, w):
    return (model.dispatch[t, w] >= 0)
model.min_production = pyo.Constraint(model.time, model.scenario, rule = min_production)

# Max production
def max_production(model, t, w):
    return (model.dispatch[t, w] <= model.size)
model.max_production = pyo.Constraint(model.time, model.scenario, rule = max_production)

# Annuity

# Power balance
def power_balance1(model, t, w):
    return model.dispatch[t, w] - model.demand[t, w] >= -model.line_cap
model.power_balance1 = pyo.Constraint(model.time, model.scenario, rule=power_balance1)

def power_balance2(model, t, w):
    return model.dispatch[t, w] - model.demand[t, w] <= model.line_cap
model.power_balance2 = pyo.Constraint(model.time, model.scenario, rule=power_balance2)

''' --------------------- Objective function --------------------- '''

# Maximize income
def obj(model):
    outcome = model.invest * model.size * (model.rate / (1 - (1 + model.rate) ** (-model.lifetime)))
    income = sum(
        model.probabilities[w] * sum((model.spot[t, w] - model.var_cost) * model.dispatch[t, w] for t in model.time)
        for w in model.scenario
    )
    return income - outcome
model.obj = pyo.Objective(rule=obj, sense=pyo.maximize)

''' --------------------- Solve problem --------------------- '''

results = opt.solve(model, load_solutions = True)
#model.display()
#model.dual.display()

print(model.size.value)
print(model.obj())

sns.set_palette("Set2")
set2_palette = sns.color_palette("Set2")

dispatch_scenario1 = []
dispatch_scenario2 = []
dispatch_scenario3 = []
for i in range(1, 8761):
    dispatch_scenario1.append(model.dispatch[i, 1].value)
    dispatch_scenario2.append(model.dispatch[i, 2].value)
    dispatch_scenario3.append(model.dispatch[i, 3].value)

dispatch_scenario1.sort(reverse = True)
dispatch_scenario2.sort(reverse = True)
dispatch_scenario3.sort(reverse = True)

# Plot all three scenarios in one graph
sns.lineplot(x=list(range(8760)), y=dispatch_scenario1, label='Scenario 1', color = "hotpink")
sns.lineplot(x=list(range(8760)), y=dispatch_scenario2, label='Scenario 2', color = "skyblue")
sns.lineplot(x=list(range(8760)), y=dispatch_scenario3, label='Scenario 3', color = "orange")

# Add labels and title
plt.xlabel('Time (hours)')
plt.ylabel('Dispatch (MW)')
plt.title('Dispatch Over Time for Different Scenarios (sorted)')

# Show the legend
plt.legend()
plt.show()


