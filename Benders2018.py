import matplotlib.pyplot as plt
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from Spotprice import *
import seaborn as sns

# Mathematical formulation master problem
def Obj_master(model):
    return - model.invest * model.size * (model.rate / (1 - (1 + model.rate) ** (-model.lifetime))) + model.alpha

def sizeRestriction1(model):
    return model.size <= model.max_plant_cap

def sizeRestriction2(model):
    return model.size >= model.min_plant_cap

def CreateCuts(model, c):
    a = 2
    b = "print"
    print(model.Phi[c])
    return (model.alpha <= model.Phi[c] + model.Lambda[c] * (model.size - model.x_hat[c]))


# Mathematical formulation of subproblem

def Obj_sub(model):
    return sum(model.probabilities[w] * sum((model.spot[t, w] - model.var_cost) * model.dispatch[t, w]
            for t in model.time) for w in model.scenario)


# Min production
def min_production(model, t, w):
    return (model.dispatch[t, w] >= 0)
# Max production
def max_production(model, t, w):
    return (model.dispatch[t, w] <= model.size)
# Power balance
def power_balance1(model, t, w):
    return model.dispatch[t, w] - model.demand[t, w] >= -model.line_cap
def power_balance2(model, t, w):
    return model.dispatch[t, w] - model.demand[t, w] <= model.line_cap
def Size_rule(model):
    return model.size == model.X_hat

# Set up model 1st stage
def ModelSetUp_Master(Cuts):
    ''''''
    
    '''Model creation'''
    master_model = pyo.ConcreteModel()

    '''Parameters'''
    # Power plant capacity
    master_model.max_plant_cap = pyo.Param(initialize = 15)  # MW
    master_model.min_plant_cap = pyo.Param(initialize = 3.5)  # MW
    master_model.invest = pyo.Param(initialize = 44000000)  # NOK
    master_model.lifetime = pyo.Param(initialize = 60)  # years
    master_model.rate = pyo.Param(initialize = 0.02)
    
    '''Variables'''
    # Size of the power plant
    master_model.size = pyo.Var(within = pyo.NonNegativeReals)

    """Cuts_information"""
    # Set for cuts
    master_model.Cut = pyo.Set(initialize = Cuts["Set"])

    # Parameter for cuts
    master_model.Phi = pyo.Param(master_model.Cut, initialize = Cuts["Phi"])
    master_model.Lambda = pyo.Param(master_model.Cut,initialize = Cuts["lambda"])
    master_model.x_hat = pyo.Param(master_model.Cut, initialize = Cuts["x_hat"])

    # Variable for alpha
    master_model.alpha = pyo.Var(bounds = (0, 1000000000))

    """Constraint cut"""
    master_model.CreateCuts = pyo.Constraint(master_model.Cut, rule = CreateCuts)

    """Constraints"""
    master_model.sizeRestriction1 = pyo.Constraint(rule = sizeRestriction1)
    master_model.sizeRestriction2 = pyo.Constraint(rule = sizeRestriction2)

    # Define objective function
    master_model.obj = pyo.Objective(rule = Obj_master, sense = pyo.maximize)

    return master_model


# Set up model 2nd stage
def ModelSetUp_subproblem(X_hat):
    ''''''
    '''Model creation'''
    sub_model = pyo.ConcreteModel()
    
    '''Define sets'''
    sub_model.scenario = pyo.Set(initialize = list(range(1,4)))
    sub_model.time = pyo.Set(initialize = list(range(1, 8761)))

    '''Define parameters'''
    sub_model.X_hat = pyo.Param(initialize = X_hat)

    # variable cost
    sub_model.var_cost = pyo.Param(initialize = 200)  # NOK/MWh

    # Spot prices for the three scenarios
    df_spot_prices = merge_lists("spotpriser_21.xlsx", "spotpriser_22.xlsx", "spotpriser_23.xlsx")
    spot_prices_dict = {(time + 1, i + 1): df_spot_prices.iloc[time, i] for time in df_spot_prices.index for i in
        range(df_spot_prices.shape[1])}
    sub_model.spot = pyo.Param(sub_model.time, sub_model.scenario, initialize = spot_prices_dict)

    # probabilities
    sub_model.probabilities = pyo.Param(sub_model.scenario,
                                    initialize = {1: 0.2, 2: 0.7, 3: 0.1})  # TODO: sett inn riktige sansynligheter

    # Demand
    df_demand = consumption_3_scenarios("rye_generation_and_load.csv")
    demand_dict = {(time + 1, i + 1): df_demand.iloc[time, i] for time in df_demand.index for i in
        range(df_demand.shape[1])}
    sub_model.demand = pyo.Param(sub_model.time, sub_model.scenario, initialize = demand_dict)

    # Line capacity
    sub_model.line_cap = pyo.Param(initialize = 5)  # MW

    '''Define variables'''
    # Dispatch
    sub_model.dispatch = pyo.Var(sub_model.time, sub_model.scenario, within = pyo.NonNegativeReals)
    sub_model.size = pyo.Var(within = pyo.NonNegativeReals)

    '''Define constraints'''
    sub_model.min_prod = pyo.Constraint(sub_model.time, sub_model.scenario, rule = min_production)
    sub_model.max_prod = pyo.Constraint(sub_model.time, sub_model.scenario, rule = max_production)
    sub_model.pow_bal1 = pyo.Constraint(sub_model.time, sub_model.scenario, rule = power_balance1)
    sub_model.pow_bal2 = pyo.Constraint(sub_model.time, sub_model.scenario, rule = power_balance2)
    sub_model.size_rule = pyo.Constraint(rule = Size_rule)

    # Define objective function
    sub_model.obj = pyo.Objective(rule = Obj_sub, sense = pyo.maximize)

    return sub_model


def Solve(model):
    opt = SolverFactory("gurobi", solver_io="python")
    model.dual = pyo.Suffix(direction = pyo.Suffix.IMPORT)
    results = opt.solve(model, load_solutions = True)
    return results, model


def DisplayResults(model):
    return print(model.display(), model.dual.display())


# Function for creating new linear cuts for optimization problem
def Cut_manage(Cuts, model):
    """Add new cut to existing dictionary of cut information"""

    # Find cut iteration by checking number of existing cuts
    cut = len(Cuts["Set"])
    # Add new cut to list, since 0-index is a thing this works well
    Cuts["Set"].append(cut)

    # Find 2nd stage cost result
    Cuts["Phi"][cut] = pyo.value(model.obj)
    # Find lambda x_hat
    Cuts["lambda"][cut] = model.dual[model.size_rule]
    Cuts["x_hat"][cut] = model.X_hat
    return Cuts


"""
Setup for benders decomposition
We perform this for x iterations
"""
# Pre-step: Formulate cut input data
Cuts = {}
Cuts["Set"] = []
Cuts["Phi"] = {}
Cuts["lambda"] = {}
Cuts["x_hat"] = {}

# This is the while-loop in principle, but for this case is only a for-loop
for i in range(10):

    # Solve 1st stage problem
    master = ModelSetUp_Master(Cuts)
    Solve(master)

    # Process 1st stage result
    X_hat = pyo.value(master.size)

    # Print results for master problem
    print(f"Iteration {i}, size = {X_hat}")

    # Setup and solve sub stage problem
    sub = ModelSetUp_subproblem(X_hat)
    Solve(sub)

    # Create new cuts for master stage problem
    Cuts = Cut_manage(Cuts, sub)

    # Print results 2nd stage
    print("Objective function:", round(pyo.value(sub.obj), 2)/1000000, " MNOK")
    print("Cut information acquired:")
    for component in Cuts:
        print(component, Cuts[component])

    # We perform a convergence check
    print("UB:", pyo.value(master.alpha.value), "- LB:", pyo.value(sub.obj))


