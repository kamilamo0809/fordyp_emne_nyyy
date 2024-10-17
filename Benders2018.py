import matplotlib.pyplot as plt
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from Spotprice import *
import seaborn as sns

''' --------------------------- Master problem --------------------------- '''

def ModelSetUp_Master(Cuts):
    '''
    This function set up the master problem based on the previously obtained cuts
    :param Cuts: List of prevously obtained cuts
    :return: master model
    '''
    
    #Model creation
    master_model = pyo.ConcreteModel()

    #Parameters
    master_model.max_plant_cap = pyo.Param(initialize = 15)  # MW
    master_model.min_plant_cap = pyo.Param(initialize = 3.5)  # MW
    master_model.invest = pyo.Param(initialize = 44000000)  # NOK
    master_model.lifetime = pyo.Param(initialize = 60)  # years
    master_model.rate = pyo.Param(initialize = 0.02)
    
    #Variables
    master_model.size = pyo.Var(within = pyo.NonNegativeReals)

    # Cuts
    master_model.Cut = pyo.Set(initialize = Cuts["Set"])
    master_model.Phi = pyo.Param(master_model.Cut, initialize = Cuts["Phi"])
    master_model.Lambda = pyo.Param(master_model.Cut,initialize = Cuts["lambda"])
    master_model.size_hat = pyo.Param(master_model.Cut, initialize = Cuts["size_hat"])

    # Variable for alpha
    master_model.alpha = pyo.Var(bounds = (0, 1000000000))

    # Constraint cut
    def CreateCuts(model, c):
        print(model.Phi[c])
        return (model.alpha <= model.Phi[c] + model.Lambda[c] * (model.size - model.size_hat[c]))
    master_model.CreateCuts = pyo.Constraint(master_model.Cut, rule = CreateCuts)

    # Constraints
    def sizeRestriction1(model):
        return model.size <= model.max_plant_cap
    master_model.sizeRestriction1 = pyo.Constraint(rule = sizeRestriction1)
    def sizeRestriction2(model):
        return model.size >= model.min_plant_cap
    master_model.sizeRestriction2 = pyo.Constraint(rule = sizeRestriction2)

    # Objective function
    def Obj_master(model):
        return - model.invest * model.size * (model.rate / (1 - (1 + model.rate)**(-model.lifetime))) + model.alpha
    master_model.obj = pyo.Objective(rule = Obj_master, sense = pyo.maximize)

    return master_model


''' ---------------------------- Sub problem ---------------------------- '''
def ModelSetUp_subproblem(size_hat):
    '''
    This function set up the subproblem based on the plant size given from the master function
    :param size_hat: Plant size obtained by prevoius master problem iteration
    :return: subproblem
    '''

    # Model creation
    sub_model = pyo.ConcreteModel()
    
    # Sets
    sub_model.scenario = pyo.Set(initialize = list(range(1,4)))
    sub_model.time = pyo.Set(initialize = list(range(1, 8761)))

    #Parameters
    sub_model.size_hat = pyo.Param(initialize = size_hat)

    sub_model.var_cost = pyo.Param(initialize = 200)  # NOK/MWh

    df_spot_prices = merge_lists("spotpriser_21.xlsx", "spotpriser_22.xlsx", "spotpriser_23.xlsx")
    spot_prices_dict = {(time + 1, i + 1): df_spot_prices.iloc[time, i] for time in df_spot_prices.index for i in
        range(df_spot_prices.shape[1])}
    sub_model.spot = pyo.Param(sub_model.time, sub_model.scenario, initialize = spot_prices_dict)

    sub_model.probabilities = pyo.Param(sub_model.scenario,
                                    initialize = {1: 0.2, 2: 0.7, 3: 0.1})  # TODO: sett inn riktige sansynligheter

    df_demand = consumption_3_scenarios("rye_generation_and_load.csv")
    demand_dict = {(time + 1, i + 1): df_demand.iloc[time, i] for time in df_demand.index for i in
        range(df_demand.shape[1])}
    sub_model.demand = pyo.Param(sub_model.time, sub_model.scenario, initialize = demand_dict)

    sub_model.line_cap = pyo.Param(initialize = 5)  # MW

    # Variables
    sub_model.dispatch = pyo.Var(sub_model.time, sub_model.scenario, within = pyo.NonNegativeReals)
    sub_model.size = pyo.Var(within = pyo.NonNegativeReals)

    #Constraints
    def min_production(model, t, w):
        return (model.dispatch[t, w] >= 0)
    sub_model.min_prod = pyo.Constraint(sub_model.time, sub_model.scenario, rule = min_production)

    def max_production(model, t, w):
        return (model.dispatch[t, w] <= model.size)
    sub_model.max_prod = pyo.Constraint(sub_model.time, sub_model.scenario, rule = max_production)

    def power_balance1(model, t, w):
        return model.dispatch[t, w] - model.demand[t, w] >= -model.line_cap
    sub_model.pow_bal1 = pyo.Constraint(sub_model.time, sub_model.scenario, rule = power_balance1)
    def power_balance2(model, t, w):
        return model.dispatch[t, w] - model.demand[t, w] <= model.line_cap
    sub_model.pow_bal2 = pyo.Constraint(sub_model.time, sub_model.scenario, rule = power_balance2)
    def Size_rule(model):
        return model.size == model.size_hat
    sub_model.size_rule = pyo.Constraint(rule = Size_rule)

    # Objective function
    def Obj_sub(model):
        return sum(model.probabilities[w] * sum((model.spot[t, w] - model.var_cost) * model.dispatch[t, w]
                                                for t in model.time) for w in model.scenario)
    sub_model.obj = pyo.Objective(rule = Obj_sub, sense = pyo.maximize)

    return sub_model


''' ----------------------------- Add cuts ----------------------------- '''
def Cut_manage(Cuts, model):
    '''
    This function adds new cuts to the dictionary of cuts
    :param Cuts: Dictionary of cuts
    :param model: Pyomo model
    :return: The updated Cuts dictionary
    '''

    # Find cut iteration by checking number of existing cuts
    cut = len(Cuts["Set"])
    # Add new cut to list
    Cuts["Set"].append(cut)

    # Find sub problem results
    Cuts["Phi"][cut] = pyo.value(model.obj)
    Cuts["lambda"][cut] = model.dual[model.size_rule]
    Cuts["size_hat"][cut] = pyo.value(model.size_hat)

    return Cuts

''' --------------------------- Solve problem --------------------------- '''
def Solve(model):
    opt = SolverFactory("gurobi", solver_io="python")
    model.dual = pyo.Suffix(direction = pyo.Suffix.IMPORT)
    results = opt.solve(model, load_solutions = True)
    return results, model

def DisplayResults(model):
    return print(model.display(), model.dual.display())

''' --------------------------- Run problem --------------------------- '''

def Run_Problem_Solver():
    iterations = 10
    Cuts = {}
    Cuts["Set"] = []
    Cuts["Phi"] = {}
    Cuts["lambda"] = {}
    Cuts["size_hat"] = {}

    for i in range(iterations):

        # Set up and solve master problem
        master = ModelSetUp_Master(Cuts)
        Solve(master)
        size_hat = pyo.value(master.size)
        print(f"Iteration {i}, size = {size_hat}")
        print("Master objective function:", round(pyo.value(master.obj)/ 1000000, 2) , " MNOK")

        # Setup and solve sub problem
        sub = ModelSetUp_subproblem(size_hat)
        Solve(sub)
        Cuts = Cut_manage(Cuts, sub)
        print("Subproblem objective function:", round(pyo.value(sub.obj)/1000000, 2), " MNOK")
        print("Cut information acquired:")
        for component in Cuts:
            print(component, Cuts[component])

        # Convergence check
        print("UB:", pyo.value(master.alpha.value), "- LB:", pyo.value(sub.obj))



Run_Problem_Solver()