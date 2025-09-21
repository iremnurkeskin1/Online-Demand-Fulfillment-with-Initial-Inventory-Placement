#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This code is used in simulations for the paper:
'Online Demand Fulfillment Problem with Initial Inventory Placement: A Regret Analysis'
co-authored by Alessandro Arlotto, Irem Nur Keskin, and Yehua Wei.

Please read the README.md file before executing the code.

"""

import numpy as np
from gurobipy import *
import gurobipy as gp
gp.setParam('OutputFlag', 0)
import matplotlib.pyplot as plt
import random
import sys

class OfflineInventoryPlacement:
    """
    This class solves an optimization model to find the offline inventory placement using 
    sample average approximation.

    Attributes:
    - model (object): The optimization model instance.
    -       x (dict): Decision variables representing the fulfillment of demand from the 
                      warehouses.
    -       y (dict): Decision variables representing the lost sales for each demand region.
    -     res (dict): Decision variables representing the initial inventories placed into 
                      the warehouses.
    -     r_c (dict): Constraints ensuring the inventory placement does not exceed 
                      warehouse capacity.
    -     d_c (dict): Demand constraints ensuring that demands are met or the sales is lost.
    -     a (object): Constraint ensuring that the sum of the inventories in the warehouses 
                      does not exceed the total inventory capacity.
    
    Methods:
    -        __init__(): Initializes the model, sets the variables, and adds the constraints.
    -optimal_solution(): Solves the optimization model and returns the optimal solution.

    """
    def __init__(self, arrivals, total_inventory_cap, obj_coef_c, obj_coef_ls, T, num_samples):
        """
        Parameters:
        -       arrivals (array-like): 2D Array representing the sequence of demand arrivals for each sample.
        - total_inventory_cap (float): Total initial inventory to be distributed.
        -    obj_coef_c (array-like): fulfillment cost matrix: rows = warehouses, 
                                      columns = regions (obj_coef_c[i,j]).
        -   obj_coef_ls (array-like): Vector of lost sales costs for each demand region.
        -                    T (int): Length of the time horizon.
        -          num_samples (int): Number of samples.

        """
        self.model = Model('offline_opt')
        self.x = {}  # decision variables (fulfillment)
        self.res = {}  # decision variables  (initial inventory placement)     
        self.r_c = {}  # resource constraints
        self.d_c = {}  # demand constraints
        self.a = {} # inventory placement constraint
        self.y = {} # decision variables (lost sales)
        m, n = obj_coef_c.shape
          
        for i in range(m):
              for j in range(n):
                for w in range(num_samples):
                        self.x[i, j, w] = self.model.addVar(name="x{},{},{}".format(i, j, w))
                        self.x[i, j, w].setAttr(GRB.attr.Obj, obj_coef_c[i, j]/num_samples)   
                        self.res[i] = self.model.addVar(name="res{}".format(i))
                        self.y[j, w] = self.model.addVar(name="y{},{}".format(j,w))
                        self.y[j, w].setAttr(GRB.attr.Obj, obj_coef_ls[j]/num_samples)

        self.model.update()
        for j in range(n):
            for w in range(num_samples):
                 self.d_c[j, w] = self.model.addConstr(quicksum(self.x[i, j, w] 
                                  for i in range(m)) + self.y[j, w] 
                                  >= list(arrivals[:,w]).count(j))  
        for i in range(m):
             for w in range(num_samples):
                 self.r_c[i, w] = self.model.addConstr(quicksum(self.x[i, j, w] 
                                  for j in range(n)) <= self.res[i])            
        self.a = self.model.addConstr(quicksum(self.res[i] for i in range(m)) 
                 <= total_inventory_cap)  
        
        self.model.setAttr("ModelSense", GRB.MINIMIZE)
        self.model.setParam('OutputFlag', 0)

    def optimal_solution(self):
        """
        Returns:
        - array-like: Offline inventory placement vector
        """
        self.model.optimize()
        return np.array([self.res[i].X for i in range(len(self.res))]).astype(int)


class FluidPlacement:
    """
    This class solves an optimization model to find the fluid inventory placement
    based on the expected demand over the horizon.

    Attributes:
    -                T (int): Length of the time horizon.
    -          drate (array): Vector of arrival probabilities for each demand region.
    -            model (obj): The optimization model instance.
    -               x (dict): Decision variables representing expected fulfillment from
                              warehouses to regions in the fluid model (x[i, j]).
    -             res (dict): Decision variables representing the initial fluid inventories
                              placed into the warehouses (res[i]).
    -               y (dict): Decision variables representing expected lost sales for
                              each demand region in the fluid model (y[j]).
    -             d_c (dict): Demand constraints ensuring that (expected) demand is
                              either fulfilled or lost: sum_i x[i, j] + y[j] ≥ T * drate[j].
    -             r_c (dict): Resource constraints ensuring the demand assigned to each warehouse
                              does not exceed its initial inventory: sum_j x[i, j] ≤ res[i].
    -                a (obj): Aggregate inventory constraint ensuring the sum of
                              warehouse inventories does not exceed the total initial inventory:
                              sum_i res[i] ≤ total_inventory_cap.

    Methods:
    -              __init__(): Initializes the model, variables, and constraints.
    - _initialize_variables(): Defines decision variables and objective coefficients.
    -      _set_constraints(): Adds demand, resource, and total capacity constraints and
                              sets the model sense to minimization.
    -     optimal_solution(): Solves the model and returns the warehouse inventory vector.
    """

    def __init__(self, total_inventory_cap, obj_coef_c, obj_coef_ls, T, drate):
        """
        Parameters:
        -   total_inventory_cap (float): Total initial inventory to be distributed.
        -       obj_coef_c (array-like): Fulfillment cost matrix with shape 
                                         (warehouses × regions), where obj_coef_c[i, j]
                                         is the unit cost of fulfilling region j from
                                         warehouse i in the fluid model.
        -      obj_coef_ls (array-like): Vector of lost sales costs for each region.
        -                       T (int): Length of the time horizon.
        -                 drate (array): Vector of probabilities of arrival from each region.
        """
        self.T = T  
        self.drate = drate  

        self.model = Model('fluidplacement')
        self._initialize_variables(obj_coef_c, obj_coef_ls)
        self._set_constraints(total_inventory_cap)
        self.model.setParam('OutputFlag', 0)

    def _initialize_variables(self, obj_coef_c, obj_coef_ls):
        """
        Initializes decision variables and attaches per-unit costs to the objective.

        """
        m, n = obj_coef_c.shape
        self.x = {(i, j): self.model.addVar(vtype='INTEGER', 
                                   obj=obj_coef_c[i, j], 
                                   name=f"x{i},{j}")  for i in range(m) for j in range(n)}

        self.res = {i: self.model.addVar(vtype='INTEGER',
                             name=f"res{i}")  for i in range(m)}

        self.y = {(j): self.model.addVar(vtype='INTEGER',
                                obj=obj_coef_ls[j], 
                                name=f"y{j}") for j in range(n)}
        self.model.update()

    def _set_constraints(self, total_inventory_cap):
        """
        Adds the fluid demand, warehouse capacity, and total inventory constraints, and
        sets the model to minimization.

        """
        m, n = obj_coef_c.shape
        self.d_c = {j: self.model.addConstr(quicksum(self.x[i, j] for i in range(m)) 
                    + self.y[j] >= (self.T) * self.drate[j]) for j in range(n)}
        self.r_c = {i: self.model.addConstr(quicksum(self.x[i, j] for j in range(n)) 
                    <= self.res[i]) for i in range(m)}
        self.a = self.model.addConstr(quicksum(self.res[i] for i in range(m)) 
                    <= total_inventory_cap)
        self.model.setAttr("ModelSense", GRB.MINIMIZE)

    def optimal_solution(self):
        """
        Solves the fluid placement model and returns:
        - array-like: The fluid inventory vector.
        """
        self.model.optimize()
        return np.array([self.res[i].X for i in range(len(self.res))])

    

class OfflineProblem:
    
    """
    This class solves an optimization model to find total cost of the offline problem
    given a sample and specific initial inventory placement approach. 

    
    Attributes:
    - model (object): The optimization model instance.
    -       x (dict): Decision variables representing the the amount fulfilled from demand 
                      regions by the warehouses.
    -       y (dict): Decision variables representing the total lost sales for each demand 
                      region.
    -     r_c (dict): Resource constraints ensuring the inventory placement does not exceed 
                      warehouse capacity.
    -     d_c (dict): Demand constraints ensuring that demands are met or the sales is lost.
    """

    def __init__(self, arrivals, inventory_cap, obj_coef_c, obj_coef_ls):
        """
        Initialize the optimization model, set variables and constraints.

        Parameters:
        -      arrivals (array-like): Array representing demand arrivals
        - inventory_cap (array-like): Capacity of each resource (inventory in the warehouses).
        -    obj_coef_c (array-like): Cost coefficients related to fulfilling demand from the warehouses.
        -   obj_coef_ls (array-like): Lost sales coefficients for each demand region.

        """
        self.model = Model('offline_opt')
        self._initialize_variables(obj_coef_c, obj_coef_ls)
        self._set_constraints(arrivals, inventory_cap)
        self.model.setParam('OutputFlag', 0)

    def _initialize_variables(self, obj_coef_c, obj_coef_ls):
        """
        Initialize the decision variables for fulfillment and lost sales.

        Parameters:
        -  obj_coef_c (array-like): Cost coefficients related to fulfilling the demand from the warehouses.
        - obj_coef_ls (array-like): Lost sales coefficients for each demand region.

        """
        m, n = obj_coef_c.shape
        self.x = {(i, j): self.model.addVar(obj=obj_coef_c[i, j], name=f"x{i},{j}") for i in range(m) for j in range(n)}
        self.y = {j: self.model.addVar(obj=obj_coef_ls[j], name=f"y{j}") for j in range(n)}
        self.model.update()

    def _set_constraints(self, arrivals, inventory_cap):
        """
        Define the demand and inventory constraints for the optimization problem.

        Parameters:
        -     arrivals (array-like): Represents the sequence of arrivals coming from 
                                     the demand regions.
        - inventory_cap (array-like): Initial inventory at each warehouse.

        """
        m, n = obj_coef_c.shape
        
        # Demand constraints
        demand = [list(arrivals).count(j) for j in range(n)]
        self.d_c = {j: self.model.addConstr(quicksum(self.x[i, j] for i in range(m)) + self.y[j] >= demand[j]) for j in range(n)}
        
        # Inventory constraints
        self.r_c = {i: self.model.addConstr(quicksum(self.x[i, j] for j in range(n)) <= inventory_cap[i]) for i in range(m)}
        
        # Set model to minimize the cost
        self.model.setAttr("ModelSense", GRB.MINIMIZE)

    def optimal_val(self):
        """
        Solve the optimization problem and return the objective value.

        Returns:
        - float: The optimal objective value of the solved model.
        
        """
        self.model.optimize()
        return self.model.objVal

    
class FluidProblem:
    """
    Models a fluid approximation of the fulfillment problem with dynamic updates for each period.
    """

    def __init__(self, arrivals, inventory_cap, obj_coef_c, obj_coef_ls, drate, T):
        """
        Initialize the optimization model with general problem structure.
        """
        self.model = Model('fluid_problem')
        self.T = T
        self.x = {}
        self.y = {}
        self.inventory_cap = inventory_cap
        self.drate = drate
        self.I, self.J = obj_coef_c.shape

        # Add decision variables
        for i in range(self.I):
            for j in range(self.J):
                self.x[i, j] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"x_{i},{j}")
        for j in range(self.J):
            self.y[j] = self.model.addVar(vtype=GRB.CONTINUOUS,lb=0, name=f"y_{j}")
        
        # Set objective
        self.obj_coef_c = obj_coef_c
        self.obj_coef_ls = obj_coef_ls
        self.model.setObjective(
            quicksum(obj_coef_c[i, j] * self.x[i, j] for i in range(self.I) for j in range(self.J)) +
            quicksum(obj_coef_ls[j] * self.y[j] for j in range(self.J)),
            GRB.MINIMIZE
        )

        # Add constraints
        self.resource_constraints = {}
        self.demand_constraints = {}

        for i in range(self.I):
            self.resource_constraints[i] = self.model.addConstr(
                quicksum(self.x[i, j] for j in range(self.J)) <= inventory_cap[i],
                name=f"cap_{i}"
            )
        
        for j in range(self.J):
            self.demand_constraints[j] = self.model.addConstr(
                quicksum(self.x[i, j] for i in range(self.I)) + self.y[j] >= 0,  # Placeholder, to be updated
                name=f"demand_{j}"
            )
        
        self.model.update()

    def update_and_optimize(self, t, new_inventory_cap):
        """
        Update the model at each time t with current inventories and future expected 
        demand and solve the optimization problem.

        Parameters:
        -  t (int): Current time period.
        - new_inventory_cap (array-like): Remaining inventories in the warehouses for this period.
        """
        # Update resource constraints
        for i in range(self.I):
            self.resource_constraints[i].setAttr(GRB.Attr.RHS, new_inventory_cap[i])

        # Update demand constraints
        for j in range(self.J):
            self.demand_constraints[j].setAttr(GRB.Attr.RHS, (self.T - t) * self.drate[j])

        # Re-optimize the model
        self.model.optimize()

        if self.model.status != GRB.OPTIMAL:
            raise RuntimeError("Optimization did not converge to an optimal solution.")
            
        
        
        # Extract x and y values
        x_values = np.array([[self.x[i, j].X for j in range(self.J)] for i in range(self.I)])
        y_values = np.array([self.y[j].X for j in range(self.J)])
        
        return np.vstack((x_values, y_values))



def policy_pi(arrivals, demand_rate, warehouse_capacity, obj_coef_c, obj_coef_ls, T, I, J, policy):
    """
 Defines the policies given an initial inventory placement 

 Parameters:
 -           arrivals (array_like): Array representing an arrival sequence over time.
 -        demand_rate (array_like): Array representing probability of arrival from each demand region.
 - warehouse_capacity (array_like): A list or array representing the initial inventories of each warehouse.
 -         obj_coef_c (array_like): 2D array representing objective coefficients for fulfillment costs.
 -        obj_coef_ls (array_like): 2D array representing objective coefficients for lost sales.
 -                         T (int): The length of the horizon.
 -                         I (int): Total number of warehouses.
 -                         J (int): Total number of demand regions.
 -                    policy (str): 'SF' (for score-based fulfillment), 'PF' (for probabilistic-fulfillment), or 'Myopic'.

 Returns:
 - float: Total cost for the given policy over the period T.
 """
    all_coefficients = np.vstack([obj_coef_c, obj_coef_ls])
    remaining_capacity = warehouse_capacity.copy()
    total_cost = 0

    fluid_sol = FluidProblem(arrivals, remaining_capacity, obj_coef_c, obj_coef_ls, demand_rate, T)


    if policy == 'SF':
        for t in range(T):
            # Solve the fluid optimization problem for the current time step
            allocation_matrix = fluid_sol.update_and_optimize(t, remaining_capacity)

            # Find the warehouse or lost sales with the highest score for the current demand
            max_index = np.argmax(allocation_matrix[:, arrivals[t]])

            # If demand is fulfilled by a warehouse, update the inventory
            if max_index != I:  # Assuming I represents lost sales
                remaining_capacity[max_index] -= 1

            # Update the total cost
            total_cost += all_coefficients[max_index, arrivals[t]]

        
    
    elif policy=='PF':      
              for t in range(T): 
                  
                    # Solve the fluid optimization problem for the current time step
                    opt_sol_fluid = fluid_sol.update_and_optimize(t, remaining_capacity) 
                    
                    # Assign probabilities to each decision using the fluid solution
                    probabilities=np.zeros([I+1,J])
                    for i in range(I+1):
                        for j in range(J):
                            probabilities[i,j]=(opt_sol_fluid[i,j])/((T-t)*demand_rate[j])
                    
                    # Round to four decimal place
                    rounded_probs = np.round(probabilities[:,arrivals[t]], 4)
                    
                    # Normalize to sum to 1
                    normalized_probs = rounded_probs / rounded_probs.sum()
                    select_warehouse = np.random.choice(list(range(0,I+1)), p=normalized_probs) 
                    
                    # If the demand is not lost, update the inventory position at the selected warehouse 
                    if select_warehouse != I: 
                        remaining_capacity[select_warehouse] -= 1
                        
                    # Update the total cost
                    total_cost += all_coefficients[select_warehouse, arrivals[t]]
                    
    elif policy == 'myopic':
        for t in range(T):
            current_arrival = arrivals[t]
            #Find all the warehouses that has an inventory and store the cost
            costs = [(all_coefficients[i, current_arrival], i) for i in range(I) if remaining_capacity[i] > 0]
            #include the lost sales cost as an option
            costs.append((obj_coef_ls[current_arrival], I))  
            #Find the decision (warehouse or lost sales) with the smallest cost 
            best_cost, best_choice = min(costs, key=lambda x: x[0])
            # If demand is fulfilled by a warehouse, update the inventory
         
            if best_choice != I:
                remaining_capacity[best_choice] -= 1
            # Update the total cost
            total_cost += best_cost
            
    
    return total_cost
          
 

def expected_offline_cost_at_k_off(arrivals, offline_inventory, obj_coef_c, obj_coef_ls, T, num_samples): 
    """
    Function to find the expected offline cost with offline inventory placement
    """
    offline_costs=np.zeros(num_samples)
    for k in range(num_samples):
        arrivals_k= arrivals[:,k]
        cost_offline= OfflineProblem(arrivals_k, offline_inventory, obj_coef_c, obj_coef_ls)
        offline_costs[k]=cost_offline.optimal_val()
    average_cost_offline=np.mean(offline_costs) 
    return average_cost_offline, offline_costs


def expected_policy_cost_at_k(arrivals, demand_rate, inventory_capacity, obj_coef_c, obj_coef_ls, T, I, J, num_samples):
    """
    For each demand sample, the function evaluates the cost of both the SF, PF, and Myopic policies 
    using the provided inventory placement vector. It then calculates the average expected cost for each 
    policy over all demand samples.
    
    
    Parameters:
    -           arrivals (array-like): Matrix of demand arrivals with dimensions [T x num_samples].
    -        demand_rate (array-like): Probability distribution of demand from regions
    - warehouse_capacity (array-like): Inventory levels for each warehouse.
    -         obj_coef_c (array_like): 2D array representing objective coefficients 
                                       for fulfillment costs.
    -        obj_coef_ls (array_like): 2D array representing objective coefficients 
                                       for lost sales.
    -                         T (int): Time horizon length.
    -                         I (int): Number of warehouses.
    -                         J (int): Number of demand regions.
    -               num_samples (int): Number of samples to evaluate.


    """

    results = []
    for k in range(arrivals.shape[1]):  # Assuming arrivals_all is [T x num_samples]
        arrival = arrivals[:, k]
        result_sf = policy_pi(arrival, demand_rate, inventory_capacity, obj_coef_c, obj_coef_ls, T, I, J, 'SF')
        result_pf = policy_pi(arrival, demand_rate, inventory_capacity, obj_coef_c, obj_coef_ls, T, I, J, 'PF')
        result_myopic = policy_pi(arrival, demand_rate, inventory_capacity, obj_coef_c, obj_coef_ls, T, I, J, 'myopic')
        results.append((result_sf, result_pf, result_myopic))

    # Extract results for each policy
    results_sf, results_pf, results_myopic = zip(*results)

    # Compute the average result for each policy
    average_result_sf = np.mean(results_sf)
    average_result_pf = np.mean(results_pf)
    average_result_myopic = np.mean(results_myopic)
    
    return average_result_sf, average_result_pf, average_result_myopic, results_myopic, results_sf, results_pf

def expected_costs(I, J, obj_coef_c, obj_coef_ls, demand_rate, theta, num_samples, horizon, example_type):
    """
    Computes expected costs for fulfillment policies over different time horizons.

    Parameters:
    -                 I (int): Number of warehouses.
    -                 J (int): Number of demand regions.
    - obj_coef_c (array-like): Coefficients related to fulfillment cost.
    -obj_coef_ls (array-like): Coefficients related to lost sales.
    -demand_rate (array-like): Probability distribution of demand.
    -           theta (float): Multiplier for calculating warehouse capacity.
    -       num_samples (int): Number of demand samples to consider.
    -          horizon (list): List of different time horizons to be evaluated.
    -      example_type (str): Example 1, Example 2, or Amazon_China example. 

    Returns:
    - avg_results (array-like): Results array containing the expected costs for different fulfillment policy 
                                and inventory placement method pairs.
     """

    # Initialize array to store average results for each horizon and strategy
    avg_results=np.zeros([len(horizon),11])
    avg_results[:, 0] = np.array(horizon)

    # Lists to store inventory values for different policies
     
    PF_all_off_results= np.zeros([len(horizon),num_samples])
    SF_all_off_results=np.zeros([len(horizon),num_samples])
    myopic_all_off_results=np.zeros([len(horizon),num_samples])
    
    PF_all_fluid_results= np.zeros([len(horizon),num_samples])
    SF_all_fluid_results=np.zeros([len(horizon),num_samples])
    myopic_all_fluid_results=np.zeros([len(horizon),num_samples])
    
    
    offline_all_results= np.zeros([len(horizon),num_samples])
    
    
    for i in range(len(horizon)):
        # Calculate the total warehouse capacity based on the current horizon and theta
        T = horizon[i]
        total_warehouse_cap = theta * T
        fluid_allocation= FluidPlacement(total_warehouse_cap, obj_coef_c, obj_coef_ls, T, demand_rate)
        fluid_inventory= fluid_allocation.optimal_solution()

        # Generate demand samples based on a fixed seed (for reproducibility)
        np.random.seed(1)
        arrivals = np.random.choice(len(demand_rate), [T, num_samples], p=demand_rate)
        
        # Determine the offline inventory placement using the OfflineInventoryPlacement method
        offline_allocation = OfflineInventoryPlacement(arrivals, total_warehouse_cap, obj_coef_c, obj_coef_ls, T, num_samples)
        offline_inventory = offline_allocation.optimal_solution()
        
        scaled_inv=np.zeros(I)
        for j in range(J):
            index = np.argmin(obj_coef_c[:,j])
            scaled_inv[index]+=demand_rate[j]*theta * T
        scaled_inv=np.floor(scaled_inv)
        
        
        # Generate a new set of demand samples
        np.random.seed(10)
        arrivals = np.random.choice(len(demand_rate), [T, num_samples], p=demand_rate)

        # Calculate the offline cost for the obtained offline inventory
        [offline_cost,offline_cost_all] = expected_offline_cost_at_k_off(arrivals, offline_inventory, obj_coef_c, obj_coef_ls, T, num_samples)

        # Calculate costs for fulfillment policies when using offline inventory values
        [SF_cost_at_k_off, PF_cost_at_k_off, myopic_cost_at_k_off , SF_cost_at_k_off_all, PF_cost_at_k_off_all, myopic_cost_at_k_off_all ] = expected_policy_cost_at_k(arrivals, demand_rate, offline_inventory, obj_coef_c, obj_coef_ls, T, I, J, num_samples)
       
        # Calculate costs for fulfillment policies when using fluid inventory values
        [SF_cost_at_k_fluid, PF_cost_at_k_fluid, myopic_cost_at_k_fluid , SF_cost_at_k_fluid_all, PF_cost_at_k_fluid_all, myopic_cost_at_k_fluid_all ] = expected_policy_cost_at_k(arrivals, demand_rate, fluid_inventory, obj_coef_c, obj_coef_ls, T, I, J, num_samples)
            
        # Calculate costs for fulfillment policies when using scaled inventory values
        [SF_cost_at_k_scaled, PF_cost_at_k_scaled, myopic_cost_at_k_scaled, SF_cost_at_k_scaled_all, PF_cost_at_k_scaled_all, myopic_cost_at_k_scaled_all ] = expected_policy_cost_at_k(arrivals, demand_rate, scaled_inv, obj_coef_c, obj_coef_ls, T, I, J, num_samples)
          
        # Update the avg_results array with calculated costs
        avg_results[i, 1] = offline_cost
        avg_results[i, 2] = SF_cost_at_k_off
        avg_results[i, 3] = PF_cost_at_k_off
        avg_results[i, 4] = myopic_cost_at_k_off
        avg_results[i, 5] = SF_cost_at_k_fluid
        avg_results[i, 6] = PF_cost_at_k_fluid
        avg_results[i, 7] = myopic_cost_at_k_fluid
        avg_results[i, 8] = SF_cost_at_k_scaled
        avg_results[i, 9] = PF_cost_at_k_scaled
        avg_results[i, 10] = myopic_cost_at_k_scaled
        
        print()
        print('Regret at T:', T)
        # compute regrets
        rows = [("SF", "off",    SF_cost_at_k_off - offline_cost),
            ("PF", "off",    PF_cost_at_k_off - offline_cost),
            ("Myopic", "off",    myopic_cost_at_k_off - offline_cost),
            ("SF", "fluid",  SF_cost_at_k_fluid - offline_cost),
            ("PF", "fluid",  PF_cost_at_k_fluid - offline_cost),
            ("Myopic", "fluid",  myopic_cost_at_k_fluid - offline_cost),
            ("SF", "scaled", SF_cost_at_k_scaled - offline_cost),
            ("PF", "scaled", PF_cost_at_k_scaled - offline_cost),
            ("Myopic", "scaled", myopic_cost_at_k_scaled - offline_cost)]
        
        print(f"{'Policy':<10} {'Placement':<10} {'Regret':>10}")
        print("-" * 32)
        for policy, method, regret in rows:
            print(f"{policy:<10} {method:<10} {regret:>10.2f}")
                

    return avg_results

#Plot the regret of score-based and probabilistic fulfillment policies

def plot_regret(avg_results):
    """
    This function visualizes the joint regret over time for different policies with
    their best inventory placement out of (off, fluid, scaled).
    
    Parameters:
    - avg_results (array-like): A 2D array containing average results for policy placement 
                                pairs at different horizon lengths T.
       
    The function produces two plots:
    1. Joint regret of SF and PF with their best inventory placement out of (off, fluid, scaled).
    2. Joint regret of SF, PF, and Myopic with their best inventory placement out of (off, fluid, scaled).
    """
    
    
    x = avg_results[:, 0]
    #Stacking the regrets of SF with offline, fluid and scaled placement, respectively.
    stacked_1 = np.stack([avg_results[:, 2], avg_results[:, 5], avg_results[:, 8]], axis=1)
    #Stacking the regrets of PF with offline, fluid and scaled placement, respectively.
    stacked_2 = np.stack([avg_results[:, 3], avg_results[:, 6], avg_results[:, 9]], axis=1)
    #Stacking the regrets of Myopic with offline, fluid and scaled placement, respectively.
    stacked_3 = np.stack([avg_results[:, 4], avg_results[:, 7], avg_results[:, 10]], axis=1)
    
    # Step 2: Calculate the regrets of polcy placement pairs by subtracting avg_results[:, 1]
    regret_1 = stacked_1 - avg_results[:, 1].reshape(-1, 1)
    regret_2 = stacked_2 - avg_results[:, 1].reshape(-1, 1)
    regret_3 = stacked_3 - avg_results[:, 1].reshape(-1, 1)
    
    # Step 3: To prevent small rounding errors (i.e. when the regret is -0.0000001)
    regret_1_non_negative = np.maximum(regret_1, 0)
    regret_2_non_negative = np.maximum(regret_2, 0)
    regret_3_non_negative = np.maximum(regret_3, 0)
    
    # Step 4: Calculate the minimum regret for each policy from the three placement methods
    y1 = np.min(regret_1_non_negative, axis=1)
    y2 = np.min(regret_2_non_negative, axis=1)
    y3 = np.min(regret_3_non_negative, axis=1)
    y4 = avg_results[:, 5] - avg_results[:, 1]  
    y5 = avg_results[:, 6] - avg_results[:, 1]  
    y6 = avg_results[:, 7] - avg_results[:, 1]  
    
    # Plot the regret of SF and PF at their best placement out of (off, fluid, scaled) as T increases. 
    plt.plot(x, y1, linestyle='-.', marker='s', markersize=10, color='r', label='SF')
    plt.plot(x, y2, linestyle='dotted', marker='o', markersize=10, color='c', label='PF')
    
    # Add axis labels, legend, and formatting
    plt.xlabel('Length of the time horizon (T)', fontsize=16)
    plt.ylabel('Regret', fontsize=15)
    plt.xticks(x, fontsize=13)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.show()
    
    # Plot the regret of SF, PF and Myopic at their best placement out of (off, fluid, scaled) as T increases. 
    plt.plot(x, y1, linestyle='-.', marker='s', markersize=10, color='r', label='SF')
    plt.plot(x, y2, linestyle='dotted', marker='o', markersize=10, color='c', label='PF')
    plt.plot(x, y3, linestyle='--', marker='^', markersize=10, color='y', label='myopic')
    
    # Add axis labels, legend, and formatting
    plt.xlabel('Length of the time horizon (T)', fontsize=16)
    plt.ylabel('Regret', fontsize=15)
    plt.xticks(x, fontsize=13)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.show()
    

#Initialize the parameters

def parameters(example_type,flexibility_structure):
    """
    Returns the parameters for the specified example type.
   
    Parameters:
    - example_type (str): The type of example. Acceptable values are 'Example1',
                          'Example2', or 'Amazon_China'.
   
    Returns:
    -                  I (int): Number of warehouses.
    -                  J (int): Number of demand regions.
    - obj_coef_c (numpy array): Coefficients matrix related to fulfillment cost, with shape [I x J].
    -obj_coef_ls (numpy array): Coefficients array related to lost sales cost, with shape [J].
    -demand_rate (numpy array): Probability of demand arrivals across demand regions.
           - num_samples (int): Number of samples 
    -    horizon (list of int): List of time horizon values to be evalutated.
   

        """
    
    horizon = [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]
    num_samples = 1000
    
    # Initial parameters for 'Example1'
    if example_type == 'Example1':
        J = 2  # Number of demand regions
        I = 2  # Number of warehouses
        obj_coef_c = np.array([[2, 5], [3, 1]])  # Fulfillment cost coefficients matrix
        
        obj_coef_ls = np.array([4,4])  # Lost sales coefficients array
        demand_rate = np.ones(J) * 1/J  # Probability distribution of demand


    # Parameters and plotting for 'Example2'
    elif example_type == 'Example2':
        I_all = 100  # Total possible warehouses (will select from this sample)
        J_all = 100  # Total possible demand regions (will select from this sample)
        
        # Seed for reproducibility
        np.random.seed(92)

        # Randomly generate coordinates for warehouses and demand regions
        coordinates_I_x = np.random.rand(I_all)
        coordinates_I_y = np.random.rand(I_all)
        coordinates_J_x = np.random.rand(J_all)
        coordinates_J_y = np.random.rand(J_all)

        # Initialize a zeros matrix for all potential cost coefficients
        obj_coef_all = np.zeros([I_all, J_all])

        # Compute distance-based coefficients for each warehouse-demand region pair
        for i in range(I_all):
            for j in range(J_all):
                obj_coef_all[i, j] = np.linalg.norm(
                    np.array([coordinates_I_x[i], coordinates_I_y[i]]) - 
                    np.array([coordinates_J_x[j], coordinates_J_y[j]])
                )

        # Number of warehouses and demand regions to consider in this example
        I = 3
        J = 5

        # Extract the relevant coefficients based on I and J
        obj_coef_c = obj_coef_all[0:I, 0:J]
        obj_coef_ls = np.array(np.ones(J) * 2)  # Given lost sales coefficients

        # Plotting warehouse and demand region locations
        plt.scatter(coordinates_I_x[0:I], coordinates_I_y[0:I], marker="s", label="Warehouses", color="c", s=150)
        plt.scatter(coordinates_J_x[0:J], coordinates_J_y[0:J], marker="o", label="Demand Regions", color="r", s=150)
        plt.legend(bbox_to_anchor=(0.7, -0.1))
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])
        plt.savefig('locations.png', format='png', dpi=1200)
        plt.show()
        
        # Parameters for the demand distribution, given constant, number of samples, and time horizons
        demand_rate = np.ones(J) * 1/J  
        
        
    elif example_type=='Amazon_China':
            from geopy.geocoders import Nominatim
            from geopy.distance import geodesic

            # Wrehouses/ Fullfillment centers (I) and demand regions (J)
            resources = ["Harbin", "Shenyang", "Beijing", "Wuhan", "Shanghai", 
                         "Xiamen", "Guangzhou", "Nanning", "Chengdu", "Xian"]

            # regions (Flattened unique regions)
            regions = ["Harbin", "Daqing", "Changchun", "Yanbian", "Shenyang", "Jinzhou", 
                        "Dalian", "Beijing", "Tianjin", "Tangshan", "Shijiazhuang", 
                        "Jinan", "Qingdao", "Wuhan", "Nanchang", "Changsha", "Hefei", 
                        "Xiangyang", "Shanghai", "Suzhou", "Hangzhou", "Ningbo", "Wenzhou", 
                        "Changzhou", "Wuxi", "Nanjing", "Xuzhou", "Xiamen", "Fuzhou", 
                        "Ganzhou", "Guangzhou", "Foshan", "Shenzhen", "Dongguan", 
                        "Nanning", "Kunming", "Guilin", "Guiyang", "Chengdu", "Chongqing", 
                        "Xian", "Hanzhong", "Zhengzhou", "Taiyuan"]

            # Demand rates (p_j) from the table (flattened)
            demand_rate = np.array([
                0.019, 0.019, 0.013, 0.013, 0.017, 0.022, 0.022, 0.057, 0.015, 0.015, 
                0.029, 0.020, 0.019, 0.029, 0.029, 0.021, 0.009, 0.012, 0.026, 0.026, 
                0.026, 0.026, 0.026, 0.019, 0.026, 0.026, 0.057, 0.009, 0.007, 0.019, 
                0.045, 0.045, 0.045, 0.028, 0.006, 0.010, 0.010, 0.011, 0.028, 0.036, 
                0.009, 0.018, 0.019, 0.019
            ])
            demand_rate = demand_rate / np.sum(demand_rate)

            # Initialize geolocator for coordinates
            geolocator = Nominatim(user_agent="geo_distance_estimator")

            # Fetch coordinates for all resources and regions
            coordinates = {}
            for location in resources + regions:
                loc = geolocator.geocode(location)
                if loc:
                    coordinates[location] = (loc.latitude, loc.longitude)
                else:
                    print(f"Coordinates not found for {location}")

            # Build cost matrix (obj_coef_c): distances between warehouse and demand regions 
            I, J = len(resources), len(regions)
            obj_coef_c = np.full((I, J), 1e9)  # Start with infinities
            obj_coef_c_0 = np.full((I, J), 1e9)
            # Populate distances for valid warehouse-demand region pairs
            resource_request_map = {
                "Harbin": ["Harbin", "Daqing", "Changchun", "Yanbian"],
                "Shenyang": ["Shenyang", "Jinzhou", "Dalian"],
                "Beijing": ["Beijing", "Tianjin", "Tangshan", "Shijiazhuang", "Jinan", "Qingdao"],
                "Wuhan": ["Wuhan", "Nanchang", "Changsha", "Hefei", "Xiangyang"],
                "Shanghai": ["Shanghai", "Suzhou", "Hangzhou", "Ningbo", "Wenzhou", 
                             "Changzhou", "Wuxi", "Nanjing", "Xuzhou"],
                "Xiamen": ["Xiamen", "Fuzhou", "Ganzhou"],
                "Guangzhou": ["Guangzhou", "Foshan", "Shenzhen", "Dongguan"],
                "Nanning": ["Nanning", "Kunming", "Guilin", "Guiyang"],
                "Chengdu": ["Chengdu", "Chongqing"],
                "Xian": ["Xian", "Hanzhong", "Zhengzhou", "Taiyuan"]
            }
            
            for i, resource in enumerate(resources):
                for request in resource_request_map[resource]:
                    if resource in coordinates and request in coordinates:
                        j = regions.index(request)
                        obj_coef_c_0[i, j] =  0.423 + 0.000541*geodesic(coordinates[resource], coordinates[request]).miles
    
            
            # Calculate the maximum cost in the no flexibility structure and use it to calculate the lost sales cost
            lost_sales = 2*np.nanmax(np.where(obj_coef_c_0!= 1e9, obj_coef_c_0, np.nan))
            obj_coef_ls = np.full(len(regions), lost_sales)
         
            
            if flexibility_structure=="Full_Flexibility":
                for i, resource in enumerate(resources):
                    for j, request in enumerate(regions):
                        if resource in coordinates and request in coordinates:
                            obj_coef_c[i, j] = 0.423 + 0.000541*geodesic(coordinates[resource], coordinates[request]).miles
                            
            elif flexibility_structure=="No_Flexibility":
                    obj_coef_c=obj_coef_c_0
                    
            else: 
                for i, resource in enumerate(resources):
                    for request in resource_request_map[resource]:
                        if resource in coordinates and request in coordinates:
                            j = regions.index(request)
                            obj_coef_c[i, j] =  0.423 + 0.000541*geodesic(coordinates[resource], coordinates[request]).miles
                
                # Limited flexibility
                additional_arcs = [("Shenyang", "Changchun"), ("Beijing", "Jinzhou"), ("Chengdu", "Hanzhong"), ("Nanning", "Chengdu")
                       , ("Guangzhou", "Guilin"), ("Xiamen", "Shenzhen"), ("Shanghai", "Xiamen"), ("Shanghai", "Hefei") ,("Beijing", "Xuzhou"), ("Xian", "Jinan") ]
                
                def add_arcs_to_cost_matrix(obj_coef_c, arcs, resources, regions, coordinates):
                          for resource, request in arcs:
                              if resource in coordinates and request in coordinates:
                                  i = resources.index(resource)
                                  j = regions.index(request)
                                  obj_coef_c[i, j] = 0.423 + 0.000541*geodesic(coordinates[resource], coordinates[request]).miles
                          return obj_coef_c
                      
                obj_coef_c = add_arcs_to_cost_matrix(obj_coef_c, additional_arcs, resources, regions, coordinates)
            

    return I, J, obj_coef_c, obj_coef_ls, demand_rate, num_samples, horizon

    
if __name__ == "__main__":
    
    # Define valid example options
    examples = ["Example1","Example2","Amazon_China"]
    
    #Define flexibility levels (for Amazon China Example)
    flexibility=["Full_Flexibility", "Limited_Flexibility", "No_Flexibility"]
    
    # User inputs example
    example_type = input("Please enter an example type (Example1, Example2 or Amazon_China): ")
    
    # Check input
    if example_type in examples:
        flexibility_structure=[]
        if example_type=='Amazon_China':
            flexibility_structure = input("Please enter a flexibility structure for Amazon China Example (Full_Flexibility, Limited_Flexibility, No_Flexibility):")
            if flexibility_structure not in flexibility:
                print("Invalid flexibility structure. Program terminated.")
                sys.exit()
    else:
        print("Invalid example option. Program terminated.")
        sys.exit()
       
    
    print("\033[91m" + example_type + "\033[0m")

    #Initialize the parameters given example
    
    theta=0.9 #Scaling parameter to identify total initial inventory across warehouses, with theta*T. Change this to try different staring inventory
    
    print('Theta (Initial inventory scaling parameter):', theta)
    [I,J,obj_coef_c, obj_coef_ls,demand_rate, num_samples, horizon]= parameters(example_type,flexibility_structure)

    #Find the expected cost of the policies 
    
    avg_results=expected_costs(I,J,obj_coef_c, obj_coef_ls,demand_rate,theta, num_samples, horizon, example_type)
  
    #Plot the regret vs length of the time horizon for different policies
   
    plot_regret(avg_results)









