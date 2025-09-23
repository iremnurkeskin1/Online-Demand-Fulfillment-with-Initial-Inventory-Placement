# <span style="color:red"> Online Demand Fulfillment with Initial Inventory Placement: Simulation Code</span>

This repository contains the code for the simulation experiments accompanying the paper:  
**"Online Demand Fulfillment with Initial Inventory Placement: A Regret Analysis"**  
by *Alessandro Arlotto, Irem Nur Keskin, and Yehua Wei*.  

The main script, **`Online_Fulfillment+Placement.py`**, implements and compares the joint regret of different **fulfillment policies** and **inventory placement methods** across three examples.  
Supporting files (e.g., `amazon_china_data.py`) provide input data and coefficients for reproducibility.



## <span style="color:blue">  Requirements </span>

The experiments in this repository were run with the following dependencies:

- **Python**: 3.11.13  
- **NumPy**: 2.3.2  
- **Matplotlib**: 3.10.5  
- **Gurobi**: 12.0.3 (with valid license)  
- **geopy**: only needed for the `Amazon_China` example


To install the required dependencies, run the following command:
```bash
pip install -r requirements.txt
```

Alternatively, you can install each dependency separately. For example, to install numpy, you may type
```bash
pip install numpy==2.3.2

Note on Gurobi: the code uses Gurobi as the optimization solver for the experiments. To run those parts you must install the Gurobi Python package and have a valid Gurobi license configured on your machine. See Gurobi's academic license page and installation instructions:

- Gurobi academic licenses: https://www.gurobi.com/academia/academic-program-and-licenses/
- Gurobi installation & Python interface: https://www.gurobi.com/documentation/
```



## <span style="color:blue"> How to Run </span>

Run the script and follow the prompts:
```bash
python Online_Fulfillment+Placement.py
```

You will be asked to choose an example:
- `Example1` (Synthetic Example with two warehouses and two demand regions)
- `Example2` (Synthetic Example with three warehouses and five demand regions)
- `Amazon_China` (Case study based on the demand regions and fulfillment centers described in Xu, Z., Zhang, H., Zhang, J., & Zhang, R. Q. (2020). *Online Demand Fulfillment Under Limited Flexibility*. *Management Science*, 66(10), 4667–4685.)

If you choose `Amazon_China`, you will also be asked to choose a **flexibility structure**:
- `Full_Flexibility`
- `Limited_Flexibility`
- `No_Flexibility`

The script then prints regret tables for each pair of fulfillment policy and inventory placement method at different horizon lengths and produces **two plots** to visualize the change in the regret of fulfillment policies vs horizon length.


## <span style="color:blue"> Parameters (Inside the Script)

A few key parameters are set in the `__main__` block (search for them in `Online_Fulfillment+Placement.py`):

- `theta = 0.9` — scaling for total initial inventory across warehouses (`theta * T` where T is the length of the horizon). In the paper, we also report the results for  theta = 0.3, theta = 0.9, theta = 1, theta=1.1 for Example 1 and Example 2.  
- `horizon = [1000, 2000, ..., 10000]` — defined in `parameters()`; edit there to change.  
- `num_samples = 1000` — number samples created to approximate the expected regrets  (also in `parameters()`).  
- Random seeds used in experiments:
  - `np.random.seed(1)` and `np.random.seed(10)` before sampling arrivals.

You can modify these directly in the script to reproduce or stress test different regimes.


## <span style="color:blue"> Notes for `Amazon_China` Example </span>

- Uses **Nominatim** via `geopy` for geocoding. This requires **internet access** and is **rate-limited**.
- We use geocoded distances to approximate **fulfillment costs** between demand regions and fulfillment centers and to set **lost-sales costs**.
- **Runtime note:** the full example can take a long time (up to a day or two depending on your machine).
- **Reproducibility:** if `geopy` fails, copy the provided coefficients from `amazon_china_data.py` and use them in the `parameters()` function of `Online_Fulfillment+Placement.py`.


## <span style="color:blue"> Contact </span>



For questions or issues, please contact **Irem Nur Keskin** — iremnurkeskin@gmail.com
