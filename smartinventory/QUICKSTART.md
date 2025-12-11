# Quick Start Guide

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. (Optional) Install Gurobi for better optimization performance:
   - Download from https://www.gurobi.com/
   - Academic licenses are free for students
   - The app will automatically use SciPy as fallback if Gurobi is not available

## Running the Application

```bash
cd smartinventory
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Using Sample Data

1. Check "Use Sample Data" in the sidebar
2. Configure forecast horizon (4-12 weeks)
3. Select optimization model (Model A or Model B)
4. Optionally set budget and capacity constraints
5. Click "Run Optimization"

## Using Your Own Data

### Sales History CSV Format:
```csv
date,sku,demand
2024-01-01,SKU001,45
2024-01-08,SKU001,52
...
```

### SKU Parameters CSV Format:
```csv
sku,unit_cost,holding_cost,stockout_penalty,volume,fixed_order_cost
SKU001,25.50,2.50,50.00,0.5,100.00
SKU002,12.00,1.20,30.00,1.2,150.00
...
```

Note: `fixed_order_cost` is optional and defaults to 0 if not provided.

## Troubleshooting

- **Import errors**: Make sure all dependencies are installed (`pip install -r requirements.txt`)
- **Optimization fails**: Check that your data files have the correct format and required columns
- **No Gurobi**: The app will automatically use SciPy, but results may be slightly different (continuous relaxation of binary variables)

