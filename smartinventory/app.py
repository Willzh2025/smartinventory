"""
SmartInventory Planner - Streamlit Application
Prescriptive Analytics Inventory Optimization Tool
"""

import streamlit as st
import pandas as pd
import os
from pathlib import Path

# Import project modules
from data_loader import prepare_optimization_data, forecast_demand_moving_average, load_sales_data
from optimizer import solve_model_a_mip, solve_model_b_service_level
from visualization import create_order_quantity_chart, create_cost_breakdown_chart, create_results_table
from insights import generate_insights

# Page configuration
st.set_page_config(
    page_title="SmartInventory Planner",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and header
st.title("📦 SmartInventory Planner")
st.markdown("### Prescriptive Analytics for Optimal Inventory Management")

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'sku_data' not in st.session_state:
    st.session_state.sku_data = None
if 'demand_forecasts' not in st.session_state:
    st.session_state.demand_forecasts = None

# Sidebar
st.sidebar.header("⚙️ Configuration")

# Data source selection
st.sidebar.subheader("📁 Data Input")

data_source = st.sidebar.radio(
    "Data Source",
    ["Use Sample Data", "Upload Files"],
    index=0,
    help="Choose to use sample data or upload your own files"
)

sales_file = None
sku_file = None

if data_source == "Upload Files":
    # Show file uploaders only when "Upload Files" is selected
    st.sidebar.markdown("---")
    sales_file = st.sidebar.file_uploader(
        "Upload Sales History CSV",
        type=['csv'],
        help="CSV with columns: date, sku, demand"
    )
    
    sku_file = st.sidebar.file_uploader(
        "Upload SKU Parameters CSV",
        type=['csv'],
        help="CSV with columns: sku, unit_cost, holding_cost, stockout_penalty, volume, fixed_order_cost"
    )
else:
    # Use sample data
    sample_dir = Path(__file__).parent / "sample_data"
    sample_sales = sample_dir / "sales_history_sample.csv"
    sample_sku = sample_dir / "sku_costs_sample.csv"
    
    if sample_sales.exists() and sample_sku.exists():
        sales_file = sample_sales
        sku_file = sample_sku
        st.sidebar.success("✓ Using sample data")

# Optimization parameters
st.sidebar.subheader("📊 Forecasting")
forecast_horizon = st.sidebar.slider(
    "Forecast Horizon (weeks)",
    min_value=4,
    max_value=12,
    value=4,
    step=1,
    help="Number of weeks to forecast demand"
)

st.sidebar.subheader("🎯 Optimization Model")
model_choice = st.sidebar.radio(
    "Select Model",
    ["Model A: Cost Minimization MIP", "Model B: Service-Level Feasibility"],
    help="Model A: Full MIP with fixed costs. Model B: Simplified service-level focus."
)

# Optional constraints
st.sidebar.subheader("🔒 Constraints (Optional)")
use_budget = st.sidebar.checkbox("Enable Budget Constraint", value=False)
budget = None
if use_budget:
    budget = st.sidebar.number_input(
        "Budget Limit ($)",
        min_value=0.0,
        value=5000.0,
        step=100.0
    )

use_capacity = st.sidebar.checkbox("Enable Storage Capacity Constraint", value=False)
capacity = None
if use_capacity:
    capacity = st.sidebar.number_input(
        "Storage Capacity (volume units)",
        min_value=0.0,
        value=100.0,
        step=10.0
    )

# Run optimization button
run_optimization = st.sidebar.button("🚀 Run Optimization", type="primary", use_container_width=True)

# Main area
if run_optimization or st.session_state.results is not None:
    if sales_file is None or sku_file is None:
        st.error("❌ Please upload both sales history and SKU parameters files, or enable 'Use Sample Data'.")
    else:
        try:
            with st.spinner("Loading data and running optimization..."):
                # Prepare data
                demand_forecasts, sku_data = prepare_optimization_data(
                    sales_file,
                    sku_file,
                    forecast_horizon
                )
                
                # Run optimization
                if "Model A" in model_choice:
                    order_quantities, shortages, total_cost, binding_constraints = solve_model_a_mip(
                        sku_data,
                        budget=budget,
                        capacity=capacity
                    )
                else:  # Model B
                    order_quantities, shortages, total_cost, binding_constraints = solve_model_b_service_level(
                        sku_data,
                        budget=budget
                    )
                
                # Store results
                st.session_state.results = {
                    'order_quantities': order_quantities,
                    'shortages': shortages,
                    'total_cost': total_cost,
                    'binding_constraints': binding_constraints
                }
                st.session_state.sku_data = sku_data
                st.session_state.demand_forecasts = demand_forecasts
            
            # Display results
            st.success("✅ Optimization completed successfully!")
            
            # Summary insights
            st.subheader("💡 Key Insights")
            
            total_order_value = sum(
                sku_data[sku_data['sku'] == sku]['unit_cost'].values[0] * q
                for sku, q in order_quantities.items()
            )
            total_shortage = sum(shortages.values())
            num_skus_ordered = sum(1 for q in order_quantities.values() if q > 0)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Cost", f"${total_cost:,.2f}")
            with col2:
                st.metric("Total Order Value", f"${total_order_value:,.2f}")
            with col3:
                st.metric("Total Shortage", f"{total_shortage:.1f} units")
            with col4:
                st.metric("SKUs to Order", f"{num_skus_ordered}/{len(order_quantities)}")
            
            # Binding constraints info
            if binding_constraints:
                binding_info = []
                if binding_constraints.get('budget', False):
                    binding_info.append("💰 Budget constraint is binding")
                if binding_constraints.get('capacity', False):
                    binding_info.append("📦 Capacity constraint is binding")
                if binding_info:
                    st.info(" | ".join(binding_info))
            
            # Visualizations - side by side
            st.subheader("📊 Optimization Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_orders = create_order_quantity_chart(
                    order_quantities,
                    demand_forecasts
                )
                st.plotly_chart(fig_orders, use_container_width=True)
            
            with col2:
                fig_costs = create_cost_breakdown_chart(
                    sku_data,
                    order_quantities,
                    shortages
                )
                st.plotly_chart(fig_costs, use_container_width=True)
            
            # Results table
            st.subheader("📋 Detailed Results")
            results_table, results_table_raw = create_results_table(
                sku_data,
                order_quantities,
                shortages,
                binding_constraints
            )
            st.dataframe(results_table, use_container_width=True, hide_index=True)
            
            # Calculate total volume for capacity utilization
            total_volume = sum(
                sku_data[sku_data['sku'] == sku]['volume'].values[0] * q
                for sku, q in order_quantities.items()
            )
            
            # Insights & Recommendations
            insights_text = generate_insights(
                results_table,
                results_table_raw,
                budget=budget,
                total_cost=total_cost,
                capacity=capacity,
                total_volume=total_volume,
                model_choice=model_choice
            )
            st.markdown("## Insights & Recommendations")
            st.markdown(insights_text)
            
            # Download results
            st.subheader("💾 Export Results")
            csv = results_table.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="inventory_optimization_results.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)

else:
    # Welcome screen
    st.info("👈 Configure your optimization parameters in the sidebar and click 'Run Optimization' to get started.")
    
    # Instructions
    with st.expander("📖 How to Use"):
        st.markdown("""
        ### Getting Started
        
        1. **Upload Data Files** (or use sample data):
           - **Sales History CSV**: Historical sales data with columns `date`, `sku`, `demand`
           - **SKU Parameters CSV**: Product parameters with columns `sku`, `unit_cost`, `holding_cost`, `stockout_penalty`, `volume`, `fixed_order_cost`
        
        2. **Configure Forecasting**:
           - Select forecast horizon (4-12 weeks)
           - The app uses moving average to predict future demand
        
        3. **Choose Optimization Model**:
           - **Model A (Cost Minimization MIP)**: Full mixed-integer program with fixed order costs
           - **Model B (Service-Level Feasibility)**: Simplified model focused on minimizing shortages
        
        4. **Set Constraints** (optional):
           - Budget limit: Maximum total purchasing cost
           - Storage capacity: Maximum total volume
        
        5. **Run Optimization**:
           - Click "Run Optimization" to get prescriptive recommendations
        
        ### Understanding the Results
        
        - **Order Quantities**: Recommended order amount for each SKU
        - **Cost Breakdown**: Detailed cost components (purchasing, fixed, holding, shortage)
        - **Binding Constraints**: Indicates which constraints are limiting the solution
        """)
    
    # Sample data preview
    if data_source == "Use Sample Data":
        sample_dir = Path(__file__).parent / "sample_data"
        sample_sales = sample_dir / "sales_history_sample.csv"
        sample_sku = sample_dir / "sku_costs_sample.csv"
        
        if sample_sales.exists() and sample_sku.exists():
            st.subheader("📊 Sample Data Preview")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Sales History Sample**")
                df_sales = pd.read_csv(sample_sales)
                st.dataframe(df_sales.head(10), use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("**SKU Parameters Sample**")
                df_sku = pd.read_csv(sample_sku)
                st.dataframe(df_sku, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>SmartInventory Planner | ISOM 839 - Prescriptive Analytics</div>",
    unsafe_allow_html=True
)

