"""
Visualization functions for inventory optimization results.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Optional


def create_order_quantity_chart(
    order_quantities: Dict[str, float],
    demand_forecasts: Optional[Dict[str, float]] = None
) -> go.Figure:
    """
    Create horizontal bar chart showing recommended order quantities.
    
    Args:
        order_quantities: Dictionary mapping SKU to order quantity
        demand_forecasts: Optional dictionary mapping SKU to demand forecast
        
    Returns:
        Plotly figure object
    """
    # Prepare data
    skus = list(order_quantities.keys())
    quantities = [order_quantities[sku] for sku in skus]
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    # Order quantities
    fig.add_trace(go.Bar(
        y=skus,
        x=quantities,
        orientation='h',
        name='Order Quantity',
        marker_color='#2E86AB',
        text=[f'{q:.1f}' for q in quantities],
        textposition='outside'
    ))
    
    # Add demand forecast line if provided
    if demand_forecasts:
        demands = [demand_forecasts.get(sku, 0) for sku in skus]
        fig.add_trace(go.Scatter(
            y=skus,
            x=demands,
            mode='markers',
            name='Demand Forecast',
            marker=dict(symbol='diamond', size=12, color='#F24236'),
            hovertemplate='Demand: %{x:.1f}<extra></extra>'
        ))
    
    fig.update_layout(
        title='Recommended Order Quantities by SKU',
        xaxis_title='Quantity',
        yaxis_title='SKU',
        height=400,
        showlegend=True,
        hovermode='closest'
    )
    
    return fig


def create_cost_breakdown_chart(
    sku_data: pd.DataFrame,
    order_quantities: Dict[str, float],
    shortages: Dict[str, float]
) -> go.Figure:
    """
    Create stacked bar chart showing cost breakdown by SKU.
    
    Args:
        sku_data: DataFrame with SKU parameters
        order_quantities: Dictionary mapping SKU to order quantity
        shortages: Dictionary mapping SKU to shortage amount
        
    Returns:
        Plotly figure object
    """
    # Calculate costs for each SKU
    cost_data = []
    
    for _, row in sku_data.iterrows():
        sku = row['sku']
        Q = order_quantities.get(sku, 0)
        s = shortages.get(sku, 0)
        
        purchasing_cost = row['unit_cost'] * Q
        fixed_cost = row['fixed_order_cost'] if Q > 0 else 0
        holding_cost = row['holding_cost'] * (Q / 2)
        shortage_cost = row['stockout_penalty'] * s
        
        cost_data.append({
            'SKU': sku,
            'Purchasing': purchasing_cost,
            'Fixed Order': fixed_cost,
            'Holding': holding_cost,
            'Shortage': shortage_cost
        })
    
    df_costs = pd.DataFrame(cost_data)
    
    # Create stacked bar chart
    fig = go.Figure()
    
    colors = {
        'Purchasing': '#2E86AB',
        'Fixed Order': '#A23B72',
        'Holding': '#F18F01',
        'Shortage': '#C73E1D'
    }
    
    for cost_type in ['Purchasing', 'Fixed Order', 'Holding', 'Shortage']:
        fig.add_trace(go.Bar(
            name=cost_type,
            x=df_costs['SKU'],
            y=df_costs[cost_type],
            marker_color=colors[cost_type]
        ))
    
    fig.update_layout(
        title='Cost Breakdown by SKU',
        xaxis_title='SKU',
        yaxis_title='Cost ($)',
        barmode='stack',
        height=400,
        hovermode='x unified'
    )
    
    return fig


def create_results_table(
    sku_data: pd.DataFrame,
    order_quantities: Dict[str, float],
    shortages: Dict[str, float],
    binding_constraints: Dict[str, bool]
) -> pd.DataFrame:
    """
    Create results table with all key metrics.
    
    Args:
        sku_data: DataFrame with SKU parameters and demand forecasts
        order_quantities: Dictionary mapping SKU to order quantity
        shortages: Dictionary mapping SKU to shortage amount
        binding_constraints: Dictionary indicating which constraints are binding
        
    Returns:
        DataFrame with results
    """
    results = []
    
    for _, row in sku_data.iterrows():
        sku = row['sku']
        Q = order_quantities.get(sku, 0)
        s = shortages.get(sku, 0)
        D = row.get('demand_forecast', 0)
        
        purchasing = row['unit_cost'] * Q
        fixed = row['fixed_order_cost'] if Q > 0 else 0
        holding = row['holding_cost'] * (Q / 2)
        shortage = row['stockout_penalty'] * s
        total_cost = purchasing + fixed + holding + shortage
        
        results.append({
            'SKU': sku,
            'Demand Forecast': f'{D:.1f}',
            'Order Quantity (Q)': f'{Q:.1f}',
            'Shortage': f'{s:.1f}',
            'Purchasing Cost': f'${purchasing:.2f}',
            'Fixed Cost': f'${fixed:.2f}',
            'Holding Cost': f'${holding:.2f}',
            'Shortage Cost': f'${shortage:.2f}',
            'Total Cost': f'${total_cost:.2f}'
        })
    
    df_results = pd.DataFrame(results)
    
    return df_results

