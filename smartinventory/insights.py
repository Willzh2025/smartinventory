"""
Insights and Recommendations Generator
Produces structured, Markdown-formatted business insights from optimization results.
"""

import pandas as pd
from typing import Optional


def generate_insights(
    results_df: pd.DataFrame,
    results_df_raw: pd.DataFrame,
    budget: Optional[float] = None,
    total_cost: Optional[float] = None,
    capacity: Optional[float] = None,
    total_volume: Optional[float] = None,
    model_choice: Optional[str] = None
) -> str:
    """
    Generates structured, human-readable insights based on optimization results.
    Returns a Markdown-formatted multi-section string.
    
    Args:
        results_df: DataFrame with optimization results
        budget: Optional budget constraint value
        total_cost: Optional total cost from optimization
        capacity: Optional capacity constraint value
        total_volume: Optional total volume used
        model_type:
            "Model A" → Cost Minimization MIP (budget includes purchasing + fixed)
            "Model B" → Service-Level Model (budget includes purchasing only)
            
    Returns:
        Markdown-formatted string with insights
    """
    # Helper function to parse numeric values from formatted strings
    def parse_float(val):
        if isinstance(val, str):
            return float(val.replace('$', '').replace(',', '').strip())
        return float(val)
    
    # Create a working copy with numeric columns
    df = results_df.copy()
    df_raw = results_df_raw.copy()

    # Parse numeric columns (handle formatted strings)
    df['Order_Qty'] = df['Order Quantity (Q)'].apply(parse_float)
    df['Demand_Numeric'] = df['Demand Forecast'].apply(parse_float)
    df['Shortage_Numeric'] = df['Shortage'].apply(parse_float)
    df['Purchasing_Cost_Numeric'] = df['Purchasing Cost'].apply(parse_float)
    df['Fixed_Cost_Numeric'] = df['Fixed Cost'].apply(parse_float)
    
    # Calculate total purchasing cost and total budget used (purchasing + fixed)
    total_purchase = df['Purchasing_Cost_Numeric'].sum()
    total_fixed = df['Fixed_Cost_Numeric'].sum()
    total_budget_used = total_purchase + total_fixed

    # ------------------------------------------------------------------------------
    # Fix Note (2025-12-11):
    # Previously, the budget utilization was computed using values from
    # `results_table`, where costs were stored as formatted strings (e.g., "$123.45").
    # These values had already been rounded for display, which introduced small
    # numerical discrepancies when summing totals and calculating budget percentages.
    #
    # To ensure full numerical accuracy, the calculation below now uses
    # `results_table_raw`, which contains the true unformatted float values produced
    # directly from the optimization model.
    #
    # total_purchase      = sum of (unit_cost * order_quantity) for all SKUs  
    # total_fixed         = sum of fixed_order_cost for all SKUs with Q > 0  
    # total_budget_used   = total_purchase + total_fixed  
    #
    # This guarantees consistency with the optimization model's internal constraints
    # and prevents misleading discrepancies in budget utilization due to display
    # formatting or rounding.
    # ------------------------------------------------------------------------------

    total_purchase = df_raw['Purchasing Cost'].sum()
    total_fixed = df_raw['Fixed Cost'].sum()
    total_budget_used = total_purchase + total_fixed
    
    # A. Budget Utilization
    budget_used_pct = None
    budget_limit = budget
    if budget is not None and total_budget_used is not None and budget > 0:
        if "Model A" in model_choice:
            budget_used_pct = (total_budget_used / budget) * 100
        else:  # Model B's constraint: budget = purchasing ONLY
            budget_used_pct = (total_purchase / budget) * 100
    
    # B. Capacity Utilization
    capacity_used_pct = None
    capacity_limit = capacity
    if capacity is not None and total_volume is not None and capacity > 0:
        capacity_used_pct = (total_volume / capacity) * 100
    
    # C. SKUs with zero order quantity
    zero_order_skus = df[df['Order_Qty'] == 0]['SKU'].tolist()
    
    # D. Highest-demand SKUs
    top_demand = df.nlargest(3, 'Demand_Numeric')[['SKU', 'Demand_Numeric']].copy()
    
    # E. Shortages
    skus_with_shortage = df[df['Shortage_Numeric'] > 0]
    top_shortage = skus_with_shortage.nlargest(3, 'Shortage_Numeric')[['SKU', 'Shortage_Numeric']].copy()
    
    # F. Purchasing cost concentration
    df['_temp_purchase'] = df['Purchasing Cost'].apply(parse_float)
    
    # Filter only SKUs that were actually purchased (Q > 0 OR purchasing cost > 0)
    df_nonzero_purchase = df[df['_temp_purchase'] > 0].copy()

    if len(df_nonzero_purchase) > 0:
        # Take the top 3 actual spending SKUs
        top_purchase = df_nonzero_purchase.nlargest(3, '_temp_purchase')[['SKU', 'Purchasing Cost']].copy()
    else:
        # No purchasing at all
        top_purchase = pd.DataFrame(columns=['SKU', 'Purchasing Cost'])

    # top_purchase = df.nlargest(3, '_temp_purchase')[['SKU', 'Purchasing Cost']].copy()
    top_purchase_sum = top_purchase['Purchasing Cost'].apply(parse_float).sum()
    pct_purchase = (top_purchase_sum / total_purchase * 100) if total_purchase > 0 else 0
    df.drop('_temp_purchase', axis=1, inplace=True)
    
    # Build insights string
    insights_parts = []
    
    # 1. Budget & Capacity Utilization
    insights_parts.append("1. Budget & Capacity Utilization")
    if budget_used_pct is not None:
        # Show purchasing cost in the dollar amount as per user format
        # insights_parts.append(f"   - Budget used: {budget_used_pct:.1f}% (\\${total_budget_used:,.2f} of \\${budget_limit:,.2f})")
        if "Model A" in model_choice:
            # Model A budget = purchasing + fixed
            insights_parts.append(f"   - Budget used: {budget_used_pct:.1f}% (\\${total_budget_used:,.2f} of \\${budget_limit:,.2f})")
            insights_parts.append(f"      - Includes purchasing (\\${total_purchase:,.2f}) + fixed order cost (\\${total_fixed:,.2f})")
        else:
            # Model B budget = purchasing ONLY
            insights_parts.append(f"   - Budget used: {budget_used_pct:.1f}% (\\${total_purchase:,.2f} of \\${budget_limit:,.2f})")
            insights_parts.append(f"      - Note: Model B does NOT include fixed order cost in optimization.")
            insights_parts.append(f"      - Actual spending including fixed cost: "f"\\${(total_purchase + total_fixed):,.2f} = purchasing cost (\\${total_purchase:,.2f}) + fixed order cost (\\${total_fixed:,.2f})")
    if capacity_used_pct is not None:
        insights_parts.append(f"   - Capacity used: {capacity_used_pct:.1f}% ({total_volume:,.0f} of {capacity_limit:,.0f} units)")
    
    # Interpretation
    if budget_used_pct is not None and capacity_used_pct is not None:
        if budget_used_pct > 95 and capacity_used_pct > 95:
            insights_parts.append("   - Interpretation: Both budget and capacity are fully utilized. Consider expanding resources or prioritizing higher-margin items.")
        elif budget_used_pct > 95:
            insights_parts.append("   - Interpretation: Budget is fully utilized. Additional budget would allow for more orders and reduced shortages.")
        elif capacity_used_pct > 95:
            insights_parts.append("   - Interpretation: Storage capacity is fully utilized. Consider expanding warehouse space or focusing on high-margin, low-volume items.")
        elif budget_used_pct < 50 and capacity_used_pct < 50:
            insights_parts.append("   - Interpretation: Both budget and capacity are underutilized. Consider increasing order quantities to better meet demand.")
        else:
            insights_parts.append("   - Interpretation: Resource utilization is moderate. Current allocation balances cost and service level effectively.")
    elif budget_used_pct is not None:
        if budget_used_pct > 95:
            insights_parts.append("   - Interpretation: Budget is fully utilized. Additional budget would allow for more orders and reduced shortages.")
        elif budget_used_pct < 50:
            insights_parts.append("   - Interpretation: Budget is underutilized. Consider increasing order quantities to better meet demand.")
        else:
            insights_parts.append("   - Interpretation: Budget utilization is moderate. Current allocation balances cost and service level effectively.")
    elif capacity_used_pct is not None:
        if capacity_used_pct > 95:
            insights_parts.append("   - Interpretation: Storage capacity is fully utilized. Consider expanding warehouse space or focusing on high-margin, low-volume items.")
        elif capacity_used_pct < 50:
            insights_parts.append("   - Interpretation: Storage capacity is underutilized. Consider increasing order quantities to better meet demand.")
        else:
            insights_parts.append("   - Interpretation: Capacity utilization is moderate. Current allocation balances cost and service level effectively.")
    else:
        insights_parts.append("   - Interpretation: No budget or capacity constraints were set.")
    
    insights_parts.append("")
    
    # 2. Ordering Decisions
    insights_parts.append("2. Ordering Decisions")
    if zero_order_skus:
        sku_list = ", ".join(zero_order_skus[:10])  # Limit to first 10 for readability
        if len(zero_order_skus) > 10:
            sku_list += f" and {len(zero_order_skus) - 10} more"
        insights_parts.append(f"   - SKUs not recommended for ordering: {sku_list}")
        
        # Analyze reasons
        zero_order_df = df[df['Order_Qty'] == 0]
        avg_demand = zero_order_df['Demand_Numeric'].mean() if len(zero_order_df) > 0 else 0
        avg_cost = zero_order_df['Purchasing_Cost_Numeric'].mean() if len(zero_order_df) > 0 else 0
        
        if avg_demand < 10 and avg_cost > 100:
            reason = "Low demand combined with high costs make these items unprofitable under current constraints."
        elif avg_demand < 10:
            reason = "Low forecasted demand relative to fixed order costs makes ordering uneconomical."
        elif avg_cost > 500:
            reason = "High purchasing costs relative to available budget make these items unaffordable."
        else:
            reason = "Fixed order costs and budget constraints make ordering these items suboptimal."
        
        insights_parts.append(f"   - Reason: {reason}")
        insights_parts.append("   - Suggested action: Review these SKUs for potential discontinuation or renegotiate supplier terms to reduce fixed costs.")
    else:
        insights_parts.append("   - SKUs not recommended for ordering: None")
        insights_parts.append("   - Reason: All SKUs received positive order quantities based on demand and cost optimization.")
        insights_parts.append("   - Suggested action: Current ordering plan is comprehensive. Monitor demand patterns for future adjustments.")
    
    insights_parts.append("")
    
    # 3. Demand & Shortages
    insights_parts.append("3. Demand & Shortages")
    
    # Highest forecasted demand
    if not top_demand.empty:
        demand_list = []
        for _, row in top_demand.iterrows():
            demand_list.append(f"{row['SKU']} ({row['Demand_Numeric']:.0f} units)")
        insights_parts.append(f"   - Highest forecasted demand: {', '.join(demand_list)}")
    else:
        insights_parts.append("   - Highest forecasted demand: No demand data available")
    
    # SKUs with shortages
    shortage_count = len(skus_with_shortage)
    insights_parts.append(f"   - SKUs with shortages: {shortage_count}")
    
    # Largest expected shortages
    if not top_shortage.empty:
        shortage_list = []
        for _, row in top_shortage.iterrows():
            shortage_list.append(f"{row['SKU']} ({row['Shortage_Numeric']:.0f} units)")
        insights_parts.append(f"   - Largest expected shortages: {', '.join(shortage_list)}")
    else:
        insights_parts.append("   - Largest expected shortages: None")
    
    # Interpretation
    if shortage_count == 0:
        insights_parts.append("   - Interpretation: All forecasted demand can be met with the current order plan. No stockouts expected.")
    elif shortage_count <= 2:
        insights_parts.append("   - Interpretation: Minor shortages expected in a few items. Consider prioritizing these SKUs if budget allows.")
    else:
        insights_parts.append("   - Interpretation: Multiple items will experience shortages. Prioritize high-demand, high-margin items to minimize revenue impact.")
    
    insights_parts.append("")
    
    # 4. Cost Concentration
    insights_parts.append("4. Cost Concentration")
    if not top_purchase.empty and total_purchase > 0:
        purchase_list = []
        for _, row in top_purchase.iterrows():
            cost_val = parse_float(row['Purchasing Cost'])
            purchase_list.append(f"{row['SKU']} (\\${cost_val:,.2f})")
        insights_parts.append(f"   - Primary cost drivers: {', '.join(purchase_list)}")
        insights_parts.append(f"   - Share of total purchasing cost: {pct_purchase:.1f}%")
        
        if pct_purchase > 70:
            insights_parts.append("   - Interpretation: Purchasing costs are highly concentrated in a few items. Monitor these SKUs closely as they drive most of the budget allocation.")
        elif pct_purchase > 50:
            insights_parts.append("   - Interpretation: Purchasing costs show moderate concentration. Top items represent a significant portion of total spend.")
        else:
            insights_parts.append("   - Interpretation: Purchasing costs are well distributed across SKUs, reducing risk from individual item price changes.")
    else:
        insights_parts.append("   - Primary cost drivers: No purchasing cost data available")
        insights_parts.append("   - Share of total purchasing cost: N/A")
        insights_parts.append("   - Interpretation: Unable to analyze cost concentration.")
    
    insights_parts.append("")
    
    # 5. Recommended Actions
    insights_parts.append("5. Recommended Actions")
    actions = []
    
    if budget_used_pct is not None and budget_used_pct > 95:
        actions.append("Consider increasing budget allocation to reduce shortages and improve service levels.")
    
    if capacity_used_pct is not None and capacity_used_pct > 95:
        actions.append("Evaluate warehouse expansion or prioritize high-margin, low-volume SKUs to optimize space utilization.")
    
    if shortage_count > 0:
        actions.append("Prioritize ordering for high-demand items with shortages to minimize lost sales opportunities.")
    
    if len(zero_order_skus) > 0:
        actions.append("Review zero-order SKUs for potential product line rationalization or supplier renegotiation.")
    
    if pct_purchase > 70:
        actions.append("Diversify purchasing across more SKUs to reduce concentration risk and improve portfolio resilience.")
    
    if not actions:
        actions.append("Continue monitoring demand patterns and adjust order quantities based on actual sales performance.")
        actions.append("Review cost structure periodically to identify opportunities for optimization.")
        actions.append("Maintain current ordering strategy while tracking key performance indicators.")
    
    # Ensure we have 3-4 actions
    if len(actions) < 3:
        actions.append("Conduct regular review of SKU performance to identify optimization opportunities.")
    
    for i, action in enumerate(actions[:4], 1):  # Limit to 4 actions
        insights_parts.append(f"   - {action}")
    
    return "\n".join(insights_parts)
