import pandas as pd
from datetime import date

def create_temporal_features(df_orders, date_col="date"):
    """
    Creates temporal features from a date column.
    Generated features:
    - day_of_week
    - month
    - quarter
    - weekend
    - holidays (simple placeholder)
    """
    df_out = df_orders.copy()
    df_out[date_col] = pd.to_datetime(df_out[date_col], errors="coerce")

    # Day of the week (0 = Monday, 6 = Sunday)
    df_out["day_of_week"] = df_out[date_col].dt.weekday

    # Month (1 to 12)
    df_out["month"] = df_out[date_col].dt.month

    # Quarter of the year (1 to 4)
    df_out["quarter"] = df_out[date_col].dt.quarter

    # Weekend indicator (1 if Saturday or Sunday, else 0)
    df_out["weekend"] = df_out["day_of_week"].isin([5, 6]).astype(int)

    # Holidays
    df_out["holidays"] = (df_out["month"] == 8).astype(int)

    return df_out

def create_customer_features(df_customers, df_orders):
    """
    Creates advanced customer features:
    - customer seniority
    - customer age
    - RFM metrics (Recency, Frequency, Monetary)
    - loyalty status segment
    """
    df_customers['registration_date'] = pd.to_datetime(df_customers['registration_date'])
    df_orders['order_date'] = pd.to_datetime(df_orders['order_date'])
    reference_date = pd.Timestamp(2025, 11, 19)


    # Customer seniority
    df_customers['seniority_days'] = (reference_date - df_customers['signup_date']).dt.days
    
    # RFM Feature Calculation
    rfm = df_orders.groupby('customer_id').agg({
        'order_date': lambda x: (reference_date - x.max()).days,  # Recency
        'order_id': 'count',                                      # Frequency
        'order_amount': 'sum'                                     # Monetary
    }).reset_index()

    rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']

    # Loyalty status assignment
    def assign_loyalty(monetary_value):
        if monetary_value > 1000:
            return "Gold"
        elif monetary_value > 500:
            return "Silver"
        else:
            return "Bronze"

    rfm['loyalty_status'] = rfm['monetary'].apply(assign_loyalty)

    # final dataframe
    df_out = df_customers.merge(rfm, on='customer_id', how='left')

    return df_out

def create_product_features(df_products, df_items, df_returns):
    """
    Create product-level features:
    - number of commands
    - total quantity sold
    - total sales amount
    - observed average price
    - returned quantity
    - return rate %
    """
    df_products = df_products.copy()
    df_items = df_items.copy()
    df_returns = pd.merge(df_returns, df_items, on='order_id', how='left')

    # Categories conversion
    df_products = pd.get_dummies(df_products, columns=['category'], prefix=['cat'], drop_first=False)

    # Product metrics
    df_metrics = df_items.groupby('product_id').agg(
    n_commands=('order_id', 'count'),
    quantity=('quantity', 'sum'),
    total_price=('item_total', 'sum'),
    price_mean_observed=('unit_price',
                         lambda s: (s * df_items.loc[s.index, 'quantity']).sum()
                                   / df_items.loc[s.index, 'quantity'].sum())).reset_index()

    # Returned quantities per product
    quantity_return = df_returns.groupby('product_id').agg({
        'quantity': 'sum',
    }).reset_index()
    quantity_return.columns = ['product_id', 'ret_commands']
    df_out= pd.merge(df_metrics, quantity_return, on='product_id', how='left')

    # Return ratio %
    df_out['ratio_pct'] = df_out.apply(
        lambda row: row['ret_commands'] / row['quantity'] * 100 
        if row['quantity'] != 0 else 0,
        axis=1
    ).round(1)

    # Final dataframe
    df_out = pd.merge(df_products, df_out, on='product_id', how='left')

    return df_out

def create_geographic_features(df_geolocation):
    """
    Crée des features géographiques
    """
    df_out = df_geolocation.copy()

    # regions
    region_map = {
        'France': 'Western Europe',
        'Germany': 'Western Europe',
        'Belgium': 'Western Europe',
        'United Kingdom': 'Western Europe',
        'Switzerland': 'Central Europe',
        'Italy': 'Southern Europe',
        'Spain': 'Southern Europe',
        'United States': 'North America',
        'Canada': 'North America'
    }
    df_out['region_group'] = df_out['country'].map(region_map)

    # Population class
    df_out['population_class'] = df_out['population'].apply(
        lambda p: 'large' if p > 1_000_000 
        else 'medium' if p >= 100_000 
        else 'small'
    )

    # Hemisphere (based on latitude)
    df_out['hemisphere'] = df_out['latitude'].apply(
        lambda lat: 'North' if lat >= 0 else 'South'
    )

    # Number of cities per country
    country_city_count = df_out.groupby('country')['city'].nunique().reset_index()
    country_city_count.columns = ['country', 'country_city_count']

    df_final = df_out.merge(country_city_count, on='country', how='left')

    return df_final

def create_business_kpis(df_orders, df_visits):
    """
    Compute key business KPIs:
    - Total Revenue (CA)
    - Average Order Value (AOV)
    - Conversion Rate
    """

    # Total Revenue (CA)
    revenue = df_orders['total_amount'].sum()

    # Tatal commands
    n_orders = df_orders['order_id'].nunique()

    # Average Order Value (AOV)
    avg_order_value = revenue / n_orders if n_orders > 0 else 0

    # Website visitors
    n_visitors = df_visits['visitors'].sum()

    # Conversion rate
    conversion_rate = (n_orders / n_visitors * 100) if n_visitors > 0 else 0

    return {
        'total_revenue': round(revenue, 2),
        'order_count': n_orders,
        'average_order_value': round(avg_order_value, 2),
        'conversion_rate_%': round(conversion_rate, 2)
    }


# create_all_features()