import pandas as pd 


def perform_cohort_analysis(df_orders):
    """
    Cohort analysis for client retention 
    """
    df = df_orders.copy()

    # Date conversion
    df['order_date'] = pd.to_datetime(df['order_date'])

    # Year-month orders 
    df['order_month'] = df['order_date'].dt.to_period('M')

    # 1st command per client
    cohort = df.groupby('customer_id')['order_month'].min().reset_index()
    cohort.columns = ['customer_id', 'cohort_month']

    df = df.merge(cohort, on='customer_id', how='left')
    
    # cohort index
    df['cohort_index'] = (
        (df['order_month'].dt.year - df['cohort_month'].dt.year) * 12 +
        (df['order_month'].dt.month - df['cohort_month'].dt.month)
    )

    # 
    cohort_counts = df.groupby(['cohort_month', 'cohort_index'])['customer_id'].nunique().reset_index()

    # 7️⃣ Pivot en matrice cohortes
    cohort_matrix = cohort_counts.pivot_table(
        index='cohort_month',
        columns='cohort_index',
        values='customer_id'
    )

    # 8️⃣ Calcul des taux de rétention (%)
    retention = cohort_matrix.divide(cohort_matrix.iloc[:, 0], axis=0).round(3)

    return retention
    


def customer_lifetime_value_analysis(df_customers, df_orders):
    """
    Calcule et analyse le CLV
    """
    # CLV historique, prédictif, segmentation

def market_basket_analysis(df_orders):
    """
    Analyse du panier d'achat
    """
    # règles association, produits fréquents

def time_series_decomposition(df_orders):
    """
    Décompose les séries temporelles
    """
    # tendance, saisonnalité, résidus

def customer_segmentation_clustering(df_customers):
    """
    Segmentation clients par clustering
    """
    # K-means, RFM clustering, profils

def predictive_analysis_churn():
    """
    Analyse prédictive du churn (basique)
    """
    # probabilité attrition, clients à risque