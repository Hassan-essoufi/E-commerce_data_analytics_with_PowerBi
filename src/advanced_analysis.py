from mlxtend.frequent_patterns import apriori, association_rules
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,accuracy_score
from sklearn.cluster import KMeans
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
    cohort_counts = df.groupby(['cohort_month', 'cohort_index'])['customer_id'].nunique().reset_index()

    # cohort Matrix
    cohort_matrix = cohort_counts.pivot_table(
        index='cohort_month',
        columns='cohort_index',
        values='customer_id'
    )

    # Retention ratio (%)
    retention = (cohort_matrix.divide(cohort_matrix.iloc[:, 0], axis=0)*100).round(3)

    return retention

def customer_lifetime_value_analysis(df_customers, df_orders):
    """
    Calculate & analyse Customer Lifetime Value (CLV)

    """
    df_customers = df_customers.copy()
    df_orders = df_orders.copy()

    # Historical CLV 
    clv_hist = df_orders.groupby('customer_id')['total_amount'].sum().reset_index()
    clv_hist.rename(columns={'total_amount': 'clv_historical'}, inplace=False)

    # Commands frequency
    freq = df_orders.groupby('customer_id')['order_id'].nunique().reset_index()
    freq.rename(columns={'order_id': 'purchase_frequency'}, inplace=True)

    # Average Order Value
    avg_order_value = (df_orders.groupby('customer_id')['total_amount'].sum() /
                       df_orders.groupby('customer_id')['order_id'].nunique()).reset_index()
    avg_order_value.rename(columns={0: 'avg_order_value'}, inplace=True)

    df_clv = df_customers.merge(clv_hist, on='customer_id', how='left')
    df_clv = df_clv.merge(freq, on='customer_id', how='left')
    df_clv = df_clv.merge(avg_order_value, on='customer_id', how='left')

    # Basic CLV predective
    estimated_lifespan_years = 3
    df_clv['clv_predicted'] = (
        df_clv['avg_order_value'] *
        df_clv['purchase_frequency'] *
        estimated_lifespan_years
    )

    # Client segmentation
    df_clv['segment'] = pd.qcut(
        df_clv['clv_predicted'],
        q=4,
        labels=['Low Value', 'Mid-Low', 'Mid-High', 'High Value']
    )

    return df_clv

def market_basket_analysis(df_orders):
    """
    Analyse du panier d'achat :
    - Trouve les produits fréquemment achetés ensemble
    - Génère des règles d'association
    """
    # Préparation du panier
    basket = df_orders.groupby(['order_id', 'product_id'])['product_id'] \
        .count().unstack().fillna(0)
    
    # One-hot encoding booléen
    basket = basket.applymap(lambda x: x > 0)

    # Itemsets fréquents
    frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)

    # Règles d'association
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    rules = rules.sort_values('lift', ascending=False).reset_index(drop=True)

    return frequent_itemsets, rules

def time_series_decomposition(df_orders):
    """
    Décompose la série temporelle des ventes :
    - tendance
    - saisonnalité
    - résidus
    """
    
    df = df_orders.copy()
    df['order_date'] = pd.to_datetime(df['order_date'])

    # Create daily time series
    ts = df.groupby('order_date')['total_amount'].sum()

    # Decomposition (additive)
    decomposition = seasonal_decompose(ts, model='additive', period=30)

    return {
        'observed': decomposition.observed,
        'trend': decomposition.trend,
        'seasonal': decomposition.seasonal,
        'residual': decomposition.resid
    }

def customer_segmentation_clustering(df_customers):
    """
    Segmentation clients par clustering (RFM + KMeans)
    """
    df = df_customers.copy()

    # RFM Features 
    rfm = df.groupby('customer_id').agg({
        'recency': 'mean',          
        'frequency': 'sum',
        'monetary': 'sum'
    }).reset_index()

    # Normalisation
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['recency', 'frequency', 'monetary']])

    # K-Means (4 clusters)
    kmeans = KMeans(n_clusters=4, random_state=42)
    rfm['cluster'] = kmeans.fit_predict(rfm_scaled)

    # cluster profiles
    cluster_profiles = rfm.groupby('cluster').mean()

    return rfm, cluster_profiles

def predictive_analysis_churn(df, target_column='churn'):
    """
    Basic churn predictive analysis
    """
    
    # Separate target and features
    y = df[target_column]
    X = df.drop(columns=[target_column])

    # Encode categorical features
    X = pd.get_dummies(X, drop_first=True)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Normalize numeric features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train a logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Print performance metrics
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return model

