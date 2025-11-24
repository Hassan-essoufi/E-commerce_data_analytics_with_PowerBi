import numpy as np 
import pandas as pd
from sklearn.ensemble import IsolationForest
from scipy import stats



def clean_customer_data(df_customers):
    """
    Nettoie et standardise les données clients
    """
    df_clean = df_customers.copy()
    
    # emails cleaning
    df_clean['email']= df_clean['email'].astype(str)
    df_clean['email']= df_clean['email'].str.strip()
    df_clean['email']= df_clean['email'].str.lower()
    df_clean.loc[df_clean['email'].isin(['', 'nan', 'none', 'null']), 'email'] = None

    # phones cleaning
    df_clean['phone'] = df_clean['phone'].astype(str).str.strip()
    df_clean['phone'] = df_clean['phone'].str.replace(r'[^\d+]', '', regex=True)
    df_clean.loc[df_clean['phone'].isin(['', 'nan', 'none', 'null']), 'phone'] = None
   
    # phone normalization 
    def normalize_phone(row):
        phone = row['phone']
        country = row['country']
        
        if pd.isna(phone) or pd.isna(country):
            return phone, 'invalide'
        
        # Déjà au format international
        if str(phone).startswith('+'):
            return phone, 'valide'
        
        country_codes = {
            'France': '33', 'United States': '1', 'Germany': '49',
            'Spain': '34', 'Italy': '39', 'United Kingdom': '44',
            'Belgium': '32', 'Switzerland': '41', 'Canada': '1'
        }

        if country not in country_codes:
            return phone, 'invalide'
        
        # FRANCE
        if country == 'France':
            if phone.startswith('0') and len(phone) == 10:
                return '+33' + phone[1:], 'valide'
            elif phone.startswith('33') and len(phone) == 11:
                return '+' + phone, 'valide'
            elif len(phone) == 9:
                return '+33' + phone, 'valide'
        
        # USA / CANADA
        elif country in ['United States', 'Canada']:
            if phone.startswith('1') and len(phone) == 11:
                return '+' + phone, 'valide'
            elif len(phone) == 10:
                return '+1' + phone, 'valide'
        
        # UK
        elif country == 'United Kingdom':
            if phone.startswith('0') and len(phone) == 11:
                return '+44' + phone[1:], 'valide'
            elif phone.startswith('44') and len(phone) == 12:
                return '+' + phone, 'valide'
        
        # GERMANY
        elif country == 'Germany':
            if phone.startswith('0') and len(phone) == 11:
                return '+49' + phone[1:], 'valide'
            elif phone.startswith('49') and len(phone) == 12:
                return '+' + phone, 'valide'
   
        # ESPAGNE
        elif country == 'Spain':
            if phone.startswith('0') and len(phone) == 9:
                return '+34' + phone[1:], 'valide'
            elif phone.startswith('34') and len(phone) == 11:
                return '+' + phone, 'valide'
        
        # ITALIE
        elif country == 'Italy':
            if phone.startswith('0') and len(phone) == 10:
                return '+39' + phone[1:], 'valide'
            elif phone.startswith('39') and len(phone) == 12:
                return '+' + phone, 'valide'
        
        # BELGIQUE
        elif country == 'Belgium':
            if phone.startswith('0') and len(phone) == 9:
                return '+32' + phone[1:], 'valide'
            elif phone.startswith('32') and len(phone) == 11:
                return '+' + phone, 'valide'
        
        # SUISSE
        elif country == 'Switzerland':
            if phone.startswith('0') and len(phone) == 9:
                return '+41' + phone[1:], 'valide'
            elif phone.startswith('41') and len(phone) == 11:
                return '+' + phone, 'valide'
        
        return phone, 'invalide'  
    
    phone_results = df_clean.apply(normalize_phone, axis=1)
    df_clean['phone'] = phone_results.apply(lambda x: x[0])
    df_clean['phone_valid'] = phone_results.apply(lambda x: x[1])
    
    # final phones validation
    def validate_phone_final(phone):
        if pd.isna(phone):
            return None, 'invalide'
        
        phone_str = str(phone)
        if (phone_str.startswith('+') and 
            len(phone_str) >= 10 and 
            len(phone_str) <= 15 and
            phone_str[1:].isdigit()):  
            return phone_str, 'valide'
        else:
            return None, 'invalide'
        
    final_results = df_clean['phone'].apply(validate_phone_final)
    df_clean['phone'] = final_results.apply(lambda x: x[0])
    df_clean['phone_valid'] = final_results.apply(lambda x: x[1])

    # dates cleaning 
    for col in ['birth_date','registration_date']:
        df_clean[col] = df_clean[col].astype(str).str.strip()
        df_clean.loc[df_clean[col].isin(['', 'nan', 'None', 'null', 'None']), col] = None
        df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce', dayfirst=False)

    # Adding ages
    current_date = pd.Timestamp(2025, 11, 19)
    mask_valid = df_clean['birth_date'].notna()
    df_clean .loc[mask_valid, 'age'] = ((current_date - df_clean.loc[mask_valid, 'birth_date']).dt.days // 365).astype(int)
    df_clean['birth_date'] = df_clean['birth_date'].dt.date.where(mask_valid, None)

    # postal_code conversion
    df_clean['postal_code'] = df_clean['postal_code'].astype(str)

    return df_clean

def clean_product_data(df_products):
    """
    Nettoie le catalogue produits
    """
    df_clean = df_products.copy()

    # Prices cleaning & positives values
    columns = ['cost_price','selling_price','stock_quantity', 'min_stock', 'weight_kg'] 
    for col in columns:
        df_clean[col] = df_clean[col].astype(str).str.strip()
        df_clean.loc[df_clean[col].isin(['', 'nan',  'None','NaN' , 'null', 'Null', 'none']), col] = None
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        df_clean.loc[df_clean[col] <= 0 , col] = None
    
    # Dates cleaning
    df_clean['creation_date'] = df_clean['creation_date'].astype(str).str.strip()
    df_clean.loc[df_clean['creation_date'].isin(['', 'nan', 'None', 'null', 'None']), 'creation_date'] = None
    df_clean['creation_date'] = pd.to_datetime(df_clean['creation_date'], errors='coerce', dayfirst=False)


    return df_clean
     
    
def clean_transaction_data(df_orders):
    """
    Nettoie les données de transactions
    """
    df_clean = df_orders.copy()

    # Positives values
    columns = ['total_amount','shipping_cost']
    for col in columns:
        df_clean[col] = df_clean[col].astype(str).str.strip()
        df_clean.loc[df_clean[col].isin(['', 'nan',  'None','NaN' , 'null', 'Null', 'none']), col] = None
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        df_clean.loc[df_clean[col] <= 0 , col] = None
    
    # Dates cleaning
    df_clean['order_date'] = df_clean['order_date'].astype(str).str.strip()
    df_clean.loc[df_clean['order_date'].isin(['', 'nan', 'None', 'null', 'None']), 'order_date'] = None
    df_clean['order_date'] = pd.to_datetime(df_clean['order_date'], errors='coerce', dayfirst=False)
    mask_valid = df_clean['order_date'].notna()
    df_clean['order_date'] = df_clean['order_date'].dt.strftime('%Y-%m-%d %H:%M:%S').where(mask_valid, 'manquant')

    # payment methods 
    df_clean['payment_method'] = df_clean['payment_method'].astype(str).str.strip().str.lower()
    df_clean.loc[df_clean['payment_method'].isin(['', 'nan', 'null', 'none']), 'payment_method'] = None
    payment_mapping = {
    'credit card': 'Credit Card',
    'paypal': 'PayPal',
    'bank transfer': 'Bank Transfer',
    'cash on delivery': 'Cash on Delivery'}
    df_clean['payment_method'] = df_clean['payment_method'].map(payment_mapping).fillna(df_clean['payment_method'])

    return df_clean

def handle_missing_values(df,data_type, numeric_strategy, categorical_strategy):
    """
    Gère les valeurs manquantes selon stratégies
    """
    # Missing values summary
    missing_summary = pd.DataFrame({
        'missing_count': df.isna().sum(),
        'missing_percent': df.isna().mean() * 100
    })
    missing_summary = missing_summary[missing_summary['missing_count'] > 0]
    df_clean = df.copy()
    if data_type == 'customer':
        # Colonnes numériques
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            if numeric_strategy == 'median':
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            elif numeric_strategy == 'mean':
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
        # Colonnes catégorielles
        for col in df_clean.select_dtypes(include=['object', 'category']).columns:
            if categorical_strategy == 'mode':
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
            elif categorical_strategy == 'constant':
                df_clean[col].fillna('Unknown', inplace=True)
    elif data_type == 'product':
        # Colonnes numériques
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
        # Colonnes catégorielles
        for col in df_clean.select_dtypes(include=['object', 'category']).columns:
            df_clean[col].fillna('Unknown', inplace=True)

    # Création des flags pour les valeurs imputées
    df_flags = pd.DataFrame(index=df.index)
    for col in df.columns:
        flag_col = f"{col}_imputed"
        df_flags[flag_col] = np.where(df[col].isna() & df_clean[col].notna(), 1, 0)

    df_final = pd.concat([df_clean, df_flags], axis=1)

    return df_final, missing_summary

def detect_outliers(df, method="IQR", columns=None, contamination=0.05):
    """
    Outliers detection
    
    method : "IQR" | "Z-score" | "IsolationForest"
    columns : liste des colonnes à analyser (numériques uniquement)
    contamination : proportion pour IsolationForest
    """
    
    df_out = df.copy()
    
    # Numeric columns selection if none
    if columns is None:
        columns = df_out.select_dtypes(include=[np.number]).columns.tolist()

    # IQR method
    if method == "IQR":
        for col in columns:
            Q1 = df_out[col].quantile(0.25)
            Q3 = df_out[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            flag_name = f"outlier_{col}_iqr"
            df_out[flag_name] = ((df_out[col] < lower) | (df_out[col] > upper)).astype(int)

    # Z-SCORE method
    elif method == "Z-score":
        for col in columns:
            z_scores = stats.zscore(df_out[col], nan_policy='omit')
            flag_name = f"outlier_{col}_z"
            df_out[flag_name] = (np.abs(z_scores) > 3).astype(int)

    # Isolation forest method
    elif method == "IsolationForest":
        clf = IsolationForest(contamination=contamination, random_state=42)
        preds = clf.fit_predict(df_out[columns])
        df_out['outlier_iforest'] = (preds == -1).astype(int)

    else:
        raise ValueError("Unknown method")

    return df_out

