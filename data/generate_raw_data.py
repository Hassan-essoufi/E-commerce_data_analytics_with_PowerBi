import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from faker import Faker

# Initialisation avec plusieurs locales
fake_fr = Faker('fr_FR')
fake_en = Faker('en_US')
fake_de = Faker('de_DE')
fake_es = Faker('es_ES')
fake_it = Faker('it_IT')

np.random.seed(42)
random.seed(42)

# Mapping pays -> locale Faker
country_locales = {
    'France': fake_fr,
    'United States': fake_en,
    'Germany': fake_de,
    'Spain': fake_es,
    'Italy': fake_it,
    'United Kingdom': fake_en,
    'Belgium': fake_fr,
    'Switzerland': fake_de,
    'Canada': fake_en,
    'Portugal': fake_es
}

# Distribution des pays (60% France, 40% autres)
country_distribution = {
    'France': 0.6,
    'United States': 0.1,
    'Germany': 0.08,
    'Spain': 0.07,
    'Italy': 0.05,
    'United Kingdom': 0.04,
    'Belgium': 0.03,
    'Switzerland': 0.02,
    'Canada': 0.01
}

def get_faker_for_country(country):
    """Retourne l'instance Faker appropriée pour le pays"""
    return country_locales.get(country, fake_fr)

def generate_customers_data(n_customers=5000):
    """
    Génère des données clients réalistes avec cohérence pays-ville
    """
    customers = []
    
    countries = list(country_distribution.keys())
    probabilities = list(country_distribution.values())
    
    for i in range(n_customers):
        # Sélectionner un pays selon la distribution
        country = np.random.choice(countries, p=probabilities)
        faker = get_faker_for_country(country)
        
        customer_id = f"CUST_{10000 + i}"
        registration_date = faker.date_between(start_date='-3y', end_date='today')
        
        customers.append({
            'customer_id': customer_id,
            'first_name': faker.first_name(),
            'last_name': faker.last_name(),
            'email': faker.email(),
            'phone': faker.phone_number(),
            'birth_date': faker.date_of_birth(minimum_age=18, maximum_age=80),
            'registration_date': registration_date,
            'country': country,
            'city': faker.city(),
            'postal_code': faker.postcode(),
            'address': faker.address().replace('\n', ', '),
            'segment': np.random.choice(['Premium', 'Standard', 'Basic'], p=[0.2, 0.5, 0.3]),
            'newsletter_subscribed': np.random.choice([True, False], p=[0.6, 0.4])
        })
    
    return pd.DataFrame(customers)

def generate_products_data(n_products=200):
    """
    Génère un catalogue produits réaliste
    """
    categories = {
        'Électronique': ['Smartphone', 'Ordinateur', 'Tablette', 'Casque Audio', 'Smartwatch'],
        'Mode': ['Vêtement Homme', 'Vêtement Femme', 'Chaussures', 'Accessoires', 'Sacs'],
        'Maison': ['Meuble', 'Décoration', 'Électroménager', 'Luminaire', 'Cuisine'],
        'Sport': ['Vélo', 'Running', 'Fitness', 'Yoga', 'Randonnée'],
        'Loisirs': ['Livre', 'Jeux Vidéo', 'Instrument', 'Jardinage', 'Bricolage']
    }
    
    products = []
    
    for i in range(n_products):
        category = np.random.choice(list(categories.keys()))
        subcategory = np.random.choice(categories[category])
        cost_price = round(np.random.uniform(10, 500), 2)
        margin = np.random.uniform(0.2, 0.8)
        selling_price = round(cost_price * (1 + margin), 2)
        
        products.append({
            'product_id': f"PROD_{5000 + i}",
            'product_name': f"{subcategory} {fake_fr.word().capitalize()} {fake_fr.random_int(100, 999)}",
            'category': category,
            'subcategory': subcategory,
            'brand': fake_fr.company(),
            'cost_price': cost_price,
            'selling_price': selling_price,
            'stock_quantity': np.random.randint(0, 1000),
            'min_stock': np.random.randint(5, 50),
            'creation_date': fake_fr.date_between(start_date='-2y', end_date='today'),
            'is_active': np.random.choice([True, False], p=[0.9, 0.1]),
            'weight_kg': round(np.random.uniform(0.1, 20), 2),
            'supplier': fake_fr.company()
        })
    
    return pd.DataFrame(products)

def generate_orders_data(n_orders=20000, customers_df=None, products_df=None):
    """
    Génère des données de commandes réalistes avec cohérence pays-ville
    """
    orders = []
    order_items = []
    
    customer_ids = customers_df['customer_id'].tolist()
    product_ids = products_df['product_id'].tolist()
    product_prices = products_df.set_index('product_id')['selling_price'].to_dict()
    
    # Créer un mapping customer_id -> pays pour la cohérence
    customer_country_map = customers_df.set_index('customer_id')['country'].to_dict()
    customer_city_map = customers_df.set_index('customer_id')['city'].to_dict()
    
    order_id_counter = 100000
    
    for i in range(n_orders):
        order_id = f"ORDER_{order_id_counter + i}"
        customer_id = np.random.choice(customer_ids)
        
        # Récupérer le pays et la ville du client pour la cohérence
        customer_country = customer_country_map[customer_id]
        customer_city = customer_city_map[customer_id]
        faker = get_faker_for_country(customer_country)
        
        order_date = faker.date_time_between(start_date='-2y', end_date='now')
        
        # Statut de commande réaliste basé sur la date
        days_since_order = (datetime.now() - order_date).days
        if days_since_order < 2:
            status = 'Processing'
        elif days_since_order < 7:
            status = np.random.choice(['Shipped', 'Delivered'], p=[0.3, 0.7])
        else:
            status = 'Delivered'
        
        # Générer 1 à 5 articles par commande
        n_items = np.random.randint(1, 6)
        total_amount = 0
        
        for j in range(n_items):
            product_id = np.random.choice(product_ids)
            quantity = np.random.randint(1, 4)
            unit_price = product_prices[product_id]
            item_total = unit_price * quantity
            total_amount += item_total
            
            order_items.append({
                'order_id': order_id,
                'product_id': product_id,
                'quantity': quantity,
                'unit_price': unit_price,
                'item_total': item_total
            })
        
        # Ajouter des frais de livraison (plus élevés pour l'international)
        domestic_countries = ['France', 'Belgium', 'Switzerland']
        if customer_country in domestic_countries:
            shipping_cost = round(np.random.uniform(0, 4.99), 2) if np.random.random() > 0.3 else 0
        else:
            shipping_cost = round(np.random.uniform(5.99, 19.99), 2)
        
        orders.append({
            'order_id': order_id,
            'customer_id': customer_id,
            'order_date': order_date,
            'total_amount': round(total_amount + shipping_cost, 2),
            'shipping_cost': shipping_cost,
            'status': status,
            'payment_method': np.random.choice(['Credit Card', 'PayPal', 'Bank Transfer', 'Cash on Delivery'], 
                                            p=[0.6, 0.2, 0.1, 0.1]),
            'shipping_country': customer_country,
            'shipping_city': customer_city,
            'shipping_address': faker.address().replace('\n', ', ')
        })
    
    return pd.DataFrame(orders), pd.DataFrame(order_items)

def generate_geolocation_data(customers_df):
    """
    Génère des données géolocalisées cohérentes avec les pays des clients
    """
    geolocation = []
    
    # Créer des points géographiques par ville et pays
    unique_locations = customers_df[['city', 'country']].drop_duplicates()
    
    # Coordonnées approximatives par pays
    country_coordinates = {
        'France': {'lat_range': (42, 51), 'lon_range': (-5, 9)},
        'United States': {'lat_range': (25, 49), 'lon_range': (-125, -67)},
        'Germany': {'lat_range': (47, 55), 'lon_range': (6, 15)},
        'Spain': {'lat_range': (36, 44), 'lon_range': (-9, 3)},
        'Italy': {'lat_range': (36, 47), 'lon_range': (6, 18)},
        'United Kingdom': {'lat_range': (50, 59), 'lon_range': (-8, 2)},
        'Belgium': {'lat_range': (49, 51), 'lon_range': (2, 6)},
        'Switzerland': {'lat_range': (46, 48), 'lon_range': (6, 10)},
        'Canada': {'lat_range': (42, 70), 'lon_range': (-141, -53)}
    }
    
    for _, location in unique_locations.head(150).iterrows():  # Limiter à 150 villes
        city = location['city']
        country = location['country']
        
        if country in country_coordinates:
            coords = country_coordinates[country]
            latitude = round(np.random.uniform(coords['lat_range'][0], coords['lat_range'][1]), 6)
            longitude = round(np.random.uniform(coords['lon_range'][0], coords['lon_range'][1]), 6)
        else:
            # Coordonnées par défaut pour l'Europe
            latitude = round(np.random.uniform(40, 60), 6)
            longitude = round(np.random.uniform(-10, 20), 6)
        
        geolocation.append({
            'city': city,
            'country': country,
            'latitude': latitude,
            'longitude': longitude,
            'population': np.random.randint(10000, 2000000)
        })
    
    return pd.DataFrame(geolocation)

def generate_returns_data(orders_df, order_items_df, n_returns=500):
    """
    Génère des données de retours réalistes
    """
    returns = []
    
    # Sélectionner des commandes livrées pour les retours
    delivered_orders = orders_df[orders_df['status'] == 'Delivered']
    
    if len(delivered_orders) < n_returns:
        n_returns = len(delivered_orders)
    
    selected_orders = delivered_orders.sample(n_returns, random_state=42)
    
    for _, order in selected_orders.iterrows():
        return_date = order['order_date'] + timedelta(days=np.random.randint(1, 30))
        
        returns.append({
            'return_id': f"RET_{fake_fr.random_int(10000, 99999)}",
            'order_id': order['order_id'],
            'customer_id': order['customer_id'],
            'return_date': return_date,
            'return_reason': np.random.choice([
                'Defective Product', 
                'Wrong Size', 
                'Does Not Match Description',
                'Changed Mind',
                'Delivery Too Late'
            ]),
            'return_amount': round(order['total_amount'] * np.random.uniform(0.3, 1.0), 2),
            'return_status': np.random.choice(['Processed', 'In Progress', 'Refunded'], p=[0.6, 0.2, 0.2]),
            'refund_method': np.random.choice(['Original Card', 'Store Credit', 'Bank Transfer'])
        })
    
    return pd.DataFrame(returns)

def generate_website_traffic_data(n_days=730):
    """
    Génère des données de trafic web réalistes
    """
    traffic = []
    start_date = datetime.now() - timedelta(days=n_days)
    
    for i in range(n_days):
        current_date = start_date + timedelta(days=i)
        
        # Variation saisonnière
        month = current_date.month
        seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * (month - 1) / 12)
        
        # Effet weekend
        is_weekend = current_date.weekday() >= 5
        weekend_factor = 1.2 if is_weekend else 1.0
        
        base_visitors = 1000
        visitors = int(base_visitors * seasonal_factor * weekend_factor * np.random.uniform(0.8, 1.2))
        
        traffic.append({
            'date': current_date.date(),
            'visitors': visitors,
            'page_views': int(visitors * np.random.uniform(3, 8)),
            'unique_visitors': int(visitors * np.random.uniform(0.7, 0.9)),
            'bounce_rate': round(np.random.uniform(0.3, 0.6), 3),
            'avg_session_duration': round(np.random.uniform(120, 480), 1),
            'conversion_rate': round(np.random.uniform(0.01, 0.05), 4),
            'traffic_source': np.random.choice(['Direct', 'SEO', 'Social Media', 'Email', 'Paid Search'], 
                                            p=[0.3, 0.25, 0.2, 0.15, 0.1])
        })
    
    return pd.DataFrame(traffic)

def add_realistic_noise_and_issues(dfs_dict):
    """
    Ajoute des problèmes réalistes aux données pour le nettoyage
    """
    # Ajouter des valeurs manquantes
    for df_name, df in dfs_dict.items():
        if df_name == 'customers':
            # 5% d'emails manquants
            mask = np.random.random(len(df)) < 0.05
            df.loc[mask, 'email'] = None
            
            # 3% de numéros de téléphone manquants
            mask = np.random.random(len(df)) < 0.03
            df.loc[mask, 'phone'] = None
        
        elif df_name == 'orders':
            # 2% d'adresses de livraison manquantes
            mask = np.random.random(len(df)) < 0.02
            df.loc[mask, 'shipping_address'] = None
    
    return dfs_dict

def save_raw_data(dfs_dict, output_dir='data/raw'):
    """
    Sauvegarde tous les DataFrames en fichiers CSV
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for df_name, df in dfs_dict.items():
        filename = f"{output_dir}/{df_name}.csv"
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"{filename} sauvegardé ({len(df)} lignes)")

def main():
    """
    Fonction principale pour générer toutes les données
    """
    print("Génération des données brutes internationales...")
    
    # Génération des données
    print("Génération des clients...")
    customers_df = generate_customers_data(5000)
    
    print("Génération des produits...")
    products_df = generate_products_data(200)
    
    print("Génération des commandes...")
    orders_df, order_items_df = generate_orders_data(20000, customers_df, products_df)
    
    print("Génération des données géographiques...")
    geolocation_df = generate_geolocation_data(customers_df)
    
    print("Génération des retours...")
    returns_df = generate_returns_data(orders_df, order_items_df, 500)
    
    print("Génération du trafic web...")
    traffic_df = generate_website_traffic_data(730)
    
    # Préparation du dictionnaire de données
    dfs_dict = {
        'customers': customers_df,
        'products': products_df,
        'orders': orders_df,
        'order_items': order_items_df,
        'geolocation': geolocation_df,
        'returns': returns_df,
        'website_traffic': traffic_df
    }
    
    # Ajouter du bruit réaliste
    print("Ajout de problèmes réalistes...")
    dfs_dict = add_realistic_noise_and_issues(dfs_dict)
    
    # Sauvegarde
    print("Sauvegarde des fichiers...")
    save_raw_data(dfs_dict)
    
    # Résumé
    print("\n" + "="*50)
    print("RÉSUMÉ DES DONNÉES GÉNÉRÉES (INTERNATIONAL)")
    print("="*50)
    for df_name, df in dfs_dict.items():
        print(f"• {df_name}: {len(df):,} lignes, {len(df.columns)} colonnes")
    
    # Distribution des pays
    country_dist = customers_df['country'].value_counts()
    print(f"\nDistribution des pays clients:")
    for country, count in country_dist.items():
        percentage = (count / len(customers_df)) * 100
        print(f"  {country}: {count} clients ({percentage:.1f}%)")
    
    print(f"\n Génération terminée! Données sauvegardées dans data/raw/")

if __name__ == "__main__":
    main()