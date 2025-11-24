import pandas as pd
import numpy as np
import os
from datetime import datetime
import re

def load_file(file_path):
    try :
        df = pd.read_csv(file_path)
        return df 
    except FileNotFoundError:

        return 'Fichier non trouve'
    except Exception as e :
        return None

def load_raw_datasets():
    datasets = {}
    required_files = ['customers.csv', 'products.csv', 'orders.csv', 'order_items.csv', 'geolocation.csv', 'returns.csv']

    for filename in required_files:
        file_path = os.path.join('data/raw', filename)
        df_name = filename.replace('.csv', '')
        df = load_file-file_path

        if df is not None :
            datasets[df_name] = df
    return datasets
                   
def check_columns_presence(df, required_columns):
    columns = df.columns.tolist()
    miss_columns = []

    for col in required_columns:
        if col not in columns :
            miss_columns.append(col)

    if len(miss_columns) == 0 :
        return "all columns are presented"
    else : 
        return f"missing columns: {miss_columns}"  
    
def check_column_types(df, column_type_mapping):

    errors = []

    for col, expected_type in column_type_mapping.items():

        # missing columns
        if col not in df.columns:
            errors.append({
                "column": col,
                "expected": str(expected_type),
                "found": "MISSING"
            })
            continue

        # specific type datetme
        if expected_type == "datetime":
            if not (pd.api.types.is_datetime64_any_dtype(df[col])):
                errors.append({
                    "column": col,
                    "expected": "datetime",
                    "found": str(df[col].dtype)
                })
            continue

        # types comparaison
        actual_dtype = df[col].dtype

        if expected_type == int and not pd.api.types.is_integer_dtype(actual_dtype):
            errors.append({
                "column": col,
                "expected": "int",
                "found": str(actual_dtype)
            })

        elif expected_type == float and not pd.api.types.is_float_dtype(actual_dtype):
            errors.append({
                "column": col,
                "expected": "float",
                "found": str(actual_dtype)
            })

        elif expected_type == bool and not pd.api.types.is_bool_dtype(actual_dtype):
            errors.append({
                "column": col,
                "expected": "bool",
                "found": str(actual_dtype)
            })

        elif expected_type == str and not pd.api.types.is_string_dtype(actual_dtype):
            errors.append({
                "column": col,
                "expected": "string",
                "found": str(actual_dtype)
            })
    
    if len(errors) == 0:
        return "tout est valide"
    else : 
        return "erreur: {errors}"

def check_missing_values(df):
    columns = {}
    missing_values = df.isnull().sum()
    for col, value in missing_values.items():
        if value != 0:
            columns[col] = value
    return columns 

def check_duplicates(df):
    duplicated_mask = df.duplicated(subset=None, keep=False)
    duplicated_rows = df[duplicated_mask]
    result = {'nbr_duplicate':duplicated_mask.sum()}
    
    if len(duplicated_rows)!= 0:
        result['duplicated_row'] = duplicated_rows
    return result



def check_value_constraints(df, constraints):
    errors = {}

    for col, rules in constraints.items():
        if col not in df.columns:
            errors[col] = "Column not found"
            continue

        col_errors = []

        # Min constraint
        if "min" in rules:
            invalid = df[df[col] < rules["min"]].index.tolist()
            if invalid:
                col_errors.append({"min_violation": invalid})

        # Max constraint
        if "max" in rules:
            invalid = df[df[col] > rules["max"]].index.tolist()
            if invalid:
                col_errors.append({"max_violation": invalid})

        # Allowed values
        if "allowed" in rules:
            invalid = df[~df[col].isin(rules["allowed"])].index.tolist()
            if invalid:
                col_errors.append({"allowed_violation": invalid})

        # Regex constraint
        if "regex" in rules:
            pattern = re.compile(rules["regex"])
            invalid = df[~df[col].astype(str).str.match(pattern)].index.tolist()
            if invalid:
                col_errors.append({"regex_violation": invalid})

        # Condition constraint
        if "condition" in rules:
            invalid = df[~df[col].apply(rules["condition"])].index.tolist()
            if invalid:
                col_errors.append({"condition_violation": invalid})

        if col_errors:
            errors[col] = col_errors

    return errors

def validate_data_schemas(df, columns_type_mapping,constraints):
    report = {}

    # missing columns verfication
    required_columns = [col for col in columns_type_mapping.keys()]
    missing_columns = check_columns_presence(df, required_columns)
    report['missing_columns'] = missing_columns
    
    # column types verification
    types_errors = check_column_types(df, columns_type_mapping)
    report['types_errors'] = types_errors

    # missing values detection
    missing_values = check_missing_values(df)
    report['missing_values'] = missing_values

    #duplicates detection
    duplicates = check_duplicates(df)
    report['duplicates'] = duplicates

    #constraints verification
    value_constraints = check_value_constraints(df, constraints)
    report['value_constraints'] = value_constraints

    return report

def merge_datasets(df1, df2,label,):
    result = pd.merge(df1, df2, on=label, how='inner')
    return result

















        



    


