import pandas as pd
import numpy as np

def preprocess_new_customer(customer_data):
    """
    Converts raw customer spending data into a log-scaled DataFrame.
    
    Parameters:
    customer_data (dict): A dictionary containing the raw spending amounts 
                          for the 6 product categories.
                          
    Returns:
    pd.DataFrame: A single-row DataFrame with logarithmically scaled values, 
                  ready for PCA transformation.
    """
    expected_columns = [
        'Fresh', 
        'Milk', 
        'Grocery', 
        'Frozen', 
        'Detergents_Paper', 
        'Delicatessen'
    ]
    df = pd.DataFrame([customer_data])

    df = df[expected_columns]

    log_scaled_data = np.log(df)

    return log_scaled_data