import pandas as pd
from datetime import datetime

class InventoryDB:
    def __init__(self):
        self.df = pd.DataFrame()
        print("✅ Inventory database initialized.")

    def insert_row(self, row_dict):
        """
        Inserta una fila nueva en el inventario.
        row_dict: diccionario donde las claves son columnas y los valores son los datos.
        """
        # Convertir el diccionario a DataFrame de 1 fila
        row_df = pd.DataFrame([row_dict])

        # Concatenar con el DataFrame interno, agregando columnas dinámicamente
        self.df = pd.concat([self.df, row_df], ignore_index=True)

    def show(self):
        """
        Muestra el DataFrame de inventario actual.
        """
        return self.df
