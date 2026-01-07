import pandas as pd
from datetime import datetime

class InventoryDB:
    def __init__(self):
        self.df = pd.DataFrame()
        print("✅ Inventory database initialized.")

    def show(self):
        """
        Muestra el DataFrame de inventario actual.
        """
        print("Current Inventory Database:"
              f"\n{self.df}")
        return self.df