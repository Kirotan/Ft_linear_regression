import matplotlib.pyplot as plt
import sys

try:
    from train import load_data
except ImportError:
    print("Erreur : Impossible d'importer 'train.py'. Vérifie qu'il est dans le dossier.")
    sys.exit()

try:
    from predict import load_model
except ImportError:
    print("Erreur : Impossible d'importer 'predict.py'.")
    sys.exit()


# Petite fonction locale pour le calcul final (car on ne l'a pas isolée ailleurs)
def make_prediction(mileage, theta0, theta1, km_min, km_max, price_min, price_max):
    mileage_norm = (mileage - km_min) / (km_max - km_min)
    price_norm = theta0 + (theta1 * mileage_norm)
    return price_norm * (price_max - price_min) + price_min


if __name__ == "__main__":
    try:
        # --- A. Récupération des données (via train.py) ---
        print("Récupération des données via train.py...")
        km, price = load_data("data.csv")
        
        # --- B. Récupération du modèle (via predict.py) ---
        print("Récupération des thétas via predict.py...")
        theta0, theta1, km_min, km_max, price_min, price_max = load_model("thetas.csv")
        
        # --- C. Création du graphique ---
        plt.figure(figsize=(10, 6))
        
        # Les points réels
        plt.plot(km, price, 'ro', label='Données réelles')
        
        # La ligne de prédiction
        # On calcule juste le point de départ (min) et de fin (max) pour tracer la ligne
        line_x = [km_min, km_max]
        line_y = [
            make_prediction(km_min, theta0, theta1, km_min, km_max, price_min, price_max),
            make_prediction(km_max, theta0, theta1, km_min, km_max, price_min, price_max)
        ]
        
        plt.plot(line_x, line_y, 'b-', linewidth=3, label='Prédiction')
        
        plt.title("Prix d'une voiture selon son kilométrage")
        plt.xlabel("Kilométrage (km)")
        plt.ylabel("Prix (€)")
        plt.legend()
        plt.grid(True)
        
        print("Affichage du graphique !")
        plt.show()

    except Exception as e:
        print(f"Une erreur est survenue : {e}")