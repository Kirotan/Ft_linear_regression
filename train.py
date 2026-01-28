import csv
import sys

def estimate_price(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)

def load_data(filename):
    mileage = []
    prices = []

    try:
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader)
            for row in reader:
                if row:
                    km_value = float(row[0])
                    price_value = float(row[1])
                    mileage.append(km_value)
                    prices.append(price_value)

    except FileNotFoundError:
        print(f"Error : File {filename} is nowhere to be find.")
        sys.exit(1) # exit program properly
    except ValueError:
        print("Error : Value in file is not a valid number.")
        sys.exit(1)

    return mileage, prices


def normalize(data):
    min_val = min(data)
    max_val = max(data)

    normalized_data = []

    for x in data:
        if max_val == min_val: # avoid division per 0
            normalized_data.append(0)
        else:
            res = (x - min_val) / (max_val - min_val)
            normalized_data.append(res)

    return normalized_data, min_val, max_val

def train_model(km_list, price_list):
    theta0 = 0.0
    theta1 = 0.0

    # hyperparameters
    learning_rate = 0.1
    iteration = 2300

    m = len(km_list) # number total of examples

    # --- Learning loop ---
    for i in range(iteration):
        sum_errors_t0 = 0
        sum_errors_t1 = 0

        for j in range(m):
            prediction = estimate_price(km_list[j], theta0, theta1) # calcul with currents thetas
            error = prediction - price_list[j]  # calcul interval with reality
            sum_errors_t0 += error # add sum for theta0
            sum_errors_t1 += error * km_list[j] # add sum for theta1 weighted by km

        # calculation of temporary values (average * learningRate)
        tmp_theta0 = learning_rate * (1/m) * sum_errors_t0
        tmp_theta1 = learning_rate * (1/m) * sum_errors_t1    

        # theta's simultanous update (soustraction because it's a slope)
        theta0 -= tmp_theta0
        theta1 -= tmp_theta1

        if i % 100 == 0:
            print(f"Itération {i}: theta0={theta0:.4f}, theta1={theta1:.4f}")

    return theta0, theta1





# --- TEST DU CODE ---
if __name__ == "__main__":
    # On appelle la fonction
    km, price = load_data("data.csv")
    
    # On affiche juste pour vérifier que ça marche
    print("--- Vérification chargement des donnees ---")
    print("Données chargées !")
    print(f"Nombre de voitures : {len(km)}")
    print(f"Premier kilométrage : {km[0]}")
    print(f"Premier prix : {price[0]}")

    # On normalise les deux listes
    # On récupère les 3 valeurs renvoyées dans 3 variables distinctes
    km_norm, km_min, km_max = normalize(km)
    price_norm, price_min, price_max = normalize(price)


    print("--- Vérification nornalisation ---")
    print(f"Km Min: {km_min}, Km Max: {km_max}")
    print(f"Price Min: {price_min}, Price Max: {price_max}")
    print(f"Premier km normalisé : {km_norm[0]}")
    print(f"Max KM normalisé : {max(km_norm)}") # Doit être égal à 1.0
    print(f"Min KM normalisé : {min(km_norm)}") # Doit être égal à 0.0


    print("--- Vérification de l'entrainement ---")
    # 1. Chargement
    print("Chargement des données...")
    km, price = load_data("data.csv")
    
    # 2. Normalisation
    print("Normalisation...")
    km_norm, km_min, km_max = normalize(km)
    price_norm, price_min, price_max = normalize(price)

    # 3. Entraînement
    print("Entraînement en cours...")
    theta0_norm, theta1_norm = train_model(km_norm, price_norm)

    print("\n--- RÉSULTATS ---")
    print(f"Theta0 final (normalisé) : {theta0_norm}")
    print(f"Theta1 final (normalisé) : {theta1_norm}")