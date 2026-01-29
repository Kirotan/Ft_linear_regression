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


def save_model(theta0, theta1, km_min, km_max, price_min, price_max, filename="thetas.csv"):
    with open(filename, 'w') as file:
        file.write(f"{theta0},{theta1},{km_min},{km_max},{price_min},{price_max}")
    print(f"Fichier {filename} créé avec succès !")



# --- TEST DU CODE ---
if __name__ == "__main__":
    try:
        # 1. On charge les données
        print("Chargement des données...")
        km, price = load_data("data.csv")
        
        # 2. On normalise (C'est CRUCIAL car c'est là que km_min et km_max sont créés)
        print("Normalisation...")
        km_norm, km_min, km_max = normalize(km)
        price_norm, price_min, price_max = normalize(price)

        # 3. On entraîne le modèle (C'est là que les thétas sont calculés)
        print("Entraînement du modèle...")
        theta0_norm, theta1_norm = train_model(km_norm, price_norm)

        print(f"--- RÉSULTATS ---")
        print(f"Theta0: {theta0_norm}")
        print(f"Theta1: {theta1_norm}")

        # 4. On sauvegarde ENFIN (maintenant que les variables existent)
        save_model(theta0_norm, theta1_norm, km_min, km_max, price_min, price_max)

    except Exception as e:
        print(f"Une erreur inattendue est survenue : {e}")