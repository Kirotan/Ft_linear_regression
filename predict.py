def load_model(filename="thetas.csv"):
    try:
        with open(filename, 'r') as file:
            content = file.read()
            values = content.split(',')
            
            theta0 = float(values[0])
            theta1 = float(values[1])
            km_min = float(values[2])
            km_max = float(values[3])
            price_min = float(values[4])
            price_max = float(values[5])
            
            return theta0, theta1, km_min, km_max, price_min, price_max

    except (FileNotFoundError, ValueError):
        print("Warning : Model not found. Default value utilisation.")
        return 0.0, 0.0, 0.0, 1.0, 0.0, 1.0 # Default value for no crash if no value but wrong prediction
    

if __name__ == "__main__":

    theta0, theta1, km_min, km_max, price_min, price_max = load_model()

    if theta0 == 0.0 and theta1 == 0.0:
        print("Note : Model not train. Prediction will be 0.")

    # 2. Demande à l'utilisateur
    user_input = input("Enter a mileage : ")

    try:
        mileage = float(user_input)
        
        if mileage < 0:
            print("Error : Mileage can't be negative.")
            exit()

        # --- STEP 1 : NORMALISATION ---
        # Formule : (value - min) / (max - min)
        mileage_norm = (mileage - km_min) / (km_max - km_min)

        # --- STEP 2 : PREDICTION ---
        price_norm = theta0 + (theta1 * mileage_norm)

        # --- STEP 3 : DENORMALISATION ---
        # Reverse formule : value_norm * (max - min) + min
        estimated_price = price_norm * (price_max - price_min) + price_min

        print(f"For {mileage} km, estimate price is : {estimated_price:.2f} euros")

    except ValueError:
        print("Error : It's not a valid number.")