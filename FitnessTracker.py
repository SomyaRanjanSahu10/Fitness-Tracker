import numpy as np
import pandas as pd
from sklearn.cluster import KMeans  # For personalized recommendations
from sklearn.preprocessing import StandardScaler  # For data normalization
import matplotlib.pyplot as plt  # For data visualization

# Function to calculate Basal Metabolic Rate (BMR)
def calculate_bmr(weight, height, age, gender):
    """
    Calculate BMR using the Harris-Benedict equation.
    Gender: 1 for male, 0 for female.
    """
    if gender == 1:
        return 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
    else:
        return 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)

# Function to calculate total daily calorie expenditure
def calculate_calorie_expenditure(bmr, activity_level):
    """
    Activity level multipliers:
    - Sedentary: 1.2
    - Lightly active: 1.375
    - Moderately active: 1.55
    - Very active: 1.725
    """
    return bmr * activity_level

# Function to provide personalized recommendations using K-Means clustering
def provide_recommendations(user_data, historical_data):
    """
    Use K-Means clustering to group users and provide recommendations.
    """
    # Normalize the data for better clustering
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(historical_data)
    
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(scaled_data)
    
    # Prepare user data for prediction
    user_features = np.array([list(user_data.values())])
    scaled_user_features = scaler.transform(user_features)
    
    # Assign the user to a cluster
    user_cluster = kmeans.predict(scaled_user_features)[0]
    
    # Provide recommendations based on the cluster
    if user_cluster == 0:
        return "Focus on increasing daily steps and adding strength training."
    elif user_cluster == 1:
        return "Maintain your current activity level and add 30 minutes of cardio."
    else:
        return "Improve sleep quality and reduce stress for better results."

# Function to visualize user activity trends
def visualize_activity_trends(user_data, historical_data):
    """
    Visualize user activity trends using matplotlib.
    """
    # Create a DataFrame for historical data
    df = pd.DataFrame(historical_data, columns=["Steps", "Calories", "Heart Rate", "Sleep", "Weight", "Height", "Age", "Gender"])
    
    # Add the user's data to the DataFrame
    user_df = pd.DataFrame([user_data.values()], columns=df.columns)
    df = pd.concat([df, user_df], ignore_index=True)
    
    # Plot the data
    plt.figure(figsize=(10, 6))
    
    # Plot steps
    plt.subplot(3, 1, 1)
    plt.plot(df["Steps"], marker="o", color="blue")
    plt.title("Daily Steps Trend")
    plt.xlabel("Days")
    plt.ylabel("Steps")
    
    # Plot calories burned
    plt.subplot(3, 1, 2)
    plt.plot(df["Calories"], marker="o", color="green")
    plt.title("Daily Calories Burned Trend")
    plt.xlabel("Days")
    plt.ylabel("Calories")
    
    # Plot heart rate
    plt.subplot(3, 1, 3)
    plt.plot(df["Heart Rate"], marker="o", color="red")
    plt.title("Daily Heart Rate Trend")
    plt.xlabel("Days")
    plt.ylabel("Heart Rate (bpm)")
    
    plt.tight_layout()
    plt.show()

# Function to get user input
def get_user_input():
    """
    Get user input for fitness tracking.
    """
    print("\n--- Personal Fitness Tracker ---")
    print("Please enter your fitness details:")
    
    weight = float(input("Weight (kg): "))
    height = float(input("Height (cm): "))
    age = int(input("Age: "))
    gender = input("Gender (male/female): ").lower()
    steps = int(input("Daily steps: "))
    calories_burned = float(input("Calories burned: "))
    heart_rate = int(input("Resting heart rate (bpm): "))
    sleep_hours = float(input("Sleep hours: "))
    
    # Convert gender to numeric (male = 1, female = 0)
    gender = 1 if gender == "male" else 0
    
    return {
        "steps": steps,
        "calories_burned": calories_burned,
        "heart_rate": heart_rate,
        "sleep_hours": sleep_hours,
        "weight": weight,
        "height": height,
        "age": age,
        "gender": gender
    }

# Main function to run the fitness tracker
def fitness_tracker():
    """
    Main function to run the Personal Fitness Tracker.
    """
    try:
        # Get user input
        user_data = get_user_input()
        
        # Historical data for clustering (simulated dataset)
        historical_data = np.array([
            [6000, 350, 85, 6, 72, 180, 30, 1],  # Male = 1, Female = 0
            [4000, 250, 75, 8, 65, 160, 28, 0],
            [7000, 400, 90, 7, 80, 175, 35, 1],
            [3000, 200, 70, 5, 60, 150, 22, 0],
            [8000, 450, 95, 8, 85, 190, 40, 1]
        ])
        
        # Calculate BMR and calorie expenditure
        bmr = calculate_bmr(user_data["weight"], user_data["height"], user_data["age"], user_data["gender"])
        calorie_expenditure = calculate_calorie_expenditure(bmr, 1.55)  # Assuming moderately active
        
        # Display user stats
        print("\n--- Your Fitness Summary ---")
        print(f"Steps: {user_data['steps']}")
        print(f"Calories Burned: {user_data['calories_burned']}")
        print(f"Heart Rate: {user_data['heart_rate']} bpm")
        print(f"Sleep: {user_data['sleep_hours']} hours")
        print(f"Daily Calorie Needs: {calorie_expenditure:.2f} kcal")

        # Provide personalized recommendations
        recommendation = provide_recommendations(user_data, historical_data)
        print(f"\nRecommendation: {recommendation}")

        # Visualize activity trends
        visualize_activity_trends(user_data, historical_data)
    
    except Exception as e:
        print(f"An error occurred: {e}")

# Run the fitness tracker
if __name__ == "__main__":
    fitness_tracker()