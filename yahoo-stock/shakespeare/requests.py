import random

# List of Shakespearean insults
snarky_comments = [
    "Thou art as fat as butter!",
    "Thou pribbling ill-nurtured scullion!",
    "Thou art a boil, a plague sore, an embossed carbuncle in my corrupted blood.",
    "Thou art a flesh-monger, a fool, and a coward.",
    "Thou art unfit for any place but hell!",
    "Thou art a lot of dirt with a speech.",
    "Thou art a very ragged knife!",
    "Thou art a sheep-biting dog!",
    "Thou art as welcome as a fart in a crowded room.",
    "Thou art like a candle in the wind. Just...fading away."
]

# Function to generate a snarky comment
def generate_snark():
    return random.choice(snarky_comments)

# AI agent that responds with a snarky comment
def shakespearean_agent():
    print("Shakespearean Snark AI: Ready to throw some shade in true Bard style!")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Shakespearean Snark AI: Farewell, you bawdy, hedge-born scullion!")
            break
        else:
            response = generate_snark()
            print(f"Shakespearean Snark AI: {response}")

# Run the agent
shakespearean_agent()

