import requests

# URL for Shakespeare's complete works on Project Gutenberg
url = "https://www.gutenberg.org/files/100/100-0.txt"

# Send GET request to fetch the content
response = requests.get(url)

# Check if the request is successful
if response.status_code == 200:
    # Get the full text of the works
    text = response.text
    # Search for insults or quotes using a keyword search (simple)
    insults = [line for line in text.split('\n') if 'thou' in line and 'art' in line]
    for insult in insults[:10]:  # Print the first 10 insults
        print(insult)
else:
    print("Failed to download the works.")

