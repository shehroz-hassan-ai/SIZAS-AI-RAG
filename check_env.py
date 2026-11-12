from dotenv import load_dotenv
import os

load_dotenv()
k = os.getenv("GROQ_API_KEY")

print("Mili:", bool(k))
if k:
    print("Shuru ke 4 characters:", k[:4])
    print("Lambai:", len(k))
    print("Space ya quote hai:", k != k.strip().strip('"').strip("'"))