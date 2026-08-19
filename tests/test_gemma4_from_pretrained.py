from taktiny.maestro import Maestro

def test():
    try:
        model = Maestro.from_pretrained("google/gemma-4-26B-A4B-it", allow_unmatched=True)
        print("Success!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test()
