import re

def clean_text(text: str) -> str:
    # Normalize dashes, quotes etc.
    text = text.replace("\u09f7", "।")  # Bengali danda fix (if needed)
    
    # Remove unwanted symbols
    text = re.sub(r'[^\u0980-\u09FF\s.,?!।\'\"-]', '', text)  # Keep Bangla chars and basic punctuation
    
    # Normalize spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


if __name__ == "__main__":
    with open("data/processed_text.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    cleaned = clean_text(raw_text)

    with open("data/cleaned_text.txt", "w", encoding="utf-8") as f:
        f.write(cleaned)

    print("✅ Cleaned text written to data/cleaned_text.txt")
