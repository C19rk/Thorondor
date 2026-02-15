import os

# Change directory here depending on the location
LABELS_FOLDER = "data/train/labels/"

def main():
    new_char = input("Enter a new first character: ").strip()
    
    if not new_char:
        print("Invalid input!")
        return
    
    for filename in os.listdir(LABELS_FOLDER):
        if not filename.endswith(".txt"):
            continue

        file_path = os.path.join(LABELS_FOLDER, filename)

        with open(file_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len (parts) == 0:
                continue

            parts[0] = new_char
            new_lines.append(" ".join(parts) + "\n")

        with open(file_path, "w") as f:
            f.writelines(new_lines)

        print(f"Updated: {filename}")

    print("All files have been updated!")

if __name__ == "__main__":
    main()