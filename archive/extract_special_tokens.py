import json

with open('vocab.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Filter out keys that start with "<reserved"
filtered_data = {key: value for key, value in data.items() if key.startswith("<reserved")}
# Sort by extracting the number from each "<reserved...>" key
sorted_reserved_data = dict(sorted(filtered_data.items(), key=lambda item: item[1]))

with open('reserved_tokens.json', 'w', encoding='utf-8') as file:
    json.dump(sorted_reserved_data, file, ensure_ascii=False, indent=4)

# Print or use `filtered_data` as needed
print(sorted_reserved_data)
print('Length:', len(sorted_reserved_data))
