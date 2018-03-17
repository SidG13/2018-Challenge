import csv
items = []
with open('ClassCombinations.csv', 'r') as file:
    for row in file:
        item = row.replace('\"', '').replace("\n", '')
        items += item.split(', ')
for item in items:
    item.strip()
items = list(set(items))
with open('classes.txt', 'w') as file:
    for item in items:
        file.write(item)
        file.write("\n")
