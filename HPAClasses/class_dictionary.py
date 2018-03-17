import csv
import json
items = {}

with open('classes.txt', 'r') as file:
    for line in file:
        items[line.replace('\n', '')] = []

with open("ClassData.csv", 'r') as file:
    reader = csv.reader(file, delimiter=',')
    for row in reader:
        for category in items.keys():
            if category in row[1].replace('\"', '').split(', '):
                items[category].append(row[0])

with open('ClassAssociations.json', 'w') as file:
    json.dump(items, file)

# to load the dictionarys
# new_items = {}
# with open('ClassAssociations.json', 'rb') as file:
#     new_items = json.load(file)
#
# print(new_items == items)
