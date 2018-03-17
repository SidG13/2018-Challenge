import csv
items = []
with open('ClassCombinations.csv', 'rb') as file:
    for row in file:
        item = row.decode('utf-8').replace('\"', '').replace("\n", '')
        items += item.split(', ')
items = list(set(items))
with open('classes.txt', 'wb') as file:
    for item in items:
        file.write(item.encode('utf-8'))
        file.write('\n'.encode('utf-8'))
