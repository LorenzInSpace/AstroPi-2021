import csv

rain = []
# label_data.csv is the bataset csv file
with open('label_data.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for i, row in enumerate(csv_reader):
        current = float(row["mm_di_pioggia"])
        rain.append(current)
avg = sum(rain)/len(rain)
print(sum([(avg-p)**2 for p in rain])/len(rain))
