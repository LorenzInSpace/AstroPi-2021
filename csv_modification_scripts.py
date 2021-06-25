import csv

"""
# Read csv from the ISS
coordinates=[]
with open('data01.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        if len(coordinates) <= int(row["Photo"]):
            coordinates.append([int(row["Photo"]), float(row["Longitude"]), float(row["Latitude"])])
        else:
            coordinates[int(row["Photo"])] = [int(row["Photo"]), float(row["Longitude"]), float(row["Latitude"])]

# Generate valid csv for earth engine
with open('data_clean.csv', mode='w') as csv_file:
    fieldnames = ['Photo', 'Longitude', 'Latitude']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for r in coordinates:
        writer.writerow({'Photo': f'{r[0]}', 'Longitude': f'{r[1]}', 'Latitude': f'{r[2]}'})
"""
# Fully or almost fully black images captured on 21/4/21
bad = [114, 117, 119, 120, 126, 127, 128, 129, 133, 134, 
       137, 138, 140, 143, 145, 147, 157, 161, 162, 163, 
       165, 166, 168, 170, 172, 172, 177, 179, 182, 186, 
       190, 193, 195, 198, 200, 199, 201, 158, 155, 130, 
       133, 213, 215, 217, 221, 222, 226, 230, 231, 234, 
       235, 240, 243, 248, 260, 265, 266, 271, 275, 276, 
       277, 278, 280, 281, 283, 287, 291, 292, 294, 295,
       296, 301, 304, 308, 309, 310, 314, 317, 317, 318,
       322, 330, 337, 340, 345, 346, 350, 348, 351, 354, 
       355, 361, 362, 363, 364, 365, 366, 798, 799, 800,
       801, 804, 805, 808, 809, 810, 811, 812, 813, 817,
       819, 823, 824, 825, 826, 828, 830, 831, 835, 837, 
       839, 841, 842, 843, 845, 846, 847, 851, 854, 855,
       856, 857, 858, 859, 861, 862, 863, 864, 866, 867,
       868, 869, 870, 873, 874, 880, 883, 884, 886, 887, 
       888, 889, 890, 891, 892, 893, 895, 897, 899, 905,
       907, 908, 910, 912, 916, 921, 922, 923, 924, 925,
       928, 930, 933, 937, 939, 944, 945, 947, 951, 952,
       953, 958, 961, 963, 964, 967, 972, 974, 977, 986,
       987, 989, 990, 991, 992, 993]
# Fully or almost fully black images captured on 19/4/15
bad = [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009,
       1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019,
       1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029,
       1030, 1031, 1032, 1033, 1034, 1035, 1311, 1312, 1313, 1314,
       1315, 1316, 1317, 1318, 1319, 1320, 1321, 1322, 1323, 1324,
       1325, 1326, 1327, 1328, 1329, 1330, 1331, 1332, 1333, 1334,
       1335, 1336, 1337, 1338, 1339]

# 1066 is the first valid image of the second batch we used
pioggia = []
valid = 0
# Read output from earth engine
with open('pioggia_test03.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for i, row in enumerate(csv_reader):
        if row["mm_di_pioggia"] != "" and i+1066 not in bad:
            pioggia.append([i+1066, float(row["mm_di_pioggia"])])
            valid += 1
print(valid)

# Generate csv valid as a label for our neural network database
with open('data_clean.csv', mode='w') as csv_file:
    fieldnames = ['Photo', 'mm_di_pioggia']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for r in pioggia:
        writer.writerow({'Photo': f'image_{r[0]:03d}.jpg', 'mm_di_pioggia': f'{r[1]}'})
