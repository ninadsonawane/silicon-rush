import json

a = open("abc.txt").read()

labels = [
    'Urea',
    'Creatinine',
    'Uric Acid',
    'Calcium, Total',
    'Phosphorous',
    'Alkaline Phosphatase (ALP)',
    'Total Protein',
    'Albumin',
    'A : G Ratio',
    'Sodium',
    'Potassium',
    'Chloride'
]

lines = a.split('\n')

def is_float(n):
    try:
        float(n)
    except ValueError:
        return False
    return True

def is_special(line):
    if '-' in line:
        return False
    words = line.split()
    for word in words:
        if is_float(word):
            return True
    return False

special_lines = []

for line in lines:
    if is_special(line):
        special_lines.append(line)
    else:
        special_lines = []
    if len(special_lines) == len(labels):
        break

values = []
for line in special_lines:
    for word in line.split():
        if is_float(word):
            values.append(float(word))
            break

results = {}
for i in range(len(labels)):
    results[labels[i]] = values[i]

json_data = json.dumps(results, indent=4)
print(json_data)



# def is_float(n):
#     try:
#         float(n)
#     except ValueError:
#         return False
#     return True


# lines = a.split('\n')
# for line in lines:
#     line.strip()
#     if line.split() and is_float(line.split()[-1]) and not is_float(line.split()[0]):
#         val = line.split()[-1]
#         field = line[:-len(val)].strip()
#         val = float(val)

#         print(field, val)

