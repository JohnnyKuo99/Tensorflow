import random

FILENAME = "data/bmi.csv"


def calculateBmi(height, weight):
    bmi = weight / ((height / 100) ** 2)
    if bmi < 18.5:
        return 'thin'
    if bmi < 25:
        return 'normal'
    return 'fat'


with open(FILENAME, 'w', encoding='UTF-8') as file1:
    file1.write('height,weight,label\n')
    category = {'thin': 0, 'normal': 0, 'fat': 0}
    for i in range(30000):
        height = random.randint(140, 200)
        weight = random.randint(36, 90)
        label = calculateBmi(height, weight)
        category[label] += 1
        file1.write("%d,%d,%s\n" % (height, weight, label))

print("generate OK")
print(category)