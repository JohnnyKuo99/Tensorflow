from tensorflow.keras import utils

orig1 = [4, 5, 6, 7, 8]
num_digit1 = 10

for orig in orig1:
    print("{}==>{}".format(orig, utils.to_categorical(orig, num_digit1)))

orig2 = 4
num_digits2 = [5, 10, 15, 20]

for num_digit in num_digits2:
    print("{}==>{}".format(orig2, utils.to_categorical(orig2, num_digit)))