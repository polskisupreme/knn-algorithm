import csv
import math


def load_data(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if not row:
                continue
            row_data = [float(x) for x in row[:-1]]
            row_data.append(row[-1])
            data.append(row_data)
    return data

#wyswietlenie listy pobranej z pliku
# lista = load_data('iris.data')
# print(lista)

def euclidean_distance(x1, x2):
    return math.sqrt(sum([(x1[i] - x2[i]) ** 2 for i in range(len(x1) - 1)]))


def get_neighbors(train_set, test_instance, k):
    distances = []
    for train_instance in train_set:
        dist = euclidean_distance(train_instance, test_instance)
        distances.append((train_instance, dist))
    distances.sort(key=lambda x: x[1])
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors


def predict_classification(train_set, test_instance, k):
    neighbors = get_neighbors(train_set, test_instance, k)
    labels = [neighbor[-1] for neighbor in neighbors]
    prediction = max(set(labels), key=labels.count)
    return prediction


def test_accuracy(train_set, test_set, k):
    correct = 0
    for test_instance in test_set:
        prediction = predict_classification(train_set, test_instance, k)
        if prediction == test_instance[-1]:
            correct += 1
    accuracy = correct / float(len(test_set)) * 100.0
    return accuracy


def classify_single_instance(train_set, test_instance, k):
    prediction = predict_classification(train_set, test_instance, k)
    return prediction


# przykład użycia
train_set = load_data('iris.data')
test_set = load_data('iris.test.data')
k = 3

accuracy = test_accuracy(train_set, test_set, k)
print('Dokładność: %.2f%%' % accuracy)

# while True:
#     instance_str = input("Podaj wektor testowy (oddzielony przecinkami): ")
#     if not instance_str:
#         break
#     instance = [float(x) for x in instance_str.split(',')]
#     prediction = classify_single_instance(train_set, instance, k)
#     print('Klasyfikacja:', prediction)


import matplotlib.pyplot as plt

# przykład użycia
train_set = load_data('iris.data')
test_set = load_data('iris.test.data')
k_values = range(1, 104)
accuracies = []
for k in k_values:
    accuracy = test_accuracy(train_set, test_set, k)
    accuracies.append(accuracy)
    print('Dokładność dla k=', k, 'wynosi: ', accuracy)

plt.plot(k_values, accuracies)
plt.title('Zależność dokładności od wartości k')
plt.xlabel('Wartość k')
plt.ylabel('Dokładność (%)')
plt.show()
