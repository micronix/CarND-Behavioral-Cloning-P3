import csv
import matplotlib.pyplot as plt
import os
from sklearn.utils import shuffle

# load samples from file into a list
def load_samples(path, filename='driving_log.csv'):
    samples = []
    with open(os.path.join(path,filename)) as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for line in reader:
            samples.append(line)
    return samples

# remove a percentage of zeros and save into a new file
def preprocess(path, zeros_left, infile='driving_log.csv', outfile='driving_log_processed.csv'):
    samples = load_samples(path, infile)
    indexes = [i for i, x in enumerate(samples) if float(x[3]) == 0]
    indexes = shuffle(indexes)
    indexes = indexes[0:int((1 - zeros_left) * len(indexes))]
    for index in sorted(indexes, reverse=True):
        del samples[index]
    with open(os.path.join(path, outfile), 'w') as csvfile:
        writer = csv.writer(csvfile)
        for sample in samples:
            writer.writerow(sample)


def visualize(path, filename='driving_log.csv'):
    samples = load_samples(path, filename)
    values = [ float(sample[3]) for sample in samples]
    zeros = [ float(sample[3]) for sample in samples if float(sample[3]) == 0]
    non = [ float(sample[3]) for sample in samples if float(sample[3]) != 0]
    print("Zeros:", len(zeros))
    print("Non-Zero:", len(non))
    print("Total: ", len(values))

    plt.hist(values,50)
    plt.show()

if __name__ == '__main__':
    data_path = '/home/rrodriguez/track1'
    data_path = 'data'
    preprocess(data_path, 0.2)
    visualize(data_path, 'driving_log_processed.csv')
