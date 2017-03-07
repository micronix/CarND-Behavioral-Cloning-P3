import csv
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
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


def load_aug(path, filename='driving_log_aug.csv'):
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

def generateShadows(path, basefile, image, angle, entries, count=5):
    filename = path + '/IMG/' + basefile
    entries.append([filename, angle])
    for i in range(count):
        img = np.copy(image)
        h, w, c = image.shape
        [x1, x2] = np.random.choice(w, 2, replace=False)
        m = h / (x2 - x1)
        b = - m * x1
        for i in range(h):
            c = int((i - b) / m)
            image[i, :c, :] = (image[i, :c, :] * .5).astype(np.int32)
        filename = path + '/IMG/' + str(i) + basefile
        cv2.imwrite(filename, img)
        entries.append([filename, angle])

def generateAugmented(path, img, angle, entries, count=5):
    basefile = img.split('/')[-1]
    image = cv2.imread(path + '/IMG/' + basefile)
    generateShadows(path, basefile, image, angle, entries, count)
    image = image[:, ::-1, :]
    generateShadows(path, basefile, image, -angle, entries, count)


def augmentData(path, infile='driving_log_processed.csv', outfile='driving_log_aug.csv'):
    samples = load_samples(path, infile)
    with open(os.path.join(path, outfile), 'w') as csvfile:
        writer = csv.writer(csvfile)
        for line in samples:
            entries = []
            angle = float(line[3])
            generateAugmented(path, line[0], angle, entries)
            generateAugmented(path, line[1], angle + 0.2, entries)
            generateAugmented(path, line[2], angle - 0.2, entries)
            for entry in entries:
                writer.writerow(entry)




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
    data_path = '/home/rrodriguez/data2'
    #data_path = 'data'
    preprocess(data_path, 0.05)
    augmentData(data_path)
    visualize(data_path, 'driving_log_aug.csv')
    visualize(data_path)
