import csv
import spacy
import pandas

path = '../resources/'
path_csv = '/home/florian/PycharmProjects/twintransformationnlp2/resources/train/tt/TT_Train_Combined.csv'
path_folder = '../resources/train/tt/'
list_sentence = []
list_word = []

def main():
    nlp = spacy.load("en_core_web_sm")

    tokenize_word(nlp)
    #print(str(list_word))

    tokenize_sentence(nlp)
    #print(str(list_sentence))

    #Mode for creation of txt files can be "sentence" or "full"
    mode = "sentence"
    createtxtfiles(mode, nlp)

def read():
    f = open(path_csv, newline='')
    test_csv = csv.DictReader(f, delimiter=';') #wichtig: gebe hiermit Trennzeichen in csv an
    return test_csv

def write(input, name):
    f = open(path + name + ".txt", "w")
    f.write(str(input))
    f.close()

def tokenize_word(nlp):
    test_csv = read()

    for item in test_csv:
        doc = nlp(item['content'])

        for sentence in doc.sents:
            for word in sentence:
                list_word.append(str(word))

    write(list_word, "word")
    return

def tokenize_sentence(nlp):
    test_csv = read()

    for item in test_csv:
        doc = nlp(item['content'])

        for sentence in doc.sents:
            list_sentence.append(sentence)

    write(list_sentence, "sentence")
    return

def createtxtfiles(mode, nlp):
    data = read()
    count = 0

    #erstelle Datei für jeden Satz
    if mode == "sentence":
        for item in data:
            doc = nlp(item['content'])

            for sentence in doc.sents:
                f = open(path_folder + str(count) + ".txt", "w")
                f.write(str(sentence))
                f.close()
                count = count + 1

    # erstelle Datei für jedes Item
    elif mode == "full":
        for item in data:
            f = open(path_folder + str(count) + ".txt", "w")
            f.write(str(item))
            f.close()
            count = count + 1

    return

if __name__ == "__main__":
    main()






#Steinbruch
def pandasread():
    data = pandas.read_csv(path_csv)
    #temp.to_csv('temp_neu') #ein dict als csv speichern
    # item.update({'processed': temp})

    return data



