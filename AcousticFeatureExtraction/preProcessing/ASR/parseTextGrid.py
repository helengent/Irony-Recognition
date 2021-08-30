#!/usr/bin/env python3

from glob import glob


def lineUp(words, phones):

    combined = dict()

    for w in words:

        wordPhones = [p for p in phones if p[1] >= w[1] and p[2] <= w[2]]
        combined[w] = wordPhones

    return combined


def parse(f):
    with open(f, "r") as t:
        raw = t.read()
    lines = raw.split("\n")
    lines = [line.strip() for line in lines]

    fileStart = float(lines[3])
    fileEnd = float(lines[4])

    phones = list()
    words = list()

    assert lines[8] == '"phone"'

    #lines[9] should be fileStart
    #lines[10] should be fileEnd
    #lines[11], I believe, is the number of phones in the utterance

    i = 12

    while i < len(lines) and lines[i] != '"IntervalTier"':
        phoneStart = float(lines[i])
        phoneEnd = float(lines[i+1])
        phone = lines[i+2][1:-1]

        phones.append((phone, phoneStart, phoneEnd, phoneEnd - phoneStart))

        i += 3

    i += 1

    assert lines[i] == '"word"'
    i += 4

    while i < len(lines) - 1:
        wordStart = float(lines[i])
        wordEnd = float(lines[i+1])
        word = lines[i+2][1:-1]

        words.append((word, wordStart, wordEnd, wordEnd - wordStart))

        i += 3

    return fileStart, fileEnd, words, phones


def main(f):

    start, end, words, phones = parse(f)
    combined = lineUp(words, phones)
    return combined, words, phones


if __name__=="__main__":

    input_dir = "data/ANH_manual"
    fileList = glob("{}/*.TextGrid".format(input_dir))

    for f in fileList:
        main(f)