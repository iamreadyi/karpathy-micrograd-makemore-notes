import ast
import torch


def rreplace(s, old, new, occurrence):
    li = s.rsplit(old, occurrence)
    return new.join(li)


words = open('turkish_names_raw.txt').read().rstrip()
# print(words)
# tek string olarak aliyorum. sonrasinda tirnaklari cope atip
words = words.split("\n")
# print(words)

# print(words.count('/'))
words_but_no_comma = []

for i in words:
    words_but_no_comma.append(rreplace(i, ',', '', 1))

words_but_not_string = []
for i in words_but_no_comma:
    # string to relevant variable. works for single
    words_but_not_string.append(ast.literal_eval(i))
# print(words_but_not_string)


names_but_no_gender = []
for i in words_but_not_string:
    names_but_no_gender.append(i[1].lower())
# print(names_but_no_gender)
single_string = ''
for i in names_but_no_gender:
    single_string += i
# print(single_string)
setim = set(single_string)
# print(len(setim))
# print(single_string.count('/'))

# Change last element with unlu
# cleaned from /
names_but_no_gender[len(names_but_no_gender)-1] = 'ünlü'
# print(single_string.count('\u0307'))
# print(single_string.count(' '))
# get rid of .lower() big i mess and space mess from data
# print(names_but_no_gender)
for index, i in enumerate(names_but_no_gender):
    names_but_no_gender[index] = i.replace('\u0307', 'i').replace(' ', '')
single_string = ''
for i in names_but_no_gender:
    single_string += i
setim = set(single_string)
# print(single_string.count('\u0307'))
# print(single_string.count(' '))
# print(setim)
# print(single_string[3900:3920])
words = names_but_no_gender
# print(names_but_no_gender[:10])
# print(len(setim))
N = torch.zeros((33, 33), dtype=torch.int32)

chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

P = (N+1).float()
P /= P.sum(1, keepdim=True)
ix = 0
for _ in range(500):
    while True:
        #         p = N[ix].float()  This is not efficient because we calculate it every time
        #         p = p / p.sum()    calculating once and storing the distributions is better
        p = P[ix]
        ix = torch.multinomial(
            p, num_samples=1, replacement=True).item()
        print(itos[ix], end="")
        if ix == 0:
            break
