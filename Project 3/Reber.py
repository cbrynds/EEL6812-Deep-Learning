import random
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torch
import itertools

class ReberDataset(Dataset):

    def __init__(self, repeat, size,split="train"):
      
      g = EmbeddedRebarGrammar()

      if split=="train":
        random.seed(42)
      else:
       	random.seed(4242101)
      
      self.data = g.generate(size//2, repeat=repeat) + g.generatePeturbed(size//4, repeat=repeat) + g.generatePeturbedEnd(size//4, repeat=repeat)
      self.labels = [1]*(size//2) + [0]*(size//2)

      self.data = [g.stringToIndex(x) for x in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        x = torch.LongTensor( self.data[idx] )+1
        y = self.labels[idx]
        return x,y 
    
    def avgLen(self):
      lens = [len(x) for x in self.data]
      return sum(lens)/len(lens)


class ReberGrammar():

  def __init__(self):
    self.graph = {0: {'B':1}, 1:{'T':2,'P':3}, 2:{'S':2, 'X':4}, 3:{'T':3, 'V':5}, 4:{'X':3,'S':6}, 5:{'P':4, 'V':6}, 6:{'E':7, 'B':1} }    
    self.chars = {'B':0, 'T':1, 'S':2, 'X':3, 'P':4, 'V':5, 'E':6}
    self.idx = {0:'B', 1:'T', 2:'S', 3:'X', 4:'P', 5:'V', 6:'E'}

  def generate(self, count=1):
    words = []
    for i in range(count):
      s = 0
      w = []
      while s != 7:
        c,ns = random.choice(list(self.graph[s].items()))
        w.append(c)
        s = ns
      words.append(w)
    return words

  def generatePeturbed(self, count=1):
    words = []
    for i in range(count):
      w = self.generate()[0]
      while True:
        for j in range(random.randint(1,5)):
          w[random.randint(0,len(w)-1)] = random.choice(list(self.chars.keys()))
        if not self.isValid(w):
          break
      words.append(w)
    return words
    
  def isValid(self, string):
    s = 0
    i = 0
    while s != 7:
      if i == len(string):
        return False
      c = string[i]
      if c not in self.graph[s].keys():
        return False
      s = self.graph[s][c]
      i+=1
    return True

  def stringToIndex(self, string):
    return [self.chars[s] for s in string]
  
  def indexToString(self, idxlist):
    return [self.idx[i] for i in idxlist]

class EmbeddedRebarGrammar(ReberGrammar):

  def generate(self, count=1, repeat=1):
    words = []
    for i in range(count):
      path = random.choice(['T','P'])
      w = ['B', path] + list(itertools.chain(*super().generate(count=repeat))) + [path, 'E']
      words.append(w)
    return words
    
  def isValid(self, string):
    if string[0] != 'B' or string[-1] != "E":
      return False
    if string[1] != string[-2] or string[1] not in ['T', 'P']:
      return False
    return super().isValid(string[2:-2])

  def generatePeturbed(self, count=1, repeat=1):
    words = []
    for i in range(count):
      w = self.generate(repeat=repeat)[0]
      while True:
        for j in range(random.randint(1,5)):
          w[random.randint(0,len(w)-1)] = random.choice(list(self.chars.keys()))
        if not self.isValid(w):
          break
      words.append(w)
    return words

  def generatePeturbedEnd(self, count=1, repeat=1):
    words = []
    for i in range(count):
      w = self.generate(repeat)[0]
      while True:
        w[-2] = random.choice(['T','P'])
        if not self.isValid(w):
          break
      words.append(w)
    return words