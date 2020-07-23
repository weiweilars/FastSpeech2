from unidecode import unidecode
from num_utils import normalize_numbers
import re

_valid_symbols = [
  'AA', 'AA0', 'AA1', 'AA2', 'AE', 'AE0', 'AE1', 'AE2', 'AH', 'AH0', 'AH1', 'AH2',
  'AO', 'AO0', 'AO1', 'AO2', 'AW', 'AW0', 'AW1', 'AW2', 'AY', 'AY0', 'AY1', 'AY2',
  'B', 'CH', 'D', 'DH', 'EH', 'EH0', 'EH1', 'EH2', 'ER', 'ER0', 'ER1', 'ER2', 'EY',
  'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH', 'IH0', 'IH1', 'IH2', 'IY', 'IY0', 'IY1',
  'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OW0', 'OW1', 'OW2', 'OY', 'OY0',
  'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UH0', 'UH1', 'UH2', 'UW',
  'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH'
]


""" from https://github.com/keithito/tacotron """


# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
    ('mrs', 'misess'),
    ('mr', 'mister'),
    ('dr', 'doctor'),
    ('st', 'saint'),
    ('co', 'company'),
    ('jr', 'junior'),
    ('maj', 'major'),
    ('gen', 'general'),
    ('drs', 'doctors'),
    ('rev', 'reverend'),
    ('lt', 'lieutenant'),
    ('hon', 'honorable'),
    ('sgt', 'sergeant'),
    ('capt', 'captain'),
    ('esq', 'esquire'),
    ('ltd', 'limited'),
    ('col', 'colonel'),
    ('ft', 'fort'),
]]

_pad = '_'
_sos = '^'
_eos = '~'
_punctuation = "!',.? "
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ['@' + s for s in _valid_symbols]

symbols = [_pad, _sos, _eos] + list(_punctuation) + list(_letters) + _arpabet

def remove_unnecessary_symbols(text):
    # added
    text = re.sub(r'[\(\)\[\]\<\>\"]+', '', text)
    return text

def expand_symbols(text):
    # added
    text = re.sub("\;", ",", text)
    text = re.sub("\:", ",", text)
    text = re.sub("\-", " ", text)
    text = re.sub("\&", "and", text)
    return text

def clean_text(text, uppercase=True):
    
    # change to correct format
    x = unidecode(text).lower()

    # change number to words
    x = normalize_numbers(x)

    # expand the addbreviations 
    for regex, replacement in _abbreviations:
        x = re.sub(regex, replacement, x)

    x = remove_unnecessary_symbols(text)
    x = expand_symbols(x)  

    if uppercase:
        x = x.upper()

    # remove extra white space 
    x = re.sub(re.compile(r'\s+'), ' ', x)

    # exception (borrow from Deepest-project/Transformer-TTs)
    if x[0]=="'" and x[-1]=="'":
        x = x[1:-1]
    
    return x
