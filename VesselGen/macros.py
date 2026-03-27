import os
from os.path import join

# [z-, z+ (in itksnap LRAPIL: I: S), y-, y+ (P: A), x-, x+ (L: R)]
# hypers to be set
# RAI orientation, zyx array layout
SPACING = (1.5, 1.5, 1.5)
BASE_PATH = '/data/dataset/Totalsegmentor_dataset'
MOD_PATH = '/data/dataset/vessel_augmentation'

Identity = lambda _: _
XYZ_AVG = lambda mm, T=round: T(mm / sum(SPACING) * 3)
X = lambda mm, T=round: T(mm / SPACING[0])
Y = lambda mm, T=round: T(mm / SPACING[1])
Z = lambda mm, T=round: T(mm / SPACING[2])

with open("process/totalseg.txt", 'r') as fp:
    seg_id_and_names = fp.readlines()
LABEL_MAPPING = {line.split('|')[1].strip(): int(line.split('|')[0].strip()) for line in seg_id_and_names}

ADJACENT_NODES_3D = [(deltax, deltay, deltaz) for deltax in (-1, 0, 1) for deltay in (-1, 0, 1) for deltaz in (-1, 0, 1)]
ADJACENT_NODES_3D.remove((0, 0, 0))  # remove stationary movement
ADJACENT_NODES_2D = [(deltax, deltay) for deltax in (-1, 0, 1) for deltay in (-1, 0, 1)]
ADJACENT_NODES_2D.remove((0, 0))  # remove stationary movement

SEG_PATH = join(MOD_PATH, 'seg')
CT_PATH = join(MOD_PATH, 'image')
MASK_PATH = join(MOD_PATH, 'mask')
TRASH_BIN = join(MOD_PATH, 'trash')
BLANK_PATH = join(MOD_PATH, 'blank')
CUT_CT_PATH = join(MOD_PATH, 'seginputs')