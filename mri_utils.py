import argparse

# TODO: discuss if we should assume that the user has properly preprocessed the .seg file (i.e. summed the masks beforehand)
# Mask Channels in .seg files
# Indices where masks occur in .seg files
# Currently assuming that the user has preprocessed the .seg file
MASK_FEMORAL_CARTILAGE = 0  # [0]
MASK_TIBIAL_CARTILAGE = [1, 2]  # [1,2]
MASK_PATELLAR_CARTILAGE = 3  # [3]
MASK_MENISCUS = [4, 5]  # [4,5]

fc = FC = MASK_FEMORAL_CARTILAGE
tc = TC = MASK_TIBIAL_CARTILAGE
pc = PC = MASK_PATELLAR_CARTILAGE
men = MEN = MASK_MENISCUS

SUPPORTED_TISSUES = ['fc', 'tc', 'pc', 'men']


def init_cmd_line(parser: argparse.ArgumentParser):
    parser.add_argument('--tissues', nargs='+',
                        default=['fc'],
                        help='tissues to segment. Use `fc` ,`tc`, `pc`, or `men`')


def parse_tissues(vargin: dict):
    str_tissues = vargin['tissues']
    tissues = []
    for t in str_tissues:
        if t not in SUPPORTED_TISSUES:
            raise ValueError('tissue corresponding to `%s` not found. Supported tissues: %s' % (t, SUPPORTED_TISSUES))

    if 'fc' in str_tissues:
        tissues.append(MASK_FEMORAL_CARTILAGE)
    if 'tc' in str_tissues:
        tissues.append(MASK_TIBIAL_CARTILAGE)
    if 'pc' in str_tissues:
        tissues.append(MASK_PATELLAR_CARTILAGE)
    if 'men' in str_tissues:
        tissues.append(MASK_MENISCUS)

    return tissues
