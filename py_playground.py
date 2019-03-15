from config import RefineNetConfig
from models import models


if __name__ == '__main__':
    A = RefineNetConfig(create_dirs=False)
    B = models.get_model(A)
    count = 172
    for l in B.layers[count:]:
        print('%d: %s' % (count, l.name))
        count += 1
