import operator

import numpy as np
from math import inf

print(np.finfo(np.float32))


def m_eps(type_val=np.float32):
    eps = [type_val(1)]
    count = 0
    while eps[count] != type_val(0):
        eps.append(type_val(1 + 1 / (pow(2, count))) - type_val(1))
        count += 1
    return eps[count - 1], count - 2


print(f'For 32BIT:\n\tULP = {m_eps()[0]} \t\t| Mantissa = {m_eps()[1]}')
print(f'For 64BIT:\n\tULP = {m_eps(np.float64)[0]} | Mantissa = {m_eps(np.float64)[1]}')


def power(type_val=np.float32, trend=inf):
    eps = type_val(1)
    count = 0
    if trend == inf:
        factor = type_val(2)
    else:
        factor = type_val(1 / 2)
    while eps != type_val(trend):
        eps = type_val(eps * factor)
        count += 1
    return count - 1


print(f'\nPower for 32BIT:\n\tMAX = {power()} \t| MIN = {power(np.float32, 0)}')
print(f'Power for 64BIT:\n\tMAX = {power(np.float64)} \t| MIN = {power(np.float64, 0)}\n')


def compare(type_val=np.float32):
    value_dict = {'1': (type_val(1)), '1+eps/2': type_val(1) + m_eps(type_val)[0] / (type_val(2)),
                  '1+eps': type_val(1) + m_eps(type_val)[0],
                  '1+eps/2+eps': type_val(1) + m_eps(type_val)[0] + m_eps(type_val)[0] / (type_val(2))}
    # print(sorted(value_dict.items(), key=operator.itemgetter(1)))

    # value_list = [(type_val(1)), type_val(1) + m_eps(type_val)[0] / (type_val(2)), type_val(1) + m_eps(type_val)[0],
    #               type_val(1) + m_eps(type_val)[0] / (type_val(2)) + m_eps(type_val)[0]]
    for first_letter, first_num in value_dict.items():
        for second_letter, second_num in value_dict.items():
            if type_val(second_num) > type_val(first_num):
                print(f'({second_num}) is greater than ({first_num})')
            elif type_val(first_num) > type_val(second_num):
                print(f'({second_num}) is less than ({first_num})')
            else:
                print(f'({second_num}) is equal ({first_num})')


compare()
