#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

a = np.array([[1, 3], [3, 2], [3, 4]])
b = np.array([[1, 2], [1, 1]])

print(b -a)
print(np.sum(np.power(a - b, 2), axis=1))