import numpy as np
from fractions import Fraction


# 此函數被用來做為Fraction的print設定
def fraction_to_str(x):
    if isinstance(x, Fraction):
        if x.denominator == 1:
            return str(x.numerator)
        else:
            return f"{x.numerator}/{x.denominator}"
    else:
        return str(x)


# 將numpy中的浮點數轉換為Fraction分數
def float_to_fraction(x):
    return Fraction(x).limit_denominator()


# 將float_to_fraction(x)函數項量化，變成可以用來對矩陣的每個元素做同樣操作的函數
vectorized_float_to_fraction = np.vectorize(float_to_fraction)


# 高斯消去法本體
def gaussian_elimination(matrix_a, matrix_b):
    # 儲存原本numpy的print設定
    original_options = np.get_printoptions()
    # 改變numpy的print選項
    np.set_printoptions(formatter={'all': lambda x: fraction_to_str(x)}, precision=3, suppress=True)

    if matrix_b.shape[1] > 1 or matrix_b.shape[0] != matrix_a.shape[0]:
        print('常數項raw不對')
        return

    matrix_c = np.hstack([matrix_a, matrix_b])
    print('增廣矩陣')
    print(matrix_c)

    matrix_c = matrix_c.astype(float)
    matrix_c = vectorized_float_to_fraction(matrix_c)

    times = min(matrix_c.shape[0], matrix_c.shape[1])

    for i in range(0, times):
        if matrix_c[i, i] == Fraction(0, 1):
            #print('除零錯誤')
            for k in range(i+1, times):
                if matrix_c[k, i] != Fraction(0, 1):
                    temp = matrix_c[i]
                    matrix_c[i] = matrix_c[k]
                    matrix_c[k] = temp
                    break

        for j in range(0, times):
            if j != i and matrix_c[j, i] != Fraction(0, 1) and matrix_c[i, i] != Fraction(0, 1):
                factor = matrix_c[j, i] / matrix_c[i, i]
                matrix_c[j] = matrix_c[j] - (matrix_c[i] * factor)

        if matrix_c[i, i] != Fraction(0, 1):
            factor = 1 / matrix_c[i, i]
            matrix_c[i] = matrix_c[i] * factor

    print('解')
    print(matrix_c)

    # 將numpy的print設定改回來，避免被當成library引用時影響其他模組的運作
    np.set_printoptions(**original_options)


# 以下為測試題目
print('----------------------- 測試 -----------------------')
A = np.array([
    [2, 3, -1],
    [1, -2, 3],
    [4, 1, -2]
    ])

B = np.array([
    [5],
    [6],
    [0]
])

print('\n第一題 投影片的作業')
gaussian_elimination(A, B)

A = np.array([
    [1, 2, 3],
    [2, 1, 3],
    [1, -1, 0],
    [1, -2, -1]
    ])

B = np.array([
    [0],
    [6],
    [6],
    [8]
])

print('\n第二題 無限多組解 https://www.youtube.com/watch?v=mujVfld2bf8')
gaussian_elimination(A, B)

A = np.array([
    [3, 5, -7],
    [8, -3, 4],
    [1, 2, 3]
    ])

B = np.array([
    [1],
    [9],
    [6]
])

print('\n第三題 一組解 https://www.youtube.com/watch?v=m__BD3t1oOM')
gaussian_elimination(A, B)

A = np.array([
    [1, 1, 1],
    [1, 2, 3],
    [2, 3, 4]
    ])

B = np.array([
    [3],
    [7],
    [11]
])

print('\n第四題 無解 https://www.junyiacademy.org/teacherpreneur/tfgcoocs/tfgcoocs-math-11/v/ub-QYhFsLjo')
gaussian_elimination(A, B)


A = np.array([
    [3, -2, 2],
    [7, -3, 2],
    [2, -1, 4]
    ])

B = np.array([
    [16],
    [26],
    [18]
])

print('\n第五題 https://www.youtube.com/watch?v=79KiCxMwspw')
gaussian_elimination(A, B)


A = np.array([
    [3, 2, 1],
    [2, 3, 1],
    [1, 2, 3]
    ])

B = np.array([
    [39],
    [34],
    [26]
])

print('\n第六題 https://www.youtube.com/watch?v=E3smq4ujIxs')
gaussian_elimination(A, B)


A = np.array([
    [1, -3, -2],
    [2, 1, 3],
    [3, -2, 5]
    ])

B = np.array([
    [-4],
    [6],
    [6]
])

print('\n第七題 https://www.youtube.com/watch?v=59KG97rbL18')
gaussian_elimination(A, B)
