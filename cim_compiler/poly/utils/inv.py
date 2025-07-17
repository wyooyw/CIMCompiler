import numpy as np

a = np.array(
    [
        [-1 / 2, 1],
        [1 / 2, 1],
    ]
)

# inv_a = inv_scale_to_integer(a)
# # pretty print inv_a
# for i in range(inv_a.rows):
#     row = inv_a[i,:]
#     row_str = [f'{str(elem):<3}' for elem in row]
#     print(f"{' '.join(row_str)}")
a_inv = np.linalg.inv(a)
print(a_inv)
