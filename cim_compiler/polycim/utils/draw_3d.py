import islpy as isl
from islplot.plotter3d import plot_set_3d

bset_data = isl.Set("{ S[i,j,k]: 0<=i<8 and 0<=j<8 and 0<=k<8 }")
s = plot_set_3d(
    bset_data, show_points=True, show_shape=False, full_page=True, scale=0.3
)
print(s)
