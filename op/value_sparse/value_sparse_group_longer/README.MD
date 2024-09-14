第三个结果（第4、5列之和）

错误：-214

正确：298

模拟器权重：

```
(Pdb) p weight_data[:,4:6]
array([[-128,    4],
       [  32,   -1],
       [-128,   -2],
       [   4,    1]], dtype=int8)
```

test权重：

```
array([-124,   31,  126,    5], dtype=int8)
```













## value sparse reduce切分

#### mapping_reduce_to_macro

整个规约轴如何切分为多次Macro计算

mapping_reduce_to_macro[i]: 第i组filter的规约轴，拆分为几次macro计算

#### macro_to_from

整个macro计算如何拆分为多次128

macro_to_from[i]: 第i组macro计算，包含多少次128计算

注意：每次macro计算，可能包含不同次数的128计算。

#### from_to_row

128拆分为几次16

from_to_to[i]：第i组128计算，包含几次16计算

注意：由于稀疏度不同，每次128都可能包含不同次数的16

#### macro_to_row

整个macro计算包含多少次16计算

用于加载weight，因为每次weight是一次性加载，不能每次进行16计算时临时加载。

macro_to_to[i]:第i次macro计算，需要加载的权重为macro_to_to[i] * n_group * n_vcol_per_group

#### 