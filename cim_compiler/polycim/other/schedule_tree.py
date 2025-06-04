import islpy as isl

"""
Try to use schedule tree to implement dwconv to macro mapping
"""

import islpy as isl

from polycim.utils.utils import get_root, pretty_print_schedule_tree

# domain =


domain = isl.Set("{ S[oh,ow,kh,kw]: 0<=oh<8 and 0<=ow<8 and 0<=kh<3 and 0<=kw<3}")
node = isl.ScheduleNode.from_domain(domain).child(0)

tile = isl.Map("{ S[h,w,r,s] -> [floor(h/2),h%2,floor(w/2),w%2,r,s]}")
skew = isl.Map("{ [ht,hp,wt,wp,r,s] -> [hp+r,wp+s,ht,wp,wt,hp] }")
merge = isl.Map("{ [s0, s1, s2, s3, s4, s5] -> [s2,s4,s3*2+s5,s0*4+s1] }")
hardware_tile = isl.Map(
    "{ [s0, s1, s2, s3] -> [(s0), (s1), (floor((s2)/4)), (floor((s3)/16)), ((s2) mod 4), ((s3) mod 16)] }"
)
schedule = tile.apply_range(skew).apply_range(merge).apply_range(hardware_tile)
print(f"{schedule=}\n")
new_domain = domain.apply(schedule)
print(f"{new_domain=}\n")
print(f"{new_domain.count_val()=}\n")
print(f"{new_domain.convex_hull()=}\n")

schedule2 = schedule.detect_equalities()
new_domain2 = domain.apply(schedule2)
print(f"{new_domain2=}\n")
print(f"{new_domain2.count_val()=}\n")

new_domain3 = new_domain2.convex_hull()
print(f"{new_domain3=}\n")
print(f"{new_domain3.count_val()=}\n")
exit()

# mupf = isl.MultiUnionPwAff("""[
# { S[oh,ow,kh,kw] -> [oh] },
# { S[oh,ow,kh,kw] -> [ow] },
# { S[oh,ow,kh,kw] -> [kh] },
# { S[oh,ow,kh,kw] -> [kw] }
# ]""")
mupf = schedule.as_multi_union_pw_aff()
node = node.insert_partial_schedule(mupf)
print(pretty_print_schedule_tree(get_root(node)))
# tile_sizes_multi_val = isl.MultiVal(" { [%s] }"%(",".join(["2", "2", "1", "1"])))
# node = node.band_tile(tile_sizes_multi_val)

# mupf = isl.MultiUnionPwAff("[{ S[i,j] -> [j]; T[i,j,k] -> [j]}]")
# node = node.insert_partial_schedule(mupf).child(0)

# usl = isl.UnionSetList.from_union_set(isl.UnionSet("{ S[i,j] }"))
# usl = usl.add(isl.UnionSet("{ T[i,j,k] }"))
# node = node.insert_sequence(usl)
# print(node.get_type())
# node = node.child(1).child(0)

# mupf = isl.MultiUnionPwAff("[{ T[i,j,k] -> [k]}]")
# node = node.insert_partial_schedule(mupf).child(0)


def gen_code_for_schedule_tree(root, context=None):
    root = get_root(root)
    if context is None:
        context = isl.Set("{ : }")
    build = isl.AstBuild.from_context(context)
    print(root.get_schedule())
    return build.node_from_schedule(root.get_schedule()).to_C_str()


print(gen_code_for_schedule_tree(node))
