import islpy as isl

from polycim.op.base_operator import BasicOperator


def get_op_dwconv2d(ic, oh, ow, kh, kw, stride, dilation, virtual_axis=True):
    if virtual_axis:
        operator = BasicOperator(
            domain=isl.BasicSet(
                f"{{ [v0,ic,oh,ow,kh,kw]: 0 <= v0 < 1 and 0<=ic<{ic} and 0<=oh<{oh} and 0<=ow<{ow} and 0<=kh<{kh} and 0<=kw<{kw} }}"
            ),
            access_I=isl.BasicMap(
                f"{{ [v0, ic,oh,ow,kh,kw] -> I[ic, oh * {stride} + kh * {dilation}, ow * {stride} + kw * {dilation} ] }}"
            ),
            access_O=isl.BasicMap("{ [v0, ic,oh,ow,kh,kw] -> O[v0, ic, oh, ow] }"),
            access_W=isl.BasicMap("{ [v0, ic,oh,ow,kh,kw] -> W[v0, ic, kh, kw] }"),
        )
    else:
        operator = BasicOperator(
            domain=isl.BasicSet(
                f"{{ [ic,oh,ow,kh,kw]: 0<=ic<{ic} and 0<=oh<{oh} and 0<=ow<{ow} and 0<=kh<{kh} and 0<=kw<{kw} }}"
            ),
            access_I=isl.BasicMap(
                f"{{ [ic,oh,ow,kh,kw] -> I[ic, oh * {stride} + kh * {dilation}, ow * {stride} + kw * {dilation} ] }}"
            ),
            access_O=isl.BasicMap("{ [ic,oh,ow,kh,kw] -> O[ic, oh, ow] }"),
            access_W=isl.BasicMap("{ [ic,oh,ow,kh,kw] -> W[ic, kh, kw] }"),
        )
    return operator


def get_op_conv2d(b, oc, ic, oh, ow, kh, kw, stride=1, virtual_axis=True):
    operator = BasicOperator(
        domain=isl.BasicSet(
            f"{{ [b,oc,ic,oh,ow,kh,kw]: 0<=b<{b} and 0<=oc<{oc} and 0<=ic<{ic} and 0<=oh<{oh} and 0<=ow<{ow} and 0<=kh<{kh} and 0<=kw<{kw} }}"
        ),
        access_I=isl.BasicMap(
            f"{{ [b,oc,ic,oh,ow,kh,kw] -> I[b, ic, oh * {stride} + kh, ow * {stride} + kw] }}"
        ),
        access_O=isl.BasicMap("{ [b,oc,ic,oh,ow,kh,kw] -> O[b, oc, oh, ow] }"),
        access_W=isl.BasicMap("{ [b,oc,ic,oh,ow,kh,kw] -> W[oc, ic, kh, kw] }"),
    )
    return operator


def get_op_dwconv3d(ic, ox, oy, oz, kx, ky, kz, stride, virtual_axis=True):
    operator = BasicOperator(
        domain=isl.BasicSet(
            f"{{ [ic, ox, oy, oz, kx, ky, kz]: 0<=ic<{ic} and 0<=ox<{ox} and 0<=oy<{oy} and 0<=oz<{oz} and 0<=kx<{kx} and 0<=ky<{ky} and 0<=kz<{kz}  }}"
        ),
        access_I=isl.BasicMap(
            "{ [ic, ox, oy, oz, kx, ky, kz] -> I[ic, ox + kx, oy + ky, oz + kz] }"
        ),
        access_O=isl.BasicMap("{ [ic, ox, oy, oz, kx, ky, kz] -> O[ic, ox, oy, oz] }"),
        access_W=isl.BasicMap("{ [ic, ox, oy, oz, kx, ky, kz] -> W[ic, kx, ky, kz] }"),
    )
    return operator


def get_op_conv1d(oc, ic, oh, ow, k, virtual_axis=True):
    operator = BasicOperator(
        domain=isl.BasicSet(
            f"{{ [oc,ic,oh,ow,k]: 0<=oc<{oc} and 0<=ic<{ic} and 0<=oh<{oh} and 0<=ow<{ow} and 0<=k<{k} }}"
        ),
        access_I=isl.BasicMap(f"{{ [oc,ic,oh,ow,k] -> I[oc, ic, oh + k, ow + k] }}"),
        access_O=isl.BasicMap("{ [oc,ic,oh,ow,k] -> O[oc, ic, oh, ow] }"),
        access_W=isl.BasicMap("{ [oc,ic,oh,ow,k] -> W[oc, ic, k] }"),
    )
    return operator
