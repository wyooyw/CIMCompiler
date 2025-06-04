import math


def im2col(
    image_col,
    image_row,
    filter_col,
    filter_row,
    in_channel,
    out_channel,
    array_row,
    array_col,
):

    col_slide = image_col - filter_col + 1
    row_slide = image_row - filter_row + 1

    col_cycle = math.ceil(out_channel / array_col)
    row_cycle = math.ceil(filter_row * filter_col * in_channel / array_row)
    total_cycle = col_slide * row_slide * row_cycle * col_cycle

    return total_cycle


def SDK(
    image_col,
    image_row,
    filter_col,
    filter_row,
    in_channel,
    out_channel,
    array_row,
    array_col,
):

    row_vector = filter_row * filter_col * in_channel
    col_vector = out_channel

    used_row = math.ceil(row_vector / array_row)
    used_col = math.ceil(col_vector / array_col)

    new_array_row = array_row * used_row
    new_array_col = array_col * used_col

    # initialize
    cycle = []
    w = []
    w.append(filter_row * filter_col)
    cycle.append(
        used_row
        * used_col
        * (image_row - filter_row + 1)
        * (image_col - filter_col + 1)
    )

    i = 0
    while True:
        i += 1
        pw_row = filter_row + i - 1
        pw_col = filter_col + i - 1
        pw = pw_row * pw_col
        if pw * in_channel <= new_array_row and i * i * out_channel <= new_array_col:
            parallel_window_row = math.ceil((image_row - (filter_row + i) + 1) / i) + 1
            parallel_window_col = math.ceil((image_col - (filter_col + i) + 1) / i) + 1

            if (
                parallel_window_row * parallel_window_row * used_row * used_col
                <= cycle[0]
            ):
                del cycle[0]
                del w[0]
                cycle.append(
                    parallel_window_row * parallel_window_col * used_row * used_col
                )

                w.append(pw)

        else:
            break

    return cycle, w


# ceil : up, floor : down
def vw_sdk(
    image_col,
    image_row,
    filter_col,
    filter_row,
    in_channel,
    out_channel,
    array_row,
    array_col,
):

    i = 0  # initialize # overlap col
    j = 1  # overlap row

    reg_total_cycle = []  # initialize
    reg_overlap_row = []
    reg_overlap_col = []
    reg_row_cycle = []
    reg_col_cycle = []
    reg_ICt = []
    reg_OCt = []
    while True:
        try:
            i += 1
            if (i + filter_col) > image_col:
                i = 1
                j += 1
                if j + filter_row > image_row:
                    break

            # for parallel_window computing
            reg_N_parallel_window_row = (
                math.ceil((image_row - (filter_row + i) + 1) / i) + 1
            )
            reg_N_parallel_window_col = (
                math.ceil((image_col - (filter_col + j) + 1) / j) + 1
            )

            # for cycle computing
            # Tiled IC
            if in_channel == 3:
                ICt = math.floor(
                    array_row / ((filter_row + i - 1) * (filter_col + j - 1))
                )
                if ICt > in_channel:
                    ICt = 3
                row_cycle = math.ceil(in_channel / ICt)
            else:
                ICt = math.floor(
                    array_row / ((filter_row + i - 1) * (filter_col + j - 1))
                )
                row_cycle = math.ceil(in_channel / ICt)

            # Tiled OC
            OCt = math.floor(array_col / (i * j))
            col_cycle = math.ceil(out_channel / OCt)

            reg_N_of_computing_cycle = (
                reg_N_parallel_window_row
                * reg_N_parallel_window_col
                * row_cycle
                * col_cycle
            )

            if i == 1:  # initialize
                reg_total_cycle.append(reg_N_of_computing_cycle)
                reg_overlap_row.append(i)
                reg_overlap_col.append(j)
                reg_row_cycle.append(row_cycle)
                reg_col_cycle.append(col_cycle)
                reg_ICt.append(ICt)
                reg_OCt.append(OCt)

            if reg_total_cycle[0] > reg_N_of_computing_cycle:
                del reg_total_cycle[0]
                del reg_overlap_row[0]
                del reg_overlap_col[0]
                del reg_row_cycle[0]
                del reg_col_cycle[0]
                del reg_ICt[0]
                del reg_OCt[0]

                reg_total_cycle.append(reg_N_of_computing_cycle)
                reg_overlap_row.append(i)
                reg_overlap_col.append(j)
                reg_row_cycle.append(row_cycle)
                reg_col_cycle.append(col_cycle)
                reg_ICt.append(ICt)
                reg_OCt.append(OCt)

            # if reg_N_of_computing_cycle==256:
            #     import pdb; pdb.set_trace()
            #     pass

        except ZeroDivisionError:
            continue

    return (
        reg_total_cycle[0],
        reg_overlap_col[0],
        reg_overlap_row[0],
        reg_row_cycle[0],
        reg_col_cycle[0],
        reg_ICt[0],
        reg_OCt[0],
    )


def network_information(network, image, array, kernel, channel):
    print("=" * 50)
    print(" Network : ", network)
    print(" Array Size = {} x {}".format(array[0], array[1]))
    print("-" * 30)

    print(" NETWORK INFORMATION ")
    print("-" * 30)
    for i in range(len(image)):
        print(" CONV LAYER " + str(i + 1))
        print("    Image   Size = {} x {}".format(image[i], image[i]))
        print("    Kernel  Size = {} x {}".format(kernel[i], kernel[i]))
        if network == "VGG13":
            print("    Channel Size = {} x {}".format(channel[i], channel[i + 1]))
        elif network == "Resnet13":
            print("    Channel Size = {} x {}".format(channel[i], channel[i]))
    print("=" * 50)


def result(network, image, array, kernel, channel):
    CC_im2col = []

    CC = []
    SDK_height = []
    SDK_width = []
    AR_cycle = []
    AC_cycle = []
    IC_tiled = []
    OC_tiled = []

    CC_SDK = []
    PW_SDK = []

    print("=" * 50)
    print(" RESULTS of COMPUTING CYCLES")
    print("-" * 30)

    if network == "VGG13":

        for i in range(len(image)):
            T_im2col = im2col(
                image[i],
                image[i],
                kernel[i],
                kernel[i],
                channel[i],
                channel[i + 1],
                array[0],
                array[1],
            )
            CC_im2col.append(T_im2col)

            T_SDK, w = SDK(
                image[i],
                image[i],
                kernel[i],
                kernel[i],
                channel[i],
                channel[i + 1],
                array[0],
                array[1],
            )
            CC_SDK.append(T_SDK[0])
            PW_SDK.append(w[0])

            T_cycle, SDK_h, SDK_w, ARC, ACC, tiled_IC, tiled_OC = vw_sdk(
                image[i],
                image[i],
                kernel[i],
                kernel[i],
                channel[i],
                channel[i + 1],
                array[0],
                array[1],
            )
            CC.append(T_cycle)
            SDK_height.append(SDK_h)
            SDK_width.append(SDK_w)
            AR_cycle.append(ARC)
            AC_cycle.append(ACC)
            IC_tiled.append(tiled_IC)
            OC_tiled.append(tiled_OC)

        for i in range(len(image)):
            print(" CONV LAYER " + str(i + 1))
            print("    Im2col = {}".format(CC_im2col[i]))
            print("    SDK    = {}".format(CC_SDK[i]))
            if i == 0:
                print(
                    "      - shape of PW = {} x {} x {} x {}".format(
                        int(math.sqrt(PW_SDK[i])),
                        int(math.sqrt(PW_SDK[i])),
                        channel[i],
                        channel[i + 1],
                    )
                )
            else:
                print(
                    "      - shape of PW = {} x {} x {} x {}".format(
                        int(math.sqrt(PW_SDK[i])),
                        int(math.sqrt(PW_SDK[i])),
                        channel[i],
                        channel[i + 1],
                    )
                )

            if CC[i] >= CC_im2col[i]:
                CC[i] = CC_im2col[i]
                print("    VW-SDK = {}".format(CC[i]))
                print(
                    "      - Optimal shape of PW = {} x {} x {} x {}".format(
                        kernel[i], kernel[i], channel[i], channel[i + 1]
                    )
                )
                print(
                    "      - Reduction Compared to Im2col = {:.2f} %".format(
                        (CC_im2col[i] - CC[i]) / CC_im2col[i] * 100
                    )
                )
                print(
                    "      - Reduction Compared to SDK    = {:.2f} %".format(
                        (CC_SDK[i] - CC[i]) / CC_SDK[i] * 100
                    )
                )

            else:
                print("    VW-SDK = {}".format(CC[i]))
                print(
                    "      - Optimal shape of PW = {} x {} x {} x {}".format(
                        kernel[i] + SDK_width[i] - 1,
                        kernel[i] + SDK_height[i] - 1,
                        IC_tiled[i],
                        OC_tiled[i],
                    )
                )
                print(
                    "      - Reduction Compared to Im2col = {:.2f} %".format(
                        (CC_im2col[i] - CC[i]) / CC_im2col[i] * 100
                    )
                )
                print(
                    "      - Reduction Compared to SDK    = {:.2f} %".format(
                        (CC_SDK[i] - CC[i]) / CC_SDK[i] * 100
                    )
                )
        print("=" * 50)

    elif network == "Resnet18":

        for i in range(len(image)):
            if i == 0:
                T_im2col = im2col(
                    image[i],
                    image[i],
                    kernel[i],
                    kernel[i],
                    channel[i],
                    channel[i + 1],
                    array[0],
                    array[1],
                )
                T_SDK, w = SDK(
                    image[i],
                    image[i],
                    kernel[i],
                    kernel[i],
                    channel[i],
                    channel[i + 1],
                    array[0],
                    array[1],
                )
                T_cycle, SDK_h, SDK_w, ARC, ACC, tiled_IC, tiled_OC = vw_sdk(
                    image[i],
                    image[i],
                    kernel[i],
                    kernel[i],
                    channel[i],
                    channel[i + 1],
                    array[0],
                    array[1],
                )

            else:
                T_im2col = im2col(
                    image[i],
                    image[i],
                    kernel[i],
                    kernel[i],
                    channel[i],
                    channel[i],
                    array[0],
                    array[1],
                )
                T_SDK, w = SDK(
                    image[i],
                    image[i],
                    kernel[i],
                    kernel[i],
                    channel[i],
                    channel[i],
                    array[0],
                    array[1],
                )
                T_cycle, SDK_h, SDK_w, ARC, ACC, tiled_IC, tiled_OC = vw_sdk(
                    image[i],
                    image[i],
                    kernel[i],
                    kernel[i],
                    channel[i],
                    channel[i],
                    array[0],
                    array[1],
                )

            CC_im2col.append(T_im2col)

            CC_SDK.append(T_SDK[0])
            PW_SDK.append(w[0])

            CC.append(T_cycle)
            SDK_height.append(SDK_h)
            SDK_width.append(SDK_w)
            AR_cycle.append(ARC)
            AC_cycle.append(ACC)
            IC_tiled.append(tiled_IC)
            OC_tiled.append(tiled_OC)

        for i in range(len(image)):
            print(" CONV LAYER " + str(i + 1))
            print("    Im2col = {}".format(CC_im2col[i]))
            print("    SDK    = {}".format(CC_SDK[i]))
            if i == 0:
                print(
                    "      - shape of PW = {} x {} x {} x {}".format(
                        int(math.sqrt(PW_SDK[i])),
                        int(math.sqrt(PW_SDK[i])),
                        channel[i],
                        channel[i + 1],
                    )
                )
            else:
                print(
                    "      - shape of PW = {} x {} x {} x {}".format(
                        int(math.sqrt(PW_SDK[i])),
                        int(math.sqrt(PW_SDK[i])),
                        channel[i],
                        channel[i],
                    )
                )

            if CC[i] > CC_im2col[i]:
                CC[i] = CC_im2col[i]
                print("    VW-SDK = {}".format(CC[i]))
                if i == 0:
                    print(
                        "      - Optimal shape of PW = {} x {} x {} x {}".format(
                            kernel[i], kernel[i], channel[i], channel[i + 1]
                        )
                    )
                else:
                    print(
                        "      - Optimal shape of PW = {} x {} x {} x {}".format(
                            kernel[i], kernel[i], channel[i], channel[i]
                        )
                    )
                print(
                    "      - Reduction Compared to Im2col = {:.2f} %".format(
                        (CC_im2col[i] - CC[i]) / CC_im2col[i] * 100
                    )
                )
                print(
                    "      - Reduction Compared to SDK    = {:.2f} %".format(
                        (CC_SDK[i] - CC[i]) / CC_SDK[i] * 100
                    )
                )

            else:
                print("    VW-SDK = {}".format(CC[i]))
                print(
                    "      - Optimal shape of PW = {} x {} x {} x {}".format(
                        kernel[i] + SDK_width[i] - 1,
                        kernel[i] + SDK_height[i] - 1,
                        IC_tiled[i],
                        OC_tiled[i],
                    )
                )
                print(
                    "      - Reduction Compared to Im2col = {:.2f} %".format(
                        (CC_im2col[i] - CC[i]) / CC_im2col[i] * 100
                    )
                )
                print(
                    "      - Reduction Compared to SDK    = {:.2f} %".format(
                        (CC_SDK[i] - CC[i]) / CC_SDK[i] * 100
                    )
                )
        print("=" * 50)
