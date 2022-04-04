import pickle

# from itertools import zip_longest


def flatten_dict(d, pre_lst=None, result=None):
    if result is None:
        result = {}
    if pre_lst is None:
        pre_lst = []
    for k, v in d.items():
        if isinstance(v, dict):
            flatten_dict(v, pre_lst=pre_lst + [k], result=result)
        else:
            new_k = ""
            for lk in pre_lst:
                new_k += lk + "."
            new_k += k
            result[new_k] = v
    return result


def main():
    # input_pickle_filename = "./standard_files/pretrain_res50x1.pkl"
    # input_pickle_filename = "../tmp_out/raft-sintel/after_change/checkpoint_0109.pkl"
    input_pickle_filename = "./detectron2/pretrain_res50x1.pkl"
    # input_pickle_filename = "./tmp/pretrain_res50x1.pkl"
    input_resnet50_torch_filename = "../R-50.pkl"
    with open(input_pickle_filename, "rb") as f:
        data = pickle.load(f)
    with open(input_resnet50_torch_filename, "rb") as f:
        torch_resnet50_data = pickle.load(f)
    # data_keys = list(data.keys())
    # print(data_keys)
    torch_resnet50_data_keys = list(torch_resnet50_data.keys())
    # torch_model_state_dict = torch_resnet50_data[torch_resnet50_data_keys[0]]
    model_state_dict = data["model"]
    torch_model_state_dict = torch_resnet50_data[torch_resnet50_data_keys[0]]
    # model_state_dict_keys = list(model_state_dict.keys())
    # print(model_state_dict_keys)
    # print(torch_model_state_dict.keys())
    # print(model_state_dict[model_state_dict_keys[0]].keys())
    # tmp_state_dict = {}
    # for k, v in model_state_dict[model_state_dict_keys[0]].items():
    #     tmp_k = k.replace("~", "")
    #     tmp_k = tmp_k.replace("//", "/")
    #     tmp_k = tmp_k.replace("/", ".")
    #     # tmp_k = tmp_k.replace("_", ".")
    #     tmp_state_dict[tmp_k] = v
    # tmp_state_dict_flat = flatten_dict(tmp_state_dict)
    # tmp_res_state = [k for k in tmp_state_dict_flat.keys() if "res" in k]
    # tmp_res_state = [k for k in tmp_res_state if "initial" not in k]
    # print([k for k in tmp_state_dict.keys() if "res" not in k])
    # res_state = [k for k in tmp_state_dict.keys() if "res" in k]
    # res_state_torch = [k for k in torch_model_state_dict.keys() if "res" in k]
    tmp_res_state = list(model_state_dict.keys())
    tmp_res_state = [k.replace("backbone.", "") for k in tmp_res_state]
    tmp_res_state = [k for k in tmp_res_state if "flow" not in k]
    res_state_torch = list(torch_model_state_dict.keys())
    res_state_torch = [k for k in res_state_torch if "run" not in k]
    res_state_torch = [k for k in res_state_torch if "fc" not in k]
    tmp_res_state = [k.replace(".", "_") for k in tmp_res_state]
    res_state_torch = [k.replace(".", "_") for k in res_state_torch]
    # res_state_torch = [k for k in res_state_torch if "stem" not in k]
    # res_state = list(tmp_state_dict.keys())
    # res_state_torch = list(torch_model_state_dict.keys())
    # res_state = sorted(res_state)
    tmp_res_state = sorted(tmp_res_state)
    res_state_torch = sorted(res_state_torch)
    # print(len(res_state), len(res_state_torch))
    # for k, tk in zip_longest(res_state, res_state_torch):
    #     tmp, tmp_keys = None, []
    #     if k:
    #         tmp = tmp_state_dict[k]
    #         tmp_keys = tmp.keys()
    #     print(tmp_keys, k, " : ", tk)
    # len_res_state = len(res_state)
    # j = 0
    # for i in range(len_res_state):
    #     k = res_state[i]
    #     tmp_dict = tmp_state_dict[k]
    #     tk = res_state_torch[j]
    #     print(i, k, " : ", tk, j)
    #     for lj, lk in enumerate(tmp_dict.keys()):
    #         lv = tmp_dict[lk]
    #         tk = res_state_torch[j]
    #         print(type(lv), i, k, lk, " : ", tk, j)
    #         j += 1
    #     # j += 1
    # for k in res_state_torch[j:]:
    #     print(k)

    print(len(tmp_res_state), len(res_state_torch))
    is_all_same_key = True
    for k, tk in zip(tmp_res_state, res_state_torch):
        is_same = k == tk
        print(k, f"\n{tk}", f"\n{is_same}")
        is_all_same_key = is_all_same_key and is_same
        # v = model_state_dict["backbone." + k.replace("_", ".")]
        # tv = torch_model_state_dict[tk.replace("_", ".")]
        # print(type(v), f"\n{type(tv)}", "\n")
        # print(v.shape, f"\n{tv.shape}", "\n")
    print(is_all_same_key)


if __name__ == "__main__":
    main()
