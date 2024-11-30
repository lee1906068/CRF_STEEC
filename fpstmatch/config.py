import argparse


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    # parser.add_argument("--eval_bsize", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200, help='max epochs')
    parser.add_argument("--lr", type=float, default=0.01,
                        help='learning rate')
    parser.add_argument("--prcdata_path",
                        type=str,
                        default="prcdata", help='')
    parser.add_argument(
        "--prcdata_dp_folder",
        type=str,
        default="dp_merge_ex/dp_merge_0.6", help='')

    parser.add_argument("--exited_model_path", type=str, default="")
    parser.add_argument(
        "--prcdata_gridroadnet_folder",
        type=str,
        default="gridroadnet_bj", help='')
    parser.add_argument(
        "--prcdata_sampledtraj_folder",
        type=str,
        default="sampledtraj_50", help='')

    parser.add_argument("--emb_linear_dim", type=int, default=128,
                        help='embedding linear dimension')
    parser.add_argument("--emb_transformer_dim", type=int, default=64,
                        help='embedding transformer dimension')
    # parser.add_argument("--layer", type=int, default=4, help='A^k num of layer neighbors')
    # parser.add_argument("--wd", type=float, default=1e-8, help='Adamw weight decay')
    # parser.add_argument("--dev_id", type=int, default=0, help='cuda id')
    # parser.add_argument("--bi", action="store_true", help='use biGRU')
    # parser.add_argument("--use_crf", action="store_true", help='use crf')
    # parser.add_argument("--atten_flag", action="store_true", help='use attention in seq2seq')
    #parser.add_argument("--eval_ratio", type=float,
    #                    default=0.5, help='ratio to eval in evalset')
    parser.add_argument("--idpdt_train", type=str, default="000",
                        help="'000' if only one object is needed to be loaded for training else None")
    # parser.add_argument("--gamma", type=float, default=10000, help='penalty for unreachable')
    # parser.add_argument("--topn", type=int, default=5, help='select topn in test mode')
    # parser.add_argument("--neg_nums", type=int, default=800, help='select negetive sampling number')
    args, _ = parser.parse_known_args()

    return args


if __name__ == "__main__":
    pass
