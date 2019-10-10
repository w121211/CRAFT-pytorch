import argparse


def str2bool(v):
    return v.lower() in ("true")


def get_parameters():
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument("--model", type=str, default="sagan", choices=["sagan", "qgan"])
    parser.add_argument(
        "--adv_loss", type=str, default="wgan-gp", choices=["wgan-gp", "hinge"]
    )
    parser.add_argument("--imsize", type=int, default=32)
    parser.add_argument("--im_channels", type=int, default=3)
    parser.add_argument("--g_num", type=int, default=5)
    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--g_conv_dim", type=int, default=64)
    parser.add_argument("--d_conv_dim", type=int, default=64)
    parser.add_argument("--lambda_gp", type=float, default=10)
    # parser.add_argument("--version", type=str, default="sagan_1")
    parser.add_argument("--clip_value", type=float, default=0.01)
    parser.add_argument("--n_masks", type=int, default=2)

    # Training setting
    parser.add_argument("--n_epochs", type=int, default=1000)
    parser.add_argument(
        "--total_step",
        type=int,
        default=1000000,
        help="how many times to update the generator",
    )
    parser.add_argument("--d_iters", type=float, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--g_lr", type=float, default=0.0001)
    parser.add_argument("--d_lr", type=float, default=0.0004)
    parser.add_argument("--lr_decay", type=float, default=0.95)
    parser.add_argument("--beta1", type=float, default=0.0)
    parser.add_argument("--beta2", type=float, default=0.9)
    parser.add_argument(
        "--n_critic",
        type=int,
        default=50,
        help="number of training steps for discriminator per iter",
    )
    parser.add_argument("--sample_interval", type=int, default=400)
    parser.add_argument("--save_interval", type=int, default=400)

    # using pretrained
    parser.add_argument("--pretrained_model", type=int, default=None)

    # Misc
    parser.add_argument("--train", type=str2bool, default=True)
    parser.add_argument("--parallel", type=str2bool, default=False)
    parser.add_argument(
        "--dataset", type=str, default="cifar", choices=["lsun", "celeb"]
    )
    parser.add_argument("--use_tensorboard", type=str2bool, default=False)

    # Path
    # parser.add_argument("--image_path", type=str, default="./out/images")
    parser.add_argument("--log_path", type=str, default="./out/logs")
    parser.add_argument("--model_save_path", type=str, default="./out/models")
    parser.add_argument("--sample_path", type=str, default="./out/samples")
    parser.add_argument("--attn_path", type=str, default="./out/attn")

    # Step size
    parser.add_argument("--log_step", type=int, default=10)
    parser.add_argument("--sample_step", type=int, default=100)

    # Dataset
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--n_layers", type=int, default=3)

    return parser.parse_args()
