from Scripts.utils.metrics import metrics
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('-d', '--device', default="cuda:0", type=str, help='device to use for training')
parser.add_argument('-b', '--batch_size', default=1, type=int, help='batch size for training')
parser.add_argument('-s', '--seed', default=[0], type=int, nargs="+", help='random seed for training')
parser.add_argument('--database_name', default=["BUSI"], type=str, nargs="+", help='database name for training')
parser.add_argument('-m', '--model_name', default=["Unet"], type=str, nargs="+", help='model name for training')
parser.add_argument('--image_size', default=512, type=int, help='image size for training')
parser.add_argument('--in_channels', default=1, type=int, help='input channels')
parser.add_argument('--out_channels', default=1, type=int, help='output channels')
parser.add_argument('--hidden_channels', default=64, type=int, help='hidden channels')
parser.add_argument('-p', '--drop_out', default=0.0, type=float, help='drop out rate')
args = parser.parse_args()


for database_name in args.database_name:
    m = metrics(csv_path=f"Results/metrics_{database_name}.csv", device=args.device)
    for seed in args.seed:
        m.load_dataset(database_name, args.batch_size, seed)
        for model_name in args.model_name:
            m.load_model(model_name, args.in_channels, args.out_channels, args.hidden_channels, seed)
            m.evaluate()