import torch

from model.MCSFD_NET import  MCSFD_Net


import argparse
import os
import sys

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


torch.manual_seed(6)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='weibo2021',
                        choices=['weibo', '4charliehebdo','4ferguson','4germanwings-crash','4ottawashooting', '4sydneysiege'],
                        help='The name of dataset')
    parser.add_argument('--epochs', type=int, default= 200, help='The number of epochs to run')
    parser.add_argument('--start_epoch', type=int, default=47, help='Continue to train')
    parser.add_argument('--data_epochs', type=int, default=50, help='T he number of epochs to run')
    parser.add_argument('--data_start_epochs', type=int, default=0, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=32, help='The size of batch')
    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--num_posts', type=int, default=50)
    parser.add_argument('--n_critic', type=int, default=3, help='the adversarial')

    parser.add_argument('--data_syn_layer', type=int, default=2, help=' number layer for data synthsizing')
    parser.add_argument('--text_embedding', type =tuple, default=(768,200), help=' reduce the dimension of the text vector')
    parser.add_argument('--encoder_pars', type=tuple, default=(1, 200, 2,100,False), help='num_layers, f_in, n_head,f_out,dropout')
    parser.add_argument('--GIN_pars', type=tuple, default=('mean', 'mean', 2, 2, 3, 20), help='aggregation_op, readout_op,'
                                                                             ' num_aggregation_layers, mlp_num_layers, num_features, hidden-dim')
    parser.add_argument('--Temp_MLP', type=tuple,default=(50,20), help='num_nodes, hidden-dim')
    parser.add_argument('--loss_func_list', type=tuple, default=(50, 20), help='num_nodes, hidden-dim')
    parser.add_argument('--lr', type=tuple, default=1e-3, help='lr')
    parser.add_argument('--dropout', type=float, default=0.0, help='attention dropout in GAT')
    parser.add_argument('--weight_decay', type=float, default=10, help='weight_decay')
    parser.add_argument('--patience', type=int, default=30, help='early stopping')
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument('--save_dir', type=str, default=r'E:\WDK_workshop\PCL_rumor_detection\result', help='Directory name to save the model')
    parser.add_argument('--model_name', type=str, default=r'GateDDL_v8.1', help='Directory name to save the GAN')
    parser.add_argument('--data_dir', default=r'E:\WDK_workshop\PCL_rumor_detection\data\weibo2021', type=str)
    parser.add_argument('--target_domain', default=r'E:\WDK_workshop\PCL_rumor_detection\data\raw_data\weibo', type=str)
    parser.add_argument('--target_domain_dataset', default=r'charliehebdo', type=str)
    parser.add_argument('--data_eval', default='', type=str)

    args = parser.parse_args()
    print(args)

    torch.set_default_dtype(torch.float32)
    device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")

    model = MCSFD_Net(args, device)


    if not os.path.exists(os.path.join(args.save_dir, args.model_name)):
        os.mkdir(os.path.join(args.save_dir,  args.model_name))
    if not os.path.exists(os.path.join(args.save_dir, args.model_name, args.dataset)):
        os.mkdir(os.path.join(args.save_dir, args.model_name, args.dataset))
    # if not os.path.exists(
    #         os.path.join(args.save_dir, args.model_name, 'pretrain_dataGCN')):
    #     os.mkdir(os.path.join(args.save_dir, args.model_name, 'pretrain_dataGCN'))
    #
    # if os.path.exists(
    #         os.path.join(args.save_dir, args.model_name, 'pretrain_dataGCN', args.dataset+'_' + str(args.data_epochs-1) + '_pretrain_dataGAN.pkl')):
    #     model_path = os.path.join(args.save_dir, args.model_name, 'pretrain_dataGCN')
    #     data_start_epoch = model.load_pretrain_dataGAN(model_path, args.dataset, str(args.data_epochs-1))
    #     data_start_epoch += 1
    # else:
    #     data_start_epoch = 0
    data_start_epochs =0

    if os.path.exists(os.path.join(args.save_dir, args.model_name, args.dataset, str(args.start_epoch) + '_model_states.pkl')):
        model_path = os.path.join(args.save_dir, args.model_name, args.dataset)
        start_epoch = model.load(model_path, args.start_epoch)
        start_epoch = start_epoch + 1
    else:
        start_epoch = 0
        print("start from epoch {}".format(start_epoch))
        argsDict = args.__dict__
        with open(os.path.join(args.save_dir, args.model_name, args.dataset, 'setting.txt'), 'w') as f:
            f.writelines('------------------ start ------------------' + '\n')
            for eachArg, value in argsDict.items():
                f.writelines(eachArg + ' : ' + str(value) + '\n')
            f.writelines('------------------- end -------------------')



    model.train_epoch(args.data_dir, start_epoch, data_start_epochs)
    # model.test(args.target_domain)
    print(" [*] Training finished!")







