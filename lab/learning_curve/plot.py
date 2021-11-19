
import os
import glob
from numpy import tile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

methods = ['STARTUP_na'] #['vanilla', 'BMS_in', 'BAS_in', 'baseline', 'baseline_na']
target_datasets = ['EuroSAT', 'CropDisease', 'ISIC', 'ChestX']

def get_logs_path(method, target = ''):
    root = './logs'
    method_path = root + '/' + method
    if os.path.isdir(method_path) == False:
        print('The methode {}\'s  path doesn\'t exist'.format(method))
    if target == '':
        return method_path
    log_path = method_path + '/' + target    
    return log_path


def plot_all():
    for method in methods:
        for target in target_datasets:
            log_path = get_logs_path(method, target)
            best_check = log_path + '/' + 'checkpoint_best.pkl'
            train_log = glob.glob(log_path + '/' + 'train_*.csv')
            val_log = glob.glob(log_path + '/' + 'val_*.csv')
            if (len(train_log) == 0) or (len(val_log) == 0):
                raise ValueError('The path {} does not contain logs'.format(log_path))
                continue
            elif (len(train_log) > 1) or (len(val_log) > 1):
                raise ValueError('The path {} contains extra logs'.format(log_path))
                continue
            else:
                train_log = train_log[0].replace('\\', '/')
                val_log = val_log[0].replace('\\', '/')
                df = pd.read_csv(val_log)
                columns = df.columns
                df = pd.DataFrame(np.repeat(df.values,2,axis=0))
                df.columns = columns
                df['Loss_train'] = pd.read_csv(train_log)['Loss']
                df.plot( y=["Loss_train", 'Loss_test'])
                df.plot( y=["top1_base_test"] )
                plt.title('{0}_{1}'.format(method, target))
                plt.show()

def compare_methodes(method1,method2, target1, target2):
    root = './logs'
    method1_path = get_logs_path(method1, target1)
    method2_path = get_logs_path(method2, target2)
    
    method1_check = method1_path + '/' + 'checkpoint_best.pkl'
    method2_check = method2_path + '/' + 'checkpoint_best.pkl'
    method1_train_log = glob.glob(method1_path + '/' + 'train_*.csv')
    method1_val_log = glob.glob(method1_path + '/' + 'val_*.csv')
    method2_train_log = glob.glob(method2_path + '/' + 'train_*.csv')
    method2_val_log = glob.glob(method2_path + '/' + 'val_*.csv')
    
    method1_train_log = method1_train_log[0].replace('\\', '/')
    method1_val_log = method1_val_log[0].replace('\\', '/')
    method2_train_log = method2_train_log[0].replace('\\', '/')
    method2_val_log = method2_val_log[0].replace('\\', '/')
    
    df_method1_train = pd.read_csv(method1_train_log)
    df_method1_val = pd.read_csv(method1_val_log)
    df_method2_train = pd.read_csv(method2_train_log)
    df_method2_val = pd.read_csv(method2_val_log)
    
    columns = df_method1_val.columns
    df_method1_val = pd.DataFrame(np.repeat(df_method1_val.values,2,axis=0))
    df_method2_val = pd.DataFrame(np.repeat(df_method2_val.values,2,axis=0))   
    df_method1_val.columns = columns 
    df_method2_val.columns = columns
        
    df = pd.DataFrame()
    df['method1_train_loss'] = df_method1_train['Loss']
    df['method2_train_loss'] = df_method2_train['Loss']
    df.plot( y=["method1_train_loss", 'method2_train_loss'])
    plt.axvline(x=332, color='blue')
    plt.axvline(x=402, color='orange')
    
    df['method1_val_loss'] = df_method1_val['Loss_test']
    df['method2_val_loss'] = df_method2_val['Loss_test']
    df.plot( y=["method1_val_loss", 'method2_val_loss'])
    plt.axvline(x=332, color='blue')
    plt.axvline(x=402, color='orange')
    
    df['{}_val_top1'.format(method1)] = df_method1_val['top1_base_test']
    df['{}_val_top1'.format(method2)] = df_method2_val['top1_base_test']
    df.plot( y=['{}_val_top1'.format(method1), '{}_val_top1'.format(method2)])
    plt.axvline(x=332, color='blue')
    plt.axvline(x=402, color='orange')
    
    plt.show()


def STARTUP_losses(method1, target1):
    root = './logs'
    method1_path = get_logs_path(method1, target1)
    
    method1_check = method1_path + '/' + 'checkpoint_best.pkl'
    method1_train_log = glob.glob(method1_path + '/' + 'train_*.csv')
    method1_val_log = glob.glob(method1_path + '/' + 'val_*.csv')
    
    method1_train_log = method1_train_log[0].replace('\\', '/')
    method1_val_log = method1_val_log[0].replace('\\', '/')
    
    df_method1_train = pd.read_csv(method1_train_log)
    df_method1_val = pd.read_csv(method1_val_log)
    
    # columns = df_method1_val.columns
    # df_method1_val = pd.DataFrame(np.repeat(df_method1_val.values,2,axis=0))  
    # df_method1_val.columns = columns
        
    df_method1_train.plot( y=['CE_Loss_source', "KL_Loss_target", 'SIMCLR_Loss_target'])
    # plt.axvline(x=354, color='blue')
    # plt.axvline(x=332, color='orange')		
    plt.title('Loss Train')
    
    df_method1_val.plot( y=["CE_Loss_source_test", 'KL_Loss_target_test', 'SIMCLR_Loss_target_test'])
    # plt.axvline(x=354, color='blue')
    # plt.axvline(x=332, color='orange')
    plt.title('Loss Val')

    
    plt.show()
    
    
def compare_ImageNets(methode_list):
    root = './logs'
    df = pd.DataFrame()
    for methode in methode_list:
        method_path = get_logs_path(methode, "")    
        method_check = method_path + '/' + 'checkpoint_best.pkl' 
        df_method = pd.read_csv(method_path + '/log.csv')    
        df[methode + ' Top1_Acc'] = df_method['Acc@1']
    
        # plt.axvline(x=332, color='blue')

    df.plot()
    plt.show()


# plot_all()

# plot_all()
# STARTUP_losses('STARTUP', 'EuroSAT')

# compare_methodes('baseline_na', 'baseline_nb', "", "")

compare_ImageNets(['ImageNet','ImageNet_na','ImageNet_nb','ImageNet_nw', 'ImageNet_na_PN'])
