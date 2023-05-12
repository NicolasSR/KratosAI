from nn_trainer import NN_Trainer

if __name__ == "__main__":

    ae_config_list = [{
        "nn_type": 'conv2d_smain', # ['dense_umain','conv2d_umain','dense_rmain','conv2d_rmain']
        "name": 'NewCode_NoForce_AugmFinetune_RandomDatabase_finetuneColab_w0.1_lam0.0_lr0.00001',
        "encoding_size": 1,
        "hidden_layers": ((16,(3,5),(1,2)),
                          (32,(3,5),(1,2))
                          ),
        "batch_size": 1,
        "epochs": 100,
        "normalization_strategy": 'channel_range',  # ['feature_stand','channel_range', 'channel_scale']
        "residual_loss_ratio": ('const', 0.1), # ('linear', 0.99999, 0.1, 100), ('const', 1.0), ('binary', 0.99999, 0.0, 2)
        "orthogonal_loss_ratio": ('const', 0.0),
        "learning_rate": ('const', 0.00001), # ('steps', 0.001, 10, 1e-6, 100), ('const', 0.001)
        "dataset_path": 'datasets_rommanager/',
        "models_path": 'saved_models_conv2d/',
        "finetune_from": 'saved_models_conv2d/W0_Colab_RandomDatabase/',
        "augmented": True,
        "use_force": False,
        "use_bias": True,
        "svd_tolerance": None
     }]
    
    for i, ae_config in enumerate(ae_config_list):
        
        print('----------  Training case ', i+1, ' of ', len(ae_config_list), '  ----------')
        training_routine=NN_Trainer(ae_config)
        training_routine.execute_training()
        print('FINISHED TRAINING')

        
