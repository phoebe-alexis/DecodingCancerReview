updated_training_code = """
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD, Adam
import torch.nn as nn
import torch as tc
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import torchtuples as tt
import numpy as np
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
import os
import pandas as pd
from NN import Simple_Model
from data import combine_data
import gc

def train_test(model, data_collection, setting, fold, PATH='./results/training/', load_weights=True):
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    assert fold == data_collection.current_test_split, 'fold not right'

    device = tc.device(setting['training_device'])

    if load_weights and os.path.exists('./results/raw_params.pt'):
        print("📦 Loading raw_params.pt...")
        model.load_state_dict(tc.load('./results/raw_params.pt'))

    model.train().to(device)

    patdata_train, surv_train = data_collection.get_train_set()
    patdata_test_val, surv_test_val = data_collection.get_test_set()

    tv_length = patdata_test_val.shape[0]
    patdata_test, surv_test = patdata_test_val[:tv_length // 2, :], surv_test_val[:tv_length // 2, :]
    patdata_val, surv_val = patdata_test_val[tv_length // 2:, :], surv_test_val[tv_length // 2:, :]

    samplenames_test_val = data_collection.get_test_names()
    test_names = samplenames_test_val[:tv_length // 2]
    val_names = samplenames_test_val[tv_length // 2:]

    test_name_frame = pd.DataFrame({'sample_name': test_names, 'type': 'test', 'fold': fold})
    val_name_frame = pd.DataFrame({'sample_name': val_names, 'type': 'val', 'fold': fold})
    train_name_frame = pd.DataFrame({'sample_name': data_collection.get_train_names(), 'type': 'train', 'fold': fold})
    name_frame = pd.concat([test_name_frame, val_name_frame, train_name_frame], axis=0)
    name_frame.to_csv('./results/training/name_frame.csv', mode='w' if fold == 0 else 'a', header=fold == 0)

    print('train size:', patdata_train.shape, 'test_size:', patdata_test.shape)

    test_data = (patdata_test, (surv_test[:, 0], surv_test[:, 1]))
    val_data = (patdata_val, (surv_val[:, 0], surv_val[:, 1]))

    coxph_model = CoxPH(model, tt.optim.Adam(setting['lr'], weight_decay=setting['weight_decay']), device=device)

    effective_epochs = []
    for exp, training_epochs in enumerate(setting['reduce_lr_epochs']):
        callbacks = [tt.callbacks.EarlyStopping(), tt.callbacks.ClipGradNorm(model, max_norm=1.0)]
        print(f'train for {training_epochs} epochs with lr {setting["lr"] * 10 ** (-exp)}')
        coxph_model.optimizer.set_lr(setting['lr'] * 10 ** (-exp))
        log = coxph_model.fit(patdata_train, (surv_train[:, 0], surv_train[:, 1]),
                              batch_size=setting['training_batch_size'], epochs=training_epochs, callbacks=callbacks,
                              val_data=val_data, val_batch_size=setting['training_batch_size'], verbose=1)
        effective_epochs.append(log.epoch)

    _ = coxph_model.compute_baseline_hazards()
    surv_pred = coxph_model.predict_surv_df(test_data[0])

    ev = EvalSurv(surv_pred, np.array(test_data[1][0]).squeeze(), np.array(test_data[1][1]).squeeze(), censor_surv='km')
    concordance = ev.concordance_td()
    time_grid = np.linspace(np.array(test_data[1][0]).squeeze().min(), np.array(test_data[1][0]).squeeze().max(), 100)
    integrated_brier_score = ev.integrated_brier_score(time_grid)
    integrated_nbll = ev.integrated_nbll(time_grid)

    print('concordance:', concordance)
    print('integrated_brier_score:', integrated_brier_score)
    print('integrated_nbll:', integrated_nbll)

    pd.DataFrame({
        'fold': [fold],
        'conc_score': concordance,
        'brier_score': integrated_brier_score,
        'ncancers': test_data[0].shape[0]
    }).to_csv('./results/training/conc_scores.csv', mode='w' if fold == 0 else 'a', header=fold == 0)

    strat_conc_scores = conc_score_per_cancer(coxph_model, data_collection)
    strat_conc_scores['fold'] = fold
    strat_conc_scores.to_csv('./results/training/stratified_conc_scores.csv', mode='w' if fold == 0 else 'a', header=fold == 0)

    return model


def conc_score_per_cancer(trained_model, data_collection):
    patdata_test_val, surv_test_val = data_collection.get_test_set()
    tv_length = patdata_test_val.shape[0]
    patdata_test, surv_test = patdata_test_val[:tv_length // 2, :], surv_test_val[:tv_length // 2, :]

    cancer_types_test = data_collection.get_cancer_types_test()[:tv_length // 2]
    results = []

    for cancer_type in data_collection.unique_cancer_types:
        current_ids = cancer_types_test == cancer_type
        test_data = (patdata_test[current_ids, :], (surv_test[current_ids, 0], surv_test[current_ids, 1]))

        if test_data[0].shape[0] < 10 or np.array(test_data[1][1]).sum() <= 5:
            continue

        surv_pred = trained_model.predict_surv_df(test_data[0])
        ev = EvalSurv(surv_pred, np.array(test_data[1][0]).squeeze(), np.array(test_data[1][1]).squeeze(), censor_surv='km')

        concordance = ev.concordance_td()
        time_grid = np.linspace(np.array(test_data[1][0]).squeeze().min(), np.array(test_data[1][0]).squeeze().max(), 100)
        integrated_brier_score = ev.integrated_brier_score(time_grid)

        results.append(pd.DataFrame({
            'cancer_type': [cancer_type],
            'concordance': concordance,
            'integrated_brier': integrated_brier_score,
            'ncancers_test': test_data[0].shape[0]
        }))

        gc.collect()

    return pd.concat(results, axis=0)


def save_predictions(model, data_collection, setting, fold, PATH='./results/training/'):
    device = tc.device(setting['training_device'])
    model.eval().to(device)

    x, _ = data_collection.get_test_set()
    x = x.to(device)
    pred = model(x).detach().cpu().numpy().squeeze()

    sample_names = data_collection.get_test_names()
    df = pd.DataFrame({'sample_name': sample_names, 'risk_prediction_all': pred})
    df.to_csv(os.path.join(PATH, 'risk_predictions.csv'), mode='w' if fold == 0 else 'a', header=fold == 0)
"""

# Save to file
with open("updated_training.py", "w") as f:
    f.write(updated_training_code)

print("updated_training.py created in current working directory.")
