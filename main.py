"""
Main script that trains, validates, and evaluates
various models including AASIST.

AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import argparse
import json
import os
import sys
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Dict, List, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA

from data_utils import (Dataset_IDASVspoofing_train, Dataset_IDASVspoofing_val, genSpoof_list)
from evaluation import calculate_EER
from utils import create_optimizer, seed_worker, set_seed, str_to_bool

warnings.filterwarnings("ignore", category=FutureWarning)

def main(args: argparse.Namespace) -> None:
    # Load experiment configs
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    
    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]
    
    if "eval_all_best" not in config:
        config["eval_all_best"] = "True"
    if "freq_aug" not in config:
        config["freq_aug"] = "False"
    
    # Create seed reproducibility  
    set_seed(args.seed, config)
    
    # Define database related paths
    output_dir = Path(args.output_dir)
    database_path = Path(config["database_path"])
    
    val_trial_path = (database_path / "IDASVspoofing.cm.val.txt.joined")
    test_trial_path = (database_path / "IDASVspoofing.cm.test.txt.joined")
    
    # Define model related paths
    model_tag = "{}_{}_ep{}_bs{}".format(
        "IDASVspoofing",
        os.path.splitext(os.path.basename(args.config))[0],
        config["num_epochs"],
        config["batch_size"]
    )
    
    if args.comment:
        model_tag = model_tag + "_{}".format(args.comment)
    
    model_tag = output_dir / model_tag
    model_save_path = model_tag / "weights"
    test_score_path = model_tag / config["eval_output"]
    writer = SummaryWriter(model_tag)
    os.makedirs(model_save_path, exist_ok=True)
    copy(args.config, model_tag / "config.conf")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # TODO: Comment out this line to test by CPU training
    if device == "cpu": 
        raise ValueError("GPU not detected!")
    
    # Define model architecture
    model = get_model(model_config, device)
    
    # Define dataloaders
    train_loader, val_loader, test_loader = get_loader(
        database_path, args.seed, config
    )
    
    if args.eval:
        model.load_state_dict(
            torch.load(config["model_path"], map_location=device)
        )
        
        print("Model loaded: {}".format(config["model_path"]))
        print("Start evaluation...")
        
        produce_evaluation_file(
            test_loader, model, device, test_score_path, test_trial_path
        )
        calculate_EER(
            cm_scores_file=test_score_path,
            output_file=model_tag / "EER.txt"
        )
        print("DONE.")
        sys.exit(0)
        
    # Get optimizer and scheduler
    optim_config["steps_per_epoch"] = len(train_loader)
    optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
    optimizer_swa = SWA(optimizer)
    
    best_val_eer = 1.
    best_test_eer = 100.
    n_swa_update = 0
    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")
    
    # Make directory for metric logging
    metric_path = model_tag / "metrics"
    os.makedirs(metric_path, exist_ok=True)
    
    # Training
    for epoch in range(config["num_epochs"]):
        print("Start training epoch{:03d}".format(epoch))
        
        running_loss = train_epoch(
            train_loader, model, optimizer, device, scheduler, config
        )
        produce_evaluation_file(
            val_loader, model, device, metric_path / "val_score.txt", val_trial_path
        )
        
        val_eer = calculate_EER(
            cm_scores_file=metric_path / "val_score.txt",
            output_file=metric_path / "val_EER_{:03d}epo.txt".format(epoch),
            printout=False
        )
        print("DONE.\nLoss:{:.5f}, val_eer: {:.3f}".format(running_loss, val_eer))
        
        writer.add_scalar("loss", running_loss, epoch)
        writer.add_scalar("val_eer", val_eer, epoch)
        
        if best_val_eer >= val_eer:
            print("Best model found at epoch: ", epoch)
            best_val_eer = val_eer
            torch.save(
                model.state_dict(),
                model_save_path / "epoch_{}_{:03.3f}.pth".format(epoch, val_eer)
            )
            
            # Do evaluation whenever best model is renewed
            if str_to_bool(config["eval_all_best"]):
                produce_evaluation_file(
                    test_loader, model, device, test_score_path, test_trial_path
                )
                
                test_eer = calculate_EER(
                    cm_scores_file=test_score_path,
                    output_file=metric_path / "EER_{:03d}epo.txt".format(epoch)
                )
                
                log_text = "epoch{:03d}, ".format(epoch)
                if test_eer < best_test_eer:
                    log_text += "Best EER: {:.4f}%".format(test_eer)
                    best_test_eer = test_eer
                if len(log_text) > 0:
                    print(log_text)
                    f_log.write(log_text + "\n")
            
            print("Saving epoch {} for SWA".format(epoch))
            optimizer_swa.update_swa()
            n_swa_update += 1
        
        writer.add_scalar("best_val_eer", best_val_eer, epoch)
        
    print("Start final evaluation")
    epoch += 1
    if n_swa_update > 0:
        optimizer_swa.swap_swa_sgd()
        optimizer_swa.bn_update(train_loader, model, device=device)
        
    produce_evaluation_file(
        test_loader, model, device, test_score_path, test_trial_path
    )
    test_eer = calculate_EER(
        cm_scores_file=test_score_path,
        output_file=model_tag / "EER.txt"
    )
    
    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")
    f_log.write("EER: {:.3f}".format(test_eer))
    f_log.close()
    
    torch.save(model.state_dict(), model_save_path / "swa.pth")
    
    if test_eer <= best_test_eer:
        best_test_eer = test_eer
    print("Exp FIN. EER: {:.3f}".format(best_test_eer))


def get_model(model_config: Dict, device: torch.device):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))

    return model

def get_loader(database_path: str, seed: int,config: dict) -> List[torch.utils.data.DataLoader]:
    train_database_path = database_path / "IDASVspoofing_train_joined/"
    val_database_path = database_path / "IDASVspoofing_val_joined/"
    test_database_path = database_path / "IDASVspoofing_test_joined/"
    
    train_list_path = database_path / "IDASVspoofing.cm.train.txt.joined"
    val_trial_path = database_path / "IDASVspoofing.cm.val.txt.joined"
    test_trial_path = database_path / "IDASVspoofing.cm.test.txt.joined"
    
    d_label_train, file_train = genSpoof_list(
        dir_meta=train_list_path, 
        is_train=True, 
        is_eval=False
    )
    print("No. training files: ", len(file_train))
    train_set = Dataset_IDASVspoofing_train(
        list_IDs=file_train,
        labels=d_label_train,
        base_dir=train_database_path
    )
    gen = torch.Generator()
    gen.manual_seed(seed)
    train_loader = DataLoader(
        train_set,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=gen
    )
    
    _, file_val = genSpoof_list(
        dir_meta=val_trial_path,
        is_train=False,
        is_eval=False
    )
    print("No. validation files: ", len(file_val))
    val_set = Dataset_IDASVspoofing_val(
        list_IDs=file_val,
        base_dir=val_database_path
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )
    
    file_test = genSpoof_list(
        dir_meta = test_trial_path,
        is_train=False,
        is_eval=True
    )
    test_set = Dataset_IDASVspoofing_val(
        list_IDs=file_test,
        base_dir=test_database_path
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def produce_evaluation_file(
    data_loader: DataLoader,
    model,
    device: torch.device,
    save_path: str,
    trial_path: str) -> None:
    """Perform evaluation and save the score to a file"""
    model.eval()
    with open(trial_path, "r") as f_trl:
        trial_lines = f_trl.readlines()
    fname_list = []
    score_list = []
    for batch_x, utt_id in data_loader:
        batch_x = batch_x.to(device)
        with torch.no_grad():
            _, batch_out = model(batch_x)
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())

    assert len(trial_lines) == len(fname_list) == len(score_list)
    with open(save_path, "w") as fh:
        for fn, sco, trl in zip(fname_list, score_list, trial_lines):
            _, utt_id, _, src, key = trl.strip().split(' ')
            assert fn == utt_id
            fh.write("{} {} {} {}\n".format(utt_id, src, key, sco))
    print("Scores saved to {}".format(save_path))


def train_epoch(
    trn_loader: DataLoader,
    model,
    optim: Union[torch.optim.SGD, torch.optim.Adam],
    device: torch.device,
    scheduler: torch.optim.lr_scheduler,
    config: argparse.Namespace):
    """Train the model for one epoch"""
    running_loss = 0
    num_total = 0.0
    ii = 0
    model.train()

    # set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    for batch_x, batch_y in trn_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        _, batch_out = model(batch_x, Freq_aug=str_to_bool(config["freq_aug"]))
        batch_loss = criterion(batch_out, batch_y)
        running_loss += batch_loss.item() * batch_size
        optim.zero_grad()
        batch_loss.backward()
        optim.step()

        if config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
            scheduler.step()
        elif scheduler is None:
            pass
        else:
            raise ValueError("scheduler error, got:{}".format(scheduler))

    running_loss /= num_total
    return running_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof detection system")
    parser.add_argument("--config",
                        dest="config",
                        type=str,
                        help="configuration file",
                        required=True)
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        help="output directory for results",
        default="./exp_result",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="random seed (default: 1234)")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="when this flag is given, evaluates given model and exit")
    parser.add_argument("--comment",
                        type=str,
                        default=None,
                        help="comment to describe the saved model")
    parser.add_argument("--eval_model_weights",
                        type=str,
                        default=None,
                        help="directory to the model weight file (can be also given in the config file)")
    main(parser.parse_args())