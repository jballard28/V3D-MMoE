import sys
sys.path.append('..')
from utils import seed_everything, setup_logger
sys.path.append('../models/fastmoe')
sys.path.append('../models/fastmoe/fmoe')
sys.path.append('../models')
sys.path.append('../data')
import os
import torch
import numpy as np
import argparse
import random
import math
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, mean_squared_error, r2_score
from copy import deepcopy
from tqdm import trange
from models import FlexMoE
from data import load_and_preprocess_data, create_loaders
from loss.cmps_loss import Loss
from loss.infonce_loss import InfoNCE
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="os.fork()")

# Utility function to convert string to bool
def str2bool(s):
    if s not in {'False', 'True', 'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return (s == 'True') or (s == 'true')

# Parse input arguments
def parse_args():
    parser = argparse.ArgumentParser(description='FlexMoE')
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--runparallel', type=str2bool, default=False)
    parser.add_argument('--run', type=int, required=True)
    parser.add_argument('--data', type=str, default='adni')
    parser.add_argument('--modality', type=str, default='IGCB') # I G C B for ADNI, L N C for MIMIC
    parser.add_argument('--initial_filling', type=str, default='mean') # None mean
    parser.add_argument('--train_epochs', type=int, default=50)
    parser.add_argument('--warm_up_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--top_k', type=int, default=4) # Number of Routers
    parser.add_argument('--num_patches', type=int, default=16) # Number of Patches for Input Token
    parser.add_argument('--num_experts', type=int, default=16) # Number of Experts
    parser.add_argument('--num_routers', type=int, default=1) # Number of Routers
    parser.add_argument('--num_layers_enc', type=int, default=1) # Number of MLP layers for encoders
    parser.add_argument('--num_layers_fus', type=int, default=1) # Number of MLP layers for fusion model
    parser.add_argument('--num_layers_pred', type=int, default=1) # Number of MLP layers for prediction head
    parser.add_argument('--num_heads', type=int, default=4) # Number of heads
    parser.add_argument('--num_workers', type=int, default=4) # Number of workers for DataLoader
    parser.add_argument('--pin_memory', type=str2bool, default=True) # Pin memory in DataLoader
    parser.add_argument('--use_common_ids', type=str2bool, default=False) # Use common ids across modalities    
    parser.add_argument('--dropout', type=float, default=0.5) # Number of Routers
    parser.add_argument('--gate_loss_weight', type=float, default=1e-2)
    parser.add_argument('--align_loss_name', type=str, default='infonce_patchsamp')
    parser.add_argument('--align_loss_weight', type=float, default=1e-3)
    parser.add_argument('--crossmod_loss_weight', type=float, default=0)
    parser.add_argument('--save', type=str2bool, default=True)
    parser.add_argument('--load_model', type=str2bool, default=False)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_runs', type=int, default=3)

    return parser.parse_known_args()

def run_epoch(args, loader, encoder_dict, modality_dict, missing_embeds, fusion_model, criterion, device, align_lossfun=None, is_training=False, optimizer=None, gate_loss_weight=0.0):
    all_preds = []
    all_labels = []
    all_probs = []
    task_losses = []
    gate_losses = []
    align_losses = []
    crossmod_losses = []
    
    if is_training:
        fusion_model.train()
        for encoder in encoder_dict.values():
            encoder.train()
    else:
        fusion_model.eval()
        for encoder in encoder_dict.values():
            encoder.eval()

    for batch_samples, batch_labels, batch_mcs, batch_observed in loader:
        batch_samples = {k: v.to(device, non_blocking=True) for k, v in batch_samples.items()}
        batch_labels = batch_labels.to(device, non_blocking=True)
        batch_mcs = batch_mcs.to(device, non_blocking=True)
        batch_observed = batch_observed.to(device, non_blocking=True)
        
        mods = [m for (m,s) in batch_samples.items()]
        fusion_input_dict = {}
        cross_enc = {m:{} for m in batch_samples.keys()} # {resulting_mod: {source_mod: embeddings}}
        masks = {}
        for i, (modality, samples) in enumerate(batch_samples.items()):
            mask = batch_observed[:, modality_dict[modality]]
            masks[modality] = mask
            encoded_samples = torch.full((samples.shape[0], args.num_patches, args.hidden_dim), torch.nan, dtype=torch.float).to(device)
            if mask.sum() > 0:
                encoded_samples[mask] = encoder_dict[modality](samples[mask])

            # Cross-encoders are only for mri and fdg
            if modality in ['mri','fdg']:
                for jj in range(len(mods)):
                    if jj != i and mods[jj] in ['mri','fdg']:
                        crossenc_samples = torch.full((samples.shape[0], args.num_patches, args.hidden_dim), torch.nan, dtype=torch.float).to(device)
                        if mask.sum() > 0:
                            othermod = mods[jj]
                            crossenc_samples[mask] = encoder_dict[modality+'2'+othermod](samples[mask])
                        cross_enc[othermod][modality] = crossenc_samples
            fusion_input_dict[modality] = encoded_samples

        # Filling in the embeddings generated by the other modality for missing entries
        for i, modality in enumerate(fusion_input_dict.keys()):
            mask = masks[modality]
            # take mean across all other available modalities
            avail_mod = []
            if (~mask).sum() > 0:
                if modality in ['mri','fdg']:
                    for jj in range(len(mods)):
                        if jj != i and mods[jj] in ['mri','fdg']:
                            sourcemod = mods[jj]
                            avail_mod.append(cross_enc[modality][sourcemod])
                    fill_data = torch.stack(avail_mod, dim=0)
                    fill_data = torch.nanmean(fill_data, dim=0)
                    fusion_input_dict[modality][~mask] = fill_data[~mask]

        fusion_input = [s for (m,s) in fusion_input_dict.items()]
        outputs = fusion_model(*fusion_input, expert_indices=batch_mcs)

        if is_training:
            optimizer.zero_grad()
            task_loss = criterion(outputs, batch_labels)
            task_losses.append(task_loss.item())
            gate_loss = fusion_model.gate_loss()
            gate_losses.append(float(gate_loss))

            # Loss between true embedding and embedding generated by the other modality - only for mri and fdg
            cross_loss = torch.tensor(0)
            complete_samp = torch.where(batch_mcs == 0)
            if len(complete_samp[0]) > 0:
                cross_losses = []
                for i,targetmod in enumerate(mods):
                    if targetmod in ['mri','fdg']:
                        self_embed = fusion_input_dict[targetmod][complete_samp]
                        tempdiffs = []
                        for jj in range(len(mods)):
                            if jj != i and mods[jj] in ['mri','fdg']:
                                sourcemod = mods[jj]
                                cross_embed = cross_enc[targetmod][sourcemod][complete_samp]
                                # calculate difference between self embedding and cross-embedding
                                crossdiff_temp = torch.sqrt(torch.sum(torch.square(self_embed-cross_embed), dim=(1,2)))
                                tempdiffs.append(crossdiff_temp)
                        # stack differences across source modalities and take the mean; add to cross_losses
                        cross_losses.append(torch.mean(torch.stack(tempdiffs,dim=0),dim=0))
    
                # stack cross_losses across target modality and take mean across target modalities
                cross_loss = torch.mean(torch.stack(cross_losses,dim=0),dim=0)
                # take mean across batch samples
                cross_loss = torch.mean(cross_loss)
            crossmod_losses.append(cross_loss.item())

            align_loss = torch.tensor(0)
            # alignment loss
            if args.align_loss_name is not None:
                if args.align_loss_name == 'cmpm':
                    m1 = fusion_input[0]
                    m2 = fusion_input[1]
                    # have to repeat batch_labels for each patch
                    lbls = batch_labels.repeat_interleave(args.num_patches)
                    align_loss = align_lossfun(m1.reshape(-1,m1.shape[2]), m2.reshape(-1,m2.shape[2]), lbls)

                elif 'infonce' in args.align_loss_name:
                    if 'patch' in args.align_loss_name:
                        # infonce_patch: negative examples = all other patches from the other modality
                        m1 = fusion_input[0]
                        m2 = fusion_input[1]
                        patch_loss = align_lossfun(m1.reshape(-1,m1.shape[2]), m2.reshape(-1,m2.shape[2]))

                        if 'sym' in args.align_loss_name: # symmetric version of patch-based infonce
                            patch_loss += align_lossfun(m2.reshape(-1,m2.shape[2]), m1.reshape(-1,m1.shape[2]))
    
                    if 'samp' in args.align_loss_name:
                        # infonce_samp: negative examples = patches from the other modality from all other samples
                        m1 = fusion_input[0]
                        m2 = fusion_input[1]
                        # Generate array of negatives
                        neg1 = [m2[list(range(i))+list(range(i+1,m2.shape[0]))] for i in range(m2.shape[0])] # patches from all but current samp
                        neg1 = [i.reshape((-1,i.shape[-1])) for i in neg1] # reshape so every patch is its own example
                        neg1 = torch.stack(neg1) # stack all examples into its own dimension (along dim 0)
                        neg1 = torch.repeat_interleave(neg1, m1.shape[1], dim=0) # repeat negatives array for each patch in the sample
    
                        samp_loss = align_lossfun(m1.reshape(-1,m1.shape[2]), m2.reshape(-1,m2.shape[2]), neg1)
    
                        if 'sym' in args.align_loss_name:
                            neg2 = [m1[list(range(i))+list(range(i+1,m1.shape[0]))] for i in range(m1.shape[0])] # patches from all but current samp
                            neg2 = [i.reshape((-1,i.shape[-1])) for i in neg2] # reshape so every patch is its own example
                            neg2 = torch.stack(neg2) # stack all examples into its own dimension (along dim 0)
                            neg2 = torch.repeat_interleave(neg2, m2.shape[1], dim=0) # repeat negatives array for each patch in the sample
    
                            samp_loss += align_lossfun(m2.reshape(-1,m2.shape[2]), m1.reshape(-1,m1.shape[2]), neg2)
    
                    if args.align_loss_name in ['infonce_patch', 'infonce_patch_sym']:
                        align_loss = patch_loss
                    elif args.align_loss_name in ['infonce_samp', 'infonce_samp_sym']:
                        align_loss = samp_loss
                    elif args.align_loss_name in ['infonce_patchsamp', 'infonce_patchsamp_sym']:
                        align_loss = patch_loss + samp_loss

            align_losses.append(align_loss.item())

            loss = task_loss + gate_loss_weight * gate_loss + args.align_loss_weight * align_loss + args.crossmod_loss_weight * cross_loss
            loss.backward()
            optimizer.step()
        else:
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            all_probs.extend(torch.nn.functional.softmax(outputs, dim=1).detach().cpu().numpy())

    if is_training:
        return task_losses, gate_losses, align_losses, crossmod_losses
    else:
        return all_preds, all_labels, all_probs


def train_and_evaluate(args, seed, save_path=None):
    seed_everything(seed)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    num_modalities = len(args.modality)

    if args.data == 'adni':
        modality_dict = {'mri':0, 'fdg':1, 'genomic':2, 'clinical':3, 'biospecimen':4}
        args.n_full_modalities = len(modality_dict)
        data_dict, encoder_dict, labels, train_ids, valid_ids, test_ids, common_test_ids, n_labels, input_dims, train_transforms, eval_transforms,  masks, observed_idx_arr, full_modality_index, orig_img_shapes = load_and_preprocess_data(args, modality_dict)
        
    train_loader, train_loader_shuffle, val_loader, test_loader, common_test_loader = create_loaders(data_dict, observed_idx_arr, labels, train_ids, valid_ids, test_ids, common_test_ids, args.batch_size, args.num_workers, args.pin_memory, input_dims, train_transforms, eval_transforms, masks, orig_img_shapes, args.use_common_ids)
    fusion_model = FlexMoE(num_modalities, full_modality_index, args.num_patches, args.hidden_dim, n_labels, args.num_layers_fus, args.num_layers_pred, args.num_experts, args.num_routers, args.top_k, args.num_heads, args.dropout).to(device)
    params = list(fusion_model.parameters()) + [param for encoder in encoder_dict.values() for param in encoder.parameters()]    
    if num_modalities > 1:
        missing_embeds = torch.nn.Parameter(torch.randn((2**num_modalities)-1, args.n_full_modalities, args.num_patches, args.hidden_dim, dtype=torch.float, device=device), requires_grad=True)
        params += [missing_embeds]
    else:
        missing_embeds = None

    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    criterion = torch.nn.CrossEntropyLoss() if args.data == 'adni' else torch.nn.CrossEntropyLoss(torch.tensor([0.25, 0.75]).to(device))

    best_val_acc = 0.0

    if save_path is None:
        for epoch in trange(args.train_epochs):
            fusion_model.train()
            for encoder in encoder_dict.values():
                encoder.train()

            if epoch >= args.warm_up_epochs:
                train_loader_new = train_loader_shuffle
                warm_up_tag = ''
                train_epochs = args.train_epochs
            else:
                # activate modality-based sorting
                train_loader_new = train_loader
                warm_up_tag = 'Warm Up ' 
                train_epochs = args.warm_up_epochs

            ## Training

            # Alignment loss
            if args.align_loss_name == 'cmpm':
                align_lossfun = Loss(num_classes=n_labels, feature_size=args.hidden_dim, resume=False, epsilon=1e-8).to(device)
            elif args.align_loss_name in ['infonce_patch','infonce_samp','infonce_patchsamp','infonce_patch_sym','infonce_samp_sym','infonce_patchsamp_sym']:
                align_lossfun = InfoNCE(negative_mode='paired').to(device)
            else:
                align_lossfun = None

            task_losses, gate_losses, align_losses, crossmod_losses = run_epoch(args, train_loader_new, encoder_dict, modality_dict, missing_embeds, fusion_model, criterion, device, align_lossfun=align_lossfun, is_training=True, optimizer=optimizer, gate_loss_weight=args.gate_loss_weight)
            
            ## Training metrics
            fusion_model.eval()
            for encoder in encoder_dict.values():
                encoder.eval()
            with torch.no_grad():
                train_preds, train_labels, train_probs = run_epoch(args, train_loader, encoder_dict, modality_dict, missing_embeds, fusion_model, criterion, device)

            train_acc = balanced_accuracy_score(train_labels, train_preds)
            train_f1 = f1_score(train_labels, train_preds, average='macro')
            train_auc = roc_auc_score(train_labels, train_probs, multi_class='ovr')

            ## Validation
            fusion_model.eval()
            for encoder in encoder_dict.values():
                encoder.eval()
            with torch.no_grad():
                val_preds, val_labels, val_probs = run_epoch(args, val_loader, encoder_dict, modality_dict, missing_embeds, fusion_model, criterion, device)
            val_acc = balanced_accuracy_score(val_labels, val_preds)
            val_f1 = f1_score(val_labels, val_preds, average='macro')
            val_auc = roc_auc_score(val_labels, val_probs, multi_class='ovr')

            if val_acc > best_val_acc:
                print(f" [(**Best**) {warm_up_tag}Epoch {epoch+1}/{train_epochs}] Val Acc: {val_acc*100:.2f}, Val F1: {val_f1*100:.2f}, Val AUC: {val_auc*100:.2f}")
                best_val_acc = val_acc
                best_val_f1 = val_f1
                best_val_auc = val_auc
                best_model_me = deepcopy(missing_embeds)
                best_model_fus = deepcopy(fusion_model)
                best_model_enc = deepcopy(encoder_dict)
    
            print(f"[Seed {seed}/{args.n_runs-1}] [{warm_up_tag}Epoch {epoch+1}/{train_epochs}] Task Loss: {np.mean(task_losses):.2f}, Router Loss: {np.mean(gate_losses):.2f}, Alignment Loss: {np.mean(align_losses):.2f}, Crossmod Loss: {np.mean(crossmod_losses):.2f} / Val Acc: {val_acc*100:.2f}, Val F1: {val_f1*100:.2f}, Val AUC: {val_auc*100:.2f} / Train Acc: {train_acc*100:.2f}, Train F1: {train_f1*100:.2f}, Train AUC: {train_auc*100:.2f}")

        # Save the best model
        if args.save:
            os.makedirs('../saves', exist_ok=True)
            save_path = f'../saves/seed_{seed}_modality_{args.modality}_npatches_{args.num_patches}_nexperts_{args.num_experts}_nheads_{args.num_heads}_hdim_{args.hidden_dim}_align_{args.align_loss_name}_wd_{args.wd}_lr_{args.lr}_train_epochs_{args.train_epochs}.pth'
            torch.save({
                'missing_embeds': best_model_me,
                'fusion_model': best_model_fus.state_dict(),
                'encoder_dict': {modality: deepcopy(encoder.state_dict()) for modality, encoder in best_model_enc.items()}
            }, save_path)

            print(f"Best model saved to {save_path}")
    
    else:
        best_model_me = missing_embeds
        best_model_fus = fusion_model
        best_model_enc = encoder_dict

        # Load the saved model onto the correct device (GPU or CPU)
        checkpoint = torch.load(save_path, map_location=device)

        # Load the models' states
        best_model_me = checkpoint['missing_embeds']
        best_model_fus.load_state_dict(checkpoint['fusion_model'])
        for modality, encoder in best_model_enc.items():
            encoder.load_state_dict(checkpoint['encoder_dict'][modality])
            encoder.to(device)
            encoder.eval()

        # Move the models to the correct device if necessary
        best_model_me.to(device)
        best_model_fus.to(device)

        ## Validation
        with torch.no_grad():
            val_preds, val_labels, val_probs = run_epoch(args, val_loader, best_model_enc, modality_dict, best_model_me, best_model_fus, criterion, device)
        best_val_acc = balanced_accuracy_score(val_labels, val_preds)
        best_val_f1 = f1_score(val_labels, val_preds, average='macro')
        best_val_auc = roc_auc_score(val_labels, val_probs, multi_class='ovr')

    ## Test
    with torch.no_grad():
        test_preds, test_labels, test_probs = run_epoch(args, test_loader, best_model_enc, modality_dict, best_model_me, best_model_fus, criterion, device)
        common_test_preds, common_test_labels, common_test_probs = run_epoch(args, common_test_loader, best_model_enc, modality_dict, best_model_me, best_model_fus, criterion, device)

    test_acc = balanced_accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds, average='macro')
    test_auc = roc_auc_score(test_labels, test_probs, multi_class='ovr')
    common_test_acc = balanced_accuracy_score(common_test_labels, common_test_preds)
    common_test_f1 = f1_score(common_test_labels, common_test_preds, average='macro')
    common_test_auc = roc_auc_score(common_test_labels, common_test_probs, multi_class='ovr')

    ## Performance on training data
    with torch.no_grad():
        train_preds, train_labels, train_probs = run_epoch(args, train_loader, best_model_enc, modality_dict, best_model_me, best_model_fus, criterion, device)

    train_acc = balanced_accuracy_score(train_labels, train_preds)
    train_f1 = f1_score(train_labels, train_preds, average='macro')
    train_auc = roc_auc_score(train_labels, train_probs, multi_class='ovr')

    return best_val_acc, best_val_f1, best_val_auc, test_acc, test_f1, test_auc, common_test_acc, common_test_f1, common_test_auc, train_acc, train_f1, train_auc


def main():
    args, _ = parse_args()

    if (not args.runparallel) | ((not args.save) & (args.load_model)):
        logger = setup_logger(f'../logs/npatches_{args.num_patches}_nexperts_{args.num_experts}_nheads_{args.num_heads}_hdim_{args.hidden_dim}_align_{args.align_loss_name}_lr_{args.lr}_train_epochs_{args.train_epochs}', f'{args.data}', f'{args.modality}.txt')
        seeds = np.arange(args.n_runs) # [0, 1, 2]
    else:
        seeds = [args.run]
    
    log_summary = "======================================================================================\n"
    
    model_kwargs = {
        "model": 'FlexMoE',
        "modality": args.modality,
        "initial_filling": args.initial_filling,
        "use_common_ids": args.use_common_ids,
        "train_epochs": args.train_epochs,
        "warm_up_epochs": args.warm_up_epochs,
        "num_experts": args.num_experts,
        "num_routers": args.num_routers,
        "top_k": args.top_k,
        "num_layers_enc": args.num_layers_enc,
        "num_layers_fus": args.num_layers_fus,
        "num_layers_pred": args.num_layers_pred,
        "num_heads": args.num_heads,
        "lr": args.lr,
        "wd": args.wd,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "num_patches": args.num_patches,
        "gate_loss_weight": args.gate_loss_weight,
        "align_loss_name": args.align_loss_name,
        "align_loss_weight": args.align_loss_weight,
        "crossmod_loss_weight": args.crossmod_loss_weight,
    }

    log_summary += f"Model configuration: {model_kwargs}\n"

    print('Modality:', args.modality)

    val_accs = []
    val_f1s = []
    val_aucs = []
    test_accs = []
    test_f1s = []
    test_aucs = []
    common_test_accs = []
    common_test_f1s = []
    common_test_aucs = []
    train_accs = []
    train_f1s = []
    train_aucs = []
    for seed in seeds:
        if (not args.save) & (args.load_model):
            save_path = f'../saves/seed_{seed}_modality_{args.modality}_npatches_{args.num_patches}_nexperts_{args.num_experts}_nheads_{args.num_heads}_hdim_{args.hidden_dim}_align_{args.align_loss_name}_wd_{args.wd}_lr_{args.lr}_train_epochs_{args.train_epochs}.pth'
        else:
            save_path = None
        val_acc, val_f1, val_auc, test_acc, test_f1, test_auc, common_test_acc, common_test_f1, common_test_auc, train_acc, train_f1, train_auc = train_and_evaluate(args, seed, save_path=save_path)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)
        val_aucs.append(val_auc)
        test_accs.append(test_acc)
        test_f1s.append(test_f1)
        test_aucs.append(test_auc)
        common_test_accs.append(common_test_acc)
        common_test_f1s.append(common_test_f1)
        common_test_aucs.append(common_test_auc)
        train_accs.append(train_acc)
        train_f1s.append(train_f1)
        train_aucs.append(train_auc)
    
    val_avg_acc = np.mean(val_accs)*100
    val_std_acc = np.std(val_accs)*100
    val_avg_f1 = np.mean(val_f1s)*100
    val_std_f1 = np.std(val_f1s)*100
    val_avg_auc = np.mean(val_aucs)*100
    val_std_auc = np.std(val_aucs)*100
    
    test_avg_acc = np.mean(test_accs)*100
    test_std_acc = np.std(test_accs)*100
    test_avg_f1 = np.mean(test_f1s)*100
    test_std_f1 = np.std(test_f1s)*100
    test_avg_auc = np.mean(test_aucs)*100
    test_std_auc = np.std(test_aucs)*100
    
    common_test_avg_acc = np.mean(common_test_accs)*100
    common_test_std_acc = np.std(common_test_accs)*100
    common_test_avg_f1 = np.mean(common_test_f1s)*100
    common_test_std_f1 = np.std(common_test_f1s)*100
    common_test_avg_auc = np.mean(common_test_aucs)*100
    common_test_std_auc = np.std(common_test_aucs)*100
    
    train_avg_acc = np.mean(train_accs)*100
    train_std_acc = np.std(train_accs)*100
    train_avg_f1 = np.mean(train_f1s)*100
    train_std_f1 = np.std(train_f1s)*100
    train_avg_auc = np.mean(train_aucs)*100
    train_std_auc = np.std(train_aucs)*100
    
    log_summary += f'[Train] Average Accuracy: {train_avg_acc:.2f} ± {train_std_acc:.2f} '
    log_summary += f'[Train] Average F1 Score: {train_avg_f1:.2f} ± {train_std_f1:.2f} '
    log_summary += f'[Train] Average AUC: {train_avg_auc:.2f} ± {train_std_auc:.2f} / '  
    log_summary += f'[Val] Average Accuracy: {val_avg_acc:.2f} ± {val_std_acc:.2f} '
    log_summary += f'[Val] Average F1 Score: {val_avg_f1:.2f} ± {val_std_f1:.2f} '
    log_summary += f'[Val] Average AUC: {val_avg_auc:.2f} ± {val_std_auc:.2f} / '  
    log_summary += f'[Test] Average Accuracy: {test_avg_acc:.2f} ± {test_std_acc:.2f} '
    log_summary += f'[Test] Average F1 Score: {test_avg_f1:.2f} ± {test_std_f1:.2f} '
    log_summary += f'[Test] Average AUC: {test_avg_auc:.2f} ± {test_std_auc:.2f} '  
    log_summary += f'[Common Test] Average Accuracy: {common_test_avg_acc:.2f} ± {common_test_std_acc:.2f} '
    log_summary += f'[Common Test] Average F1 Score: {common_test_avg_f1:.2f} ± {common_test_std_f1:.2f} '
    log_summary += f'[Common Test] Average AUC: {common_test_avg_auc:.2f} ± {common_test_std_auc:.2f} '  
    
    print(model_kwargs)
    print(f'[Train] Average Accuracy: {train_avg_acc:.2f} ± {train_std_acc:.2f} / Average F1 Score: {train_avg_f1:.2f} ± {train_std_f1:.2f} / Average AUC: {train_avg_auc:.2f} ± {train_std_auc:.2f}')
    print(f'[Val] Average Accuracy: {val_avg_acc:.2f} ± {val_std_acc:.2f} / Average F1 Score: {val_avg_f1:.2f} ± {val_std_f1:.2f} / Average AUC: {val_avg_auc:.2f} ± {val_std_auc:.2f}')
    print(f'[Test] Average Accuracy: {test_avg_acc:.2f} ± {test_std_acc:.2f} / Average F1 Score: {test_avg_f1:.2f} ± {test_std_f1:.2f} / Average AUC: {test_avg_auc:.2f} ± {test_std_auc:.2f}')
    print(f'[Common Test] Average Accuracy: {common_test_avg_acc:.2f} ± {common_test_std_acc:.2f} / Average F1 Score: {common_test_avg_f1:.2f} ± {common_test_std_f1:.2f} / Average AUC: {common_test_avg_auc:.2f} ± {common_test_std_auc:.2f}')

    if (not args.runparallel) | ((not args.save) & (args.load_model)):
        logger.info(log_summary)

if __name__ == '__main__':
    main()
