from __future__ import annotations
import math, shlex, sys, time, shutil
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader
from .runtime import save_checkpoint
from .runtime import config_sha256
from .data import (CombinedNaturalImageDataset,ImageStore,
    build_combined_response_table,dataset_manifest_fingerprint,session_mapping_dict)
from .evaluate import collect_predictions,compute_natural_image_metrics,save_evaluation
from .analysis import training_loss
from .analysis import per_neuron_masked_pearson,summarize_correlations
from .model import SinzV1Model
from .orchestrator import RunRegistry
from .data import PreprocessConfig
from .runtime import EpochTiming,batch_bar,epoch_bar,format_seconds,gpu_memory_gb
from .runtime import capture_rng_state,restore_rng_state,seed_everything,seed_worker
from .data import create_split,load_split,save_split

def _optimizer(model,cfg,lr):
    o=cfg["training"]["optimizer"]
    name = str(o.get("name", "AdamW")).lower()
    optimizer_cls = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}.get(name)
    if optimizer_cls is None:
        raise ValueError(f"Unsupported optimizer.name={o.get('name')!r}")
    return optimizer_cls([p for p in model.parameters() if p.requires_grad],
        lr=float(lr),betas=tuple(float(x) for x in o["betas"]),
        eps=float(o["eps"]),weight_decay=float(o["weight_decay"]))

def _set_lr(optimizer,lr):
    for g in optimizer.param_groups: g["lr"]=float(lr)

def _initialize_readout_bias_from_targets(model, response, mask):
    y = np.asarray(response, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    counts = m.sum(axis=0)
    sums = np.where(m, y, 0.0).sum(axis=0)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    readout = model.readout
    if hasattr(readout, "heads"):
        start = 0
        with torch.no_grad():
            for animal in ("F", "N"):
                head = readout.heads[animal]
                width = head.outdims
                if head.bias is not None:
                    head.bias.copy_(torch.as_tensor(means[start:start+width], dtype=head.bias.dtype, device=head.bias.device))
                start += width
        if start != len(means):
            raise ValueError("Readout means do not match combined neuron axis")
    else:
        bias = getattr(readout, "bias", None)
        if bias is not None:
            with torch.no_grad():
                bias.copy_(torch.as_tensor(means, dtype=bias.dtype, device=bias.device))

def _scheduler(optimizer,cfg,phase="joint_finetune"):
    sc=cfg["training"].get("scheduler",{})
    if not sc.get("enabled",True) or (phase == "readout_only" and not sc.get("enable_phase1",False)):
        return None
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=str(sc.get("mode","max")),
        factor=float(sc.get("factor",0.3)),
        patience=int(sc.get("patience",5)),
        threshold=float(sc.get("threshold",1e-4)),
        threshold_mode=str(sc.get("threshold_mode","abs")),
        min_lr=float(sc.get("min_lr",1e-6)),
    )

def _configure_model_for_phase(model,phase,cfg):
    if phase=="readout_only":
        model.set_core_trainable(False)
        model.train()
        if cfg["training"]["phase1_freeze"]["freeze_batchnorm_running_stats"]:
            model.core.eval()
        else:
            model.core.train()
        model.readout.train()
        return
    if phase=="joint_finetune":
        model.set_core_trainable(True)
        model.train()
        model.core.train()
        model.readout.train()
        return
    raise ValueError(phase)

def _phase_for_epoch(epoch,cfg):
    return "readout_only" if epoch<int(cfg["training"]["readout_only_epochs"]) else "joint_finetune"

def _make_loader(dataset,cfg,shuffle,seed):
    lc=cfg["data"]["loader"]; gen=torch.Generator(); gen.manual_seed(int(seed))
    nw=int(lc["num_workers"])
    return DataLoader(dataset,batch_size=int(lc["batch_size"]),shuffle=shuffle,
        num_workers=nw,pin_memory=bool(lc["pin_memory"]),generator=gen,
        worker_init_fn=seed_worker,persistent_workers=nw>0)

def _prune_uncommitted_checkpoints(run_dir, committed_next_epoch):
    """Discard auxiliary checkpoints from epochs newer than latest.pt."""
    checkpoint_dir=Path(run_dir)/"checkpoints"
    candidates=[checkpoint_dir/"best.pt",checkpoint_dir/"phase_best.pt"]
    candidates.extend(checkpoint_dir.glob("epoch_*.pt"))
    candidates.extend(checkpoint_dir.glob("phase1_final_epoch_*.pt"))
    for path in candidates:
        if not path.exists():
            continue
        try:
            payload=torch.load(path,map_location="cpu")
            epoch=int(payload.get("epoch",-1))
        except Exception:
            continue
        if epoch >= int(committed_next_epoch):
            path.unlink()

def _optimizer_for_resume(model,cfg,next_epoch,checkpoint_phase=None):
    p1=int(cfg["training"]["readout_only_epochs"])
    # latest.pt at next_epoch==p1 was written after epoch p1-1, therefore it
    # still contains the phase-1 optimizer topology.
    phase = checkpoint_phase or ("readout_only" if next_epoch <= p1 else "joint_finetune")
    if phase == "readout_only":
        _configure_model_for_phase(model,phase,cfg)
        return _optimizer(model,cfg,cfg["training"]["phase1_lr"])
    if cfg["training"]["optimizer"]["reset_at_phase2"]:
        _configure_model_for_phase(model,phase,cfg)
        return _optimizer(model,cfg,cfg["training"]["phase2_lr"])
    # With reset_at_phase2=false, the historical optimizer retains the
    # readout parameter group and appends a second group for the core.
    _configure_model_for_phase(model,"readout_only",cfg)
    opt = _optimizer(model,cfg,cfg["training"]["phase1_lr"])
    _configure_model_for_phase(model,phase,cfg)
    existing = {id(p) for g in opt.param_groups for p in g["params"]}
    new = [p for p in model.parameters() if p.requires_grad and id(p) not in existing]
    if new:
        opt.add_param_group({"params": new, "lr": float(cfg["training"]["phase2_lr"])})
    _set_lr(opt,cfg["training"]["phase2_lr"])
    return opt

@torch.no_grad()
def _validation_corr(model,loader,device,limit_batches=None):
    model.eval(); pp=[]; rr=[]; mm=[]
    for i,b in enumerate(loader):
        if limit_batches is not None and i>=int(limit_batches): break
        pp.append(model(b["image"].to(device,non_blocking=True),sample=False).cpu().numpy())
        rr.append(b["response"].numpy()); mm.append(b["mask"].numpy())
    corr=per_neuron_masked_pearson(np.concatenate(pp),np.concatenate(rr),np.concatenate(mm))
    return corr,summarize_correlations(corr)

def _header(run_id,candidate_id,member,cfg,tr,va,te,device,resume):
    print("="*88); print("RUN      ",run_id); print("CANDIDATE",candidate_id)
    print(f"MEMBER    {member+1}/{cfg['training']['ensemble_members']}")
    print("DEVICE   ",device,("| "+torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "")
    print(f"DATA      {len(tr.session_info)} sessions | {tr.responses.shape[1]} neurons")
    print(f"TRAIN     {len(tr.image_ids)} rows"); print(f"VAL       {len(va.image_ids)} rows")
    print(f"TEST      {len(np.unique(te.image_ids))} unique images | {len(te.image_ids)} repeat rows")
    print(f"PHASE 1   1-{cfg['training']['readout_only_epochs']} | readout | lr={cfg['training']['phase1_lr']}")
    print(f"PHASE 2   {cfg['training']['readout_only_epochs']+1}-{cfg['training']['total_epochs']} | core+readout | lr={cfg['training']['phase2_lr']}")
    print("RESUME   ","yes" if resume else "no"); print("="*88)

def train_member(cfg:Dict[str,Any],member_index:int,candidate_id:Optional[str]=None,notes=None,
                 resume_run_id:Optional[str]=None,dry_run=False,limit_train_batches=None,
                 limit_val_batches=None,evaluate_test=True)->str:
    member_index=int(member_index); seed=int(cfg["training"]["member_seeds"][member_index])
    seeds=seed_everything(seed,cfg); dc=cfg["data"]
    is_tvsd = dc.get("kind") == "tvsd"
    if is_tvsd:
        from .tvsd_data import TVSDTrainingBundle
        bundle = TVSDTrainingBundle(cfg)
        full, test = bundle.full, bundle.test
        normalization_stats = bundle.normalization_stats
        dfp = bundle.fingerprint
        mapping = bundle.mapping
        print("TVSD frozen partition and grayscale normalization:", normalization_stats)
    else:
        rc=dc["response"]
        pc_base=PreprocessConfig.from_dict(dc["preprocessing"])
        store=ImageStore(str(Path(dc["root"])/dc["images_subdir"]),dc["image_filename_pattern"],dc["image_id_offset"])
        print("Preparing transformed image bank and normalization statistics ...")
        pc, normalization_stats = store.prepare(pc_base, dc["preprocessing"].get("normalization", {}))
        print(f"NORMALIZE  N={normalization_stats['n_images']} | mean={pc.mean:.8f} | std={pc.std:.8f}")
        full=build_combined_response_table(dc["root"],"train",rc["time_bins"],rc["average_bins"],rc["expected_total_neurons"])
        test=build_combined_response_table(dc["root"],"test",rc["time_bins"],rc["average_bins"],rc["expected_total_neurons"])
        dfp=dataset_manifest_fingerprint(dc["root"],cfg["reproducibility"]["full_dataset_sha256"])
        mapping=session_mapping_dict(full.session_info)
    reg=RunRegistry(cfg["project"]["sqlite_path"]); reg.initialize()
    command=shlex.join(sys.argv)

    if resume_run_id is None:
        if candidate_id is None:
            raise ValueError("candidate_id is required for a new run")
        candidate_id=str(candidate_id)
        if is_tvsd:
            split=bundle.split
        else:
            sc=dc["split"]; split=create_split(full.image_ids,float(sc["train_fraction"]),int(sc["seed"]))
        run=reg.start_run(candidate_id,member_index,command,seeds,dfp,split.sha256,mapping,cfg=cfg,notes=notes)
        run_id=run["run_id"]; run_dir=Path(run["run_dir"]); start_epoch=0; payload=None
        save_split(split,str(run_dir/"split.json")); torch.save(capture_rng_state(),run_dir/"rng_start.pt")
        reg.log_artifact(run_id,"split",str(run_dir/"split.json"))
        import json
        with open(run_dir/"normalization.json","w") as f:
            json.dump(normalization_stats,f,indent=2,sort_keys=True)
        reg.log_artifact(run_id,"normalization",str(run_dir/"normalization.json"))
    else:
        run_id=str(resume_run_id); run=reg.get_run(run_id); run_dir=Path(run["run_dir"])
        stored_candidate_id=str(run["candidate_id"])
        if candidate_id is not None and str(candidate_id)!=stored_candidate_id:
            raise ValueError("candidate mismatch")
        candidate_id=stored_candidate_id
        if int(run["member_index"])!=member_index: raise ValueError("member mismatch")
        if run["config_sha256"]!=config_sha256(cfg): raise ValueError("config changed: resume refused")
        if run["dataset_fingerprint"]!=dfp: raise ValueError("dataset changed: resume refused")
        split=load_split(str(run_dir/"split.json"))
        if is_tvsd and split.sha256 != bundle.split.sha256:
            raise ValueError("Frozen TVSD image partition changed: resume refused")
        latest=run_dir/"checkpoints/latest.pt"
        if not latest.exists(): raise FileNotFoundError(f"no recovery checkpoint: {latest}")
        payload=torch.load(latest,map_location="cpu"); start_epoch=int(payload["next_epoch"])
        if payload["split_sha256"]!=split.sha256: raise ValueError("split mismatch")
        _prune_uncommitted_checkpoints(run_dir,start_epoch)
        reg.mark_running(run_id,command)

    tr=full.subset_by_image_ids(split.train_image_ids); va=full.subset_by_image_ids(split.val_image_ids)
    if is_tvsd:
        trds=bundle.dataset("train")
        vads=bundle.dataset("val")
        teds=bundle.dataset("test")
    else:
        trds=CombinedNaturalImageDataset(tr,store,pc)
        vads=CombinedNaturalImageDataset(va,store,pc)
        teds=CombinedNaturalImageDataset(test,store,pc)
    vloader=_make_loader(vads,cfg,False,seeds["dataloader"]+1000000)
    tloader=_make_loader(teds,cfg,False,seeds["dataloader"]+2000000)
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _header(run_id,candidate_id,member_index,cfg,tr,va,test,device,payload is not None)

    total=int(cfg["training"]["total_epochs"]); p1=int(cfg["training"]["readout_only_epochs"])
    if dry_run:
        total=min(total,start_epoch+1); limit_train_batches=limit_train_batches or 2; limit_val_batches=limit_val_batches or 2
        print("DRY RUN   one bounded epoch; final test evaluation disabled")
    if is_tvsd:
        from .model import TVSDV1Model
        model=TVSDV1Model(cfg).to(device)
    else:
        model=SinzV1Model(cfg).to(device)
    if payload is None:
        _initialize_readout_bias_from_targets(model, tr.responses, tr.mask)
        print("INIT     readout bias = observed training-response mean")
        _configure_model_for_phase(model,"readout_only",cfg)
        opt=_optimizer(model,cfg,cfg["training"]["phase1_lr"])
        scheduler=_scheduler(opt,cfg,"readout_only")
        best=-math.inf; best_epoch=None; phase_best=-math.inf
        step=0; bad_epochs=0; lr_decays=0
    else:
        opt=_optimizer_for_resume(model,cfg,start_epoch,payload.get("phase"))
        model.load_state_dict(payload["model_state_dict"]); opt.load_state_dict(payload["optimizer_state_dict"])
        scheduler=_scheduler(opt,cfg,payload.get("phase","joint_finetune"))
        if scheduler is not None and payload.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(payload["scheduler_state_dict"])
        restore_rng_state(payload["rng_state"]); best=payload.get("best_score")
        best=-math.inf if best is None else float(best); best_epoch=payload.get("best_epoch")
        saved_metrics=payload.get("metrics",{})
        step=int(saved_metrics.get("global_step",0))
        bad_epochs=int(saved_metrics.get("bad_epochs",0))
        lr_decays=int(saved_metrics.get("lr_decays",0))
        phase_best=float(saved_metrics.get("phase_best",best if payload.get("phase")=="readout_only" else -math.inf))
        print(f"RECOVERY  epoch {start_epoch+1} from {run_dir/'checkpoints/latest.pt'}")

    gamma=float(cfg["training"]["loss"]["gamma_readout"]); eps=float(cfg["training"]["loss"]["eps"])
    bestp=run_dir/"checkpoints/best.pt"; phase_bestp=run_dir/"checkpoints/phase_best.pt"
    latestp=run_dir/"checkpoints/latest.pt"; lastp=run_dir/"checkpoints/last.pt"
    timing=EpochTiming(cfg["training"]["progress"]["eta_window_epochs"]); wall=time.monotonic()
    try:
        outer=epoch_bar(range(start_epoch,total),total=total,initial=start_epoch,
                        disable=not cfg["training"]["progress"]["enabled"])
        for epoch in outer:
            estart=time.monotonic(); phase=_phase_for_epoch(epoch,cfg)
            if epoch==p1:
                _configure_model_for_phase(model,"joint_finetune",cfg)
                if cfg["training"]["optimizer"]["reset_at_phase2"]:
                    opt=_optimizer(model,cfg,cfg["training"]["phase2_lr"])
                else:
                    existing={id(p) for g in opt.param_groups for p in g["params"]}
                    new=[p for p in model.parameters() if p.requires_grad and id(p) not in existing]
                    if new: opt.add_param_group({"params":new,"lr":float(cfg["training"]["phase2_lr"])})
                    _set_lr(opt,cfg["training"]["phase2_lr"])
                if cfg["training"].get("scheduler",{}).get("reset_at_phase2",True):
                    scheduler=_scheduler(opt,cfg,"joint_finetune")
                phase_best=-math.inf; bad_epochs=0; lr_decays=0
                phase_bestp.unlink(missing_ok=True)
            _configure_model_for_phase(model,phase,cfg)
            loader=_make_loader(trds,cfg,True,seeds["dataloader"]+epoch)
            ls=ps=0.0; nb=0
            inner=batch_bar(loader,epoch,int(cfg["training"]["total_epochs"]),phase,
                            disable=not cfg["training"]["progress"]["enabled"])
            for bi,b in enumerate(inner):
                if limit_train_batches is not None and bi>=int(limit_train_batches): break
                x=b["image"].to(device,non_blocking=True); y=b["response"].to(device,non_blocking=True); m=b["mask"].to(device,non_blocking=True)
                opt.zero_grad(set_to_none=True); pred=model(x,sample=None)
                z=training_loss(model,pred,y,m,gamma,
                                dataset_size=len(trds),
                                batch_size=int(y.shape[0]),
                                eps=eps)
                if not torch.isfinite(z["loss"]): raise FloatingPointError(f"non-finite loss e{epoch} b{bi}")
                z["loss"].backward(); opt.step(); lv=float(z["loss"].detach().cpu()); pv=float(z["poisson"].detach().cpu())
                ls+=lv; ps+=pv; nb+=1; step+=1
                inner.set_postfix(loss=f"{lv:.4f}",pois=f"{pv:.4f}",lr=f"{opt.param_groups[0]['lr']:.1e}",gpu=gpu_memory_gb())
                hb=int(cfg["training"]["progress"]["heartbeat_every_batches"])
                if hb>0 and step%hb==0:
                    reg.update_progress(run_id,phase,epoch,bi+1,len(loader),step,
                        (epoch+(bi+1)/max(len(loader),1))/int(cfg["training"]["total_epochs"]),
                        lv,pv,float(opt.param_groups[0]["lr"]),None if best==-math.inf else best,best_epoch,
                        time.monotonic()-wall,timing.eta(int(cfg["training"]["total_epochs"])-epoch-1),
                        str(latestp) if latestp.exists() else None)
            inner.close()
            if nb==0: raise RuntimeError("zero training batches")
            _,vs=_validation_corr(model,vloader,device,limit_val_batches)
            ml,mp,lr=ls/nb,ps/nb,float(opt.param_groups[0]["lr"])
            reg.log_epoch(run_id,epoch,phase,ml,mp,vs["mean"],vs["median"],lr)
            phase_improved=np.isfinite(vs["mean"]) and vs["mean"]>phase_best
            global_improved=np.isfinite(vs["mean"]) and vs["mean"]>best
            if phase_improved:
                phase_best=float(vs["mean"]); bad_epochs=0
            else:
                bad_epochs += 1
            if global_improved:
                best=float(vs["mean"]); best_epoch=epoch
            old_lr=float(opt.param_groups[0]["lr"])
            if scheduler is not None:
                scheduler.step(vs["mean"])
                new_lr=float(opt.param_groups[0]["lr"])
                if new_lr < old_lr - 1e-15:
                    lr_decays += 1
                    bad_epochs=0
            else:
                new_lr=old_lr
            metric_state={"val_corr_mean":vs["mean"],"val_corr_median":vs["median"],
                          "train_loss":ml,"global_step":step,"bad_epochs":bad_epochs,
                          "lr_decays":lr_decays,"phase_best":phase_best}
            if phase_improved:
                save_checkpoint(str(phase_bestp),model,opt,epoch,phase,run_id,cfg,split.sha256,capture_rng_state(),
                    metric_state,phase_best,epoch,scheduler=scheduler)
            if global_improved:
                h=save_checkpoint(str(bestp),model,opt,epoch,phase,run_id,cfg,split.sha256,capture_rng_state(),
                    metric_state,best,best_epoch,scheduler=scheduler)
                reg.log_checkpoint(run_id,epoch,phase,str(bestp),h["file_sha256"],h["parameter_sha256"],True)
            sc=cfg["training"].get("scheduler",{})
            stop_after_decay = (
                phase == "joint_finetune"
                and new_lr < old_lr - 1e-15
                and lr_decays > int(sc.get("max_lr_decays",3))
            )
            decay_restore = new_lr < old_lr - 1e-15 and phase_bestp.exists()
            if decay_restore:
                phase_state=torch.load(phase_bestp,map_location=device)
                model.load_state_dict(phase_state["model_state_dict"])
                opt.load_state_dict(phase_state["optimizer_state_dict"])
                _set_lr(opt,new_lr)
            h=save_checkpoint(str(latestp),model,opt,epoch,phase,run_id,cfg,split.sha256,capture_rng_state(),
                metric_state,best,best_epoch,scheduler=scheduler)
            reg.log_checkpoint(run_id,epoch,phase,str(latestp),h["file_sha256"],h["parameter_sha256"],False)
            every=int(cfg["training"]["checkpoint"]["archive_every_epochs"])
            if epoch==p1-1 or (every>0 and (epoch+1)%every==0):
                ap=run_dir/"checkpoints"/(f"phase1_final_epoch_{epoch+1:03d}.pt" if epoch==p1-1 else f"epoch_{epoch+1:03d}.pt")
                shutil.copy2(latestp,ap); reg.log_artifact(run_id,"checkpoint_archive",str(ap))
            sec=time.monotonic()-estart; timing.add(sec); eta=timing.eta(int(cfg["training"]["total_epochs"])-epoch-1)
            reg.update_progress(run_id,phase,epoch,nb,len(loader),step,(epoch+1)/int(cfg["training"]["total_epochs"]),
                ml,mp,lr,None if best==-math.inf else best,best_epoch,time.monotonic()-wall,eta,str(latestp))
            outer.set_postfix(val_r=f"{vs['mean']:.4f}",best="--" if best_epoch is None else f"{best:.4f}@{best_epoch+1}",
                              lr=f"{new_lr:.1e}",decays=lr_decays,
                              epoch_t=format_seconds(sec),ETA=format_seconds(eta))
            if stop_after_decay:
                print(f"EARLY STOP  learning-rate decay limit exceeded ({lr_decays})")
                break
            payload=None
        outer.close()
        if dry_run:
            reg.finish_run(run_id,"dry_run_completed"); print("DRY RUN PASS"); return run_id
        shutil.copy2(latestp,lastp); reg.log_artifact(run_id,"final_checkpoint",str(lastp))
        if not evaluate_test:
            reg.finish_run(run_id,"completed")
            reg.export_results(cfg["project"]["result_sheet_path"])
            return run_id
        cp=bestp if cfg["training"]["checkpoint"]["use_best_for_test"] else lastp
        state=torch.load(cp,map_location=device); model.load_state_dict(state["model_state_dict"])
        collected=collect_predictions(model,tloader,device); metrics=compute_natural_image_metrics(collected)
        edir=run_dir/"evaluation"; save_evaluation(str(edir),run_id,metrics,collected,cfg["evaluation"]["save_predictions"])
        for name in ("corr_to_average","single_trial_corr"):
            vec=metrics[name]; sk="corr_to_average_summary" if name=="corr_to_average" else "single_trial_summary"
            sm=metrics[sk]; reg.log_metric(run_id,"test",name,"mean",sm["mean"]); reg.log_metric(run_id,"test",name,"median",sm["median"])
            for ni,v in enumerate(vec):
                if np.isfinite(v): reg.log_metric(run_id,"test",name,"neuron",float(v),ni)
        reg.log_artifact(run_id,"evaluation_summary",str(edir/"metrics.json"))
        reg.log_artifact(run_id,"per_neuron_metrics",str(edir/"per_neuron.csv"))
        torch.save(capture_rng_state(),run_dir/"rng_end.pt"); reg.finish_run(run_id,"completed")
        reg.export_results(cfg["project"]["result_sheet_path"]); return run_id
    except KeyboardInterrupt:
        reg.finish_run(run_id,"interrupted","partial epoch discarded; resume from latest completed epoch")
        print("\nInterrupted: partial epoch discarded. Resume from latest.pt."); raise
    except Exception as exc:
        reg.finish_run(run_id,"failed",repr(exc)); raise
