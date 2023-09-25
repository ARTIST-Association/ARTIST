from yacs.config import CfgNode
import os
import training
from heliostat_models import Heliostat

def check_config_file_on_common_mistakes(cfg: CfgNode) -> None:
    print("Loaded Switches:")
    print(f"Heliostat shape: {cfg.H.SHAPE}")
    print(f"Solar distribution: {cfg.AC.SUN.DISTRIBUTION}")
    print(f"Scheduler: {cfg.TRAIN.SCHEDULER.NAME}")
    print(f"Optimizer: {cfg.TRAIN.OPTIMIZER.NAME}")
    print(f"Loss: {cfg.TRAIN.LOSS.NAME}")

    warnings_found = False
    if cfg.TRAIN.LOSS.USE_L1_WEIGHT_DECAY:
        if not cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY == 0:
            warnings_found = True
            print("WARNING: Do you really want to use L2 and L1 weight decay?")
    if cfg.TRAIN.SCHEDULER.NAME.lower() == "cyclic":
        if not cfg.TRAIN.SCHEDULER.CYCLIC.BASE_LR == cfg.TRAIN.OPTIMIZER.LR:
            warnings_found = True
            print(
                "WARNING: Cyclic base LR and optimizer LR should be the same")
    if not cfg.CP_PATH == "":
        print("continue without loading...")
        if not os.path.isfile(os.path.expanduser(cfg.CP_PATH)):
            warnings_found = True
            print(
                "WARNING: Checkpoint path not found; "
                "continuing without loading..."
            )
    if (
            cfg.LOAD_OPTIMIZER_STATE
            and not os.path.isfile(training.get_opt_cp_path(cfg.CP_PATH))
    ):
        warnings_found = True
        print(
            "WARNING: Optimizer checkpoint not found; "
            "continuing without loading..."
        )

    nurbs_focus_point = cfg.NURBS.FACETS.CANTING.FOCUS_POINT
    heliostat_cfg = Heliostat.select_heliostat_builder(cfg.H)[1]
    heliostat_focus_point = heliostat_cfg.FACETS.CANTING.FOCUS_POINT
    if (
            nurbs_focus_point != 'inherit'
            and nurbs_focus_point != heliostat_focus_point
    ):
        warnings_found = True
        print(
            "WARNING: Focus points of target and trained heliostat "
            "do not match!"
        )

    if not warnings_found:
        print("No warnings found. Good Luck!")
        print("=============================")