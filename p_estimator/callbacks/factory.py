from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from p_estimator.callbacks.image_log import LogImageCallback
from p_estimator.callbacks.metric import MetricCallback
from p_estimator.callbacks.image_q_value_vs_p import LogQvsPCallback

def create_callbacks(cfg:dict):
    callbacks = []
    for callback in cfg["TRAINER"]["callbacks"]:
        callbacks.append(get_callback(callback, cfg))
    return callbacks

def get_callback(callback:str, cfg:dict):
    if callback == "ModelCheckpoint":
        return ModelCheckpoint(
            save_top_k=cfg["TRAINER"]["callbacks"]["ModelCheckpoint"]["save_top_k"],
            monitor=cfg["TRAINER"]["callbacks"]["ModelCheckpoint"]["monitor"],
            mode=cfg["TRAINER"]["callbacks"]["ModelCheckpoint"]["mode"],
            dirpath=cfg.GENERAL_PARMAS.output_path
        )
    elif callback == "LogImageCallback":
        return LogImageCallback()
    elif callback == "LearningRateMonitor":
        return LearningRateMonitor(
            logging_interval=cfg["TRAINER"]["callbacks"]["LearningRateMonitor"]["logging_interval"]
            )
    elif callback == "MetricCallback":
        return MetricCallback()
    elif callback == "LogQvsPCallback":
        return LogQvsPCallback()
    else:
        raise ValueError(f"callback {callback} not supported")