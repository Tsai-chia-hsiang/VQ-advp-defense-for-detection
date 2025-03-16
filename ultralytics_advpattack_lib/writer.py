from pathlib import Path
from typing import Optional
from torch.utils.tensorboard import SummaryWriter

class TrainingMetricsWriter():
    
    def __init__(self, loss_order:list[str], metrics_order:list[str], file:Path, tb_dir:Optional[Path]=None):
        
        file.parent.mkdir(parents=True, exist_ok=True)
        self.fp = open(file, mode='w+')
        self.loss_order = loss_order
        self.metrics_order = metrics_order
        self.tb = self._init_tb(tb_dir=tb_dir)
        csv_bar = f"epoch," + ",".join([_ for _ in self.loss_order])+","+",".join([_ for _ in metrics_order])
        print(csv_bar, file=self.fp, flush=True)
    
    def _init_tb(self, tb_dir:Optional[Path]=None)->None|SummaryWriter:
        if tb_dir is not None:
            log_dir = tb_dir/f"tb"
            log_dir.mkdir(parents=True, exist_ok=True)
            return SummaryWriter(log_dir)
        return None
    
    def __call__(self, epoch:int, loss:dict[str, float], metrics:dict[str, float])->None:
        """
        Write training loss as well as validation metrics in this epoch `e`
        """
        msg = f"{epoch},"+",".join([str(loss[k]) for k in self.loss_order])+"," + ",".join([str(metrics[k]) for k in self.metrics_order])
        print(msg, file=self.fp, flush=True)
        if self.tb is not None:
            self.tb.add_scalars(main_tag='train_loss', tag_scalar_dict=loss, global_step=epoch)
            self.tb.add_scalars(main_tag='val_metrics', tag_scalar_dict=metrics, global_step=epoch)
    
    def close(self):
        self.fp.close()
        if self.tb is not None:
            self.tb.close()