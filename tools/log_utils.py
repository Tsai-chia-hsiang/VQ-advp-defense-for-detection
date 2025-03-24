import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from .cfg_utils import write_json

class MetricsLogger():
    
    def __init__(self, save_dir:Path,  metrics:str, items:tuple[str]=('normal', 'problematic')):
        
        self.items = items
        self.metrics = metrics
        self.values = {k:np.empty(shape=(0), dtype=np.float32) for k in items}
        self.save_dir=save_dir
        self.statics = None
    
    def __call__(self, i:dict[str, np.ndarray]):
        for k in self.items:
            self.values[k] = np.concatenate((self.values[k], np.array(i[k]).reshape(-1)),axis=0)
    
    def end(self, show=False):
        
        self.statics= {k:None for k in self.items}
        
        for k,v in self.values.items():
            self.statics[k] = {
                **{
                'mean':float(np.mean(v)),
                'std':float(np.std(v)),
                'max':float(v.max()),
                'min':float(v.min())
                }, 
                **{
                    f"{q}%": float(np.percentile(v, q)) 
                    for q in list(range(10, 110, 10))+[95]
                }
            }
            if show:
                print(f"{k}:{self.statics[k]}")
    
    def _barchart(self, item_color:dict[str, str], nbins: int = 20, acc=False)->plt.Figure:
        
        # 1. Determine min and max values (rounded to nearest bsize)
        maxv = max([v['95%'] for _, v in self.statics.items()])
        minv = min([v.min() for  _, v in self.values.items()]+[0])

        bins = np.linspace(minv, maxv, nbins)
        w = np.diff(bins)
        fig = plt.figure(dpi=400)
        for label in item_color:
            hist, _ = np.histogram(np.clip(self.values[label], minv, maxv), bins=bins)
            if acc:
                hist = np.cumsum(hist)
            plt.bar(
                bins[:-1], hist, width=w, align='edge', 
                color=item_color[label], label=label, 
                alpha=0.8
            )
        plt.xlabel(self.metrics)
        plt.ylabel("counts" if not acc else "accumulate")
        plt.grid(True)
        if acc:
            plt.gca().invert_xaxis()

        plt.title(f"Comparsion of {self.metrics}")
        plt.legend()  # Show labels
        return fig

    def flush(self, barchart=True, **kwargs):
        
        self.save_dir.mkdir(parents=True, exist_ok=True)

        for k,v in self.values.items():
            np.save(self.save_dir/f"{k}", v)
        
        if self.statics is not None:
            write_json(self.statics, self.save_dir/f"statics.json")
            if barchart:
                g = self._barchart(**kwargs)
                g.savefig(fname=self.save_dir/"count.png")
                plt.close()

                g = self._barchart(acc=True, **kwargs)
                g.savefig(fname=self.save_dir/"accumu.png")
                plt.close()

    def end_n_flush(self, show=False, barchart=True,**kwargs):
        self.end(show=show)
        self.flush(barchart=barchart, **kwargs)
