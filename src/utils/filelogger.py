import csv
from pathlib import Path
from typing import Dict

import pandas as pd


class BufferedFileLogger:
    def __init__(
            self,
            file_name,
            file_path='.',
            buffer_size=1000,
            header=["metric", "value", "global_step"],
            mode='a',
            postfix=None,
    ):
        self.file_path = Path(file_path)
        self.file_path.mkdir(parents=True, exist_ok=True)
        self.file_name = file_name
        self.buffer_size = buffer_size
        self.buffer = []

        self.postfix = postfix if postfix is not None else []

        # check if file exists
        if not (self.file_path / self.file_name).exists():
            mode = 'w'
            exists = False
        else:
            exists = True

        self.file = open(
            self.file_path / self.file_name,
            mode=mode,
            newline='',
            buffering=1  # Line buffering
        )

        self.writer = csv.writer(self.file)
        if not exists:
            # Write the header of the CSV file
            self.writer.writerow(header)

    def add_scalar(self, *args):
        args = args if isinstance(args, list) else list(args)
        self.buffer.append(args + self.postfix )
        if len(self.buffer) >= self.buffer_size:
            self._flush()

    def add_dict(self, data: Dict[str, float], global_step: int):
        """
        Add a dictionary of data to the buffer.
        """
        for key, value in data.items():
            self.buffer.append((key, value, global_step))
        if len(self.buffer) >= self.buffer_size:
            self._flush()

    def _flush(self):
        if self.buffer:
            self.writer.writerows(self.buffer)
            self.buffer = []

    def close(self):
        self._flush()
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        if exc_type is not None:
            print(f"Exception occurred: {exc_value}")

    def __repr__(self):
        s = f"BufferedFileLogger(file_path={self.file_path.absolute()}, file_name={self.file_name})\n"
        with open(self.file_path / self.file_name, 'r') as f:
            s += f.read()

        return s

    @property
    def dataframe(self):
        self._flush()  # Ensure latest data
        return pd.read_csv(self.file_path / self.file_name)

    def plot_scalar_curve(
            self, metric,
            x='global_step',
            y='value',
            title=None,
            plot=True,
            figsize=(12, 6),
            ax=None,
            label=None
    ):
        """Plot scalar metric curve from logged data.

        Args:
            metric: Key identifying metric to plot
            title: Plot title (optional)
            plot: If True displays plot immediately, if False returns axis object
            figsize: Tuple specifying figure dimensions (width, height)

        Returns:
            matplotlib.axes.Axes if plot=False, otherwise None
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        self._flush()  # Ensure latest data

        # Read and filter data
        df = self.dataframe.query('metric == @metric')

        if df.empty:
            raise ValueError(f"No data found for metric: {metric}")

        # Create plot
        if ax is None:
            plt.figure(figsize=figsize)

        if label is None:
            label = metric

        ax = sns.lineplot(
            data=df,
            x=x,
            y=y,
            errorbar=('ci', 95),  # Add confidence intervals
            estimator='mean',  # Aggregate if multiple runs exist
            ax=ax,
            label=label,
        )

        # Formatting
        ax.set(
            xlabel='Global Step',
            ylabel=metric.replace('_', ' ').title(),
            title=title or f'{metric} Development'
        )
        plt.tight_layout()

        return ax if not plot else plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a BufferedFileLogger instance
        logger = BufferedFileLogger(file_name='metrics.csv', file_path=temp_dir)

        # Simulate adding data
        for step in range(1, 101):
            logger.add_scalar('loss', step * 0.1, step)
            logger.add_scalar('accuracy', 0.5 + (step * 0.01), step)

        # Flush remaining data and close the logger
        logger.close()

        # Plot loss curve
        logger.plot_scalar_curve(metric="loss", title="Training Loss Curve")

        # Plot accuracy curve (example of returning axis object)
        ax = logger.plot_scalar_curve(metric="accuracy", plot=False)
        ax.set_title("Accuracy Curve (Returned Axes)")
        plt.show()
