import csv
from pathlib import Path
from typing import Dict


class BufferedFileLogger:
    def __init__(
            self,
            file_name,
            file_path='.',
            buffer_size=1000,
            header=("metric", "value", "global_step"),
            mode='a'
    ):
        self.file_path = Path(file_path)
        self.file_path.mkdir(parents=True, exist_ok=True)
        self.file_name = file_name
        self.buffer_size = buffer_size
        self.buffer = []

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
        self.buffer.append(args)
        if len(self.buffer) >= self.buffer_size:
            self._flush()

    def add_dict(self, data:Dict[str, float], global_step:int):
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

    def __repr__(self):
        s = f"BufferedFileLogger(file_path={self.file_path.absolute()}, file_name={self.file_name})\n"
        with open(self.file_path / self.file_name, 'r') as f:
            s += f.read()

        return s
