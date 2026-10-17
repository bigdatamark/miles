# [MODULE DOC] Thin singleton wrapper around the TensorBoard scalar writer. The first construction in a
# process creates the writer (directory from the TENSORBOARD_DIR environment variable, else
# tensorboard_log/project/experiment); later constructions return the same instance. The writer class
# is imported optionally and is none if the package is missing.
import datetime
import logging
import os
from miles.utils.misc import SingletonMeta

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

logger = logging.getLogger(__name__)


class _TensorboardAdapter(metaclass=SingletonMeta):
    # [CLASS DOC] HELPER: Singleton adapter holding the scalar writer (class-level default none until
    # initialized). The string literal after the writer attribute is not a real docstring.
    _writer = None

    """
    # Usage example: This will return the same instance every rank
    # tb = _TensorboardAdapter(args)  # Initialize on first call
    # tb.log({"Loss": 0.1}, step=1)

    # In other files:
    # from tensorboard_utils import _TensorboardAdapter
    # tb = _TensorboardAdapter(args)  # No parameters needed to get existing instance
    # tb.log({"Accuracy": 0.9}, step=1)
    """

    def __init__(self, args):
        # [FUNCTION DOC] HELPER: Asserts the use-tensorboard argument is truthy. If a project name is
        # given or the TENSORBOARD_DIR variable is set: when a project name exists but no experiment
        # name, the experiment name becomes the current local timestamp (year-month-day_hour-minute-
        # second); then initializes the writer. Otherwise raises a value error asking for the names or
        # the directory variable. Runs only on first construction (singleton).
        assert args.use_tensorboard, f"{args.use_tensorboard=}"
        tb_project_name = args.tb_project_name
        tb_experiment_name = args.tb_experiment_name
        if tb_project_name is not None or os.environ.get("TENSORBOARD_DIR", None):
            if tb_project_name is not None and tb_experiment_name is None:
                tb_experiment_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self._initialize(tb_project_name, tb_experiment_name)
        else:
            raise ValueError("tb_project_name and tb_experiment_name, or TENSORBOARD_DIR are required")

    def _initialize(self, tb_project_name, tb_experiment_name):
        # [FUNCTION DOC] HELPER: Chooses the log directory (TENSORBOARD_DIR if set, else a path built
        # from project and experiment names), creates it, logs the location and creates the writer.
        # Fails with a type error if the writer package was unavailable (class is none).
        """Actual initialization logic"""
        # Get tensorboard directory from environment variable or use default path
        tensorboard_dir = os.environ.get("TENSORBOARD_DIR", f"tensorboard_log/{tb_project_name}/{tb_experiment_name}")
        os.makedirs(tensorboard_dir, exist_ok=True)
        logger.info(f"Saving tensorboard log to {tensorboard_dir}.")
        self._writer = SummaryWriter(tensorboard_dir)

    def log(self, data, step):
        # [FUNCTION DOC] HELPER: Writes each entry of the mapping as a scalar under its key at the
        # step. Non-scalar values make the writer raise.
        """Log data to tensorboard

        Args:
            data (dict): Dictionary containing metric names and values
            step (int): Current step/epoch number
        """
        for key in data:
            self._writer.add_scalar(key, data[key], step)

    def finish(self):
        # [FUNCTION DOC] HELPER: Closes the writer (flushing pending events).
        """Close the tensorboard writer"""
        self._writer.close()
