import tkinter
from tkinter import ttk


class ProgressBar:
    """Class for generating a tkinter-based progress bar shown in an external
    window with a fancier design."""

    def __init__(
        self, n_steps: int, title: str, handle_n_exceeded: str = "warning"
    ) -> None:
        self.n_steps = n_steps
        self.title = title
        self.handle_n_exceeded = handle_n_exceeded

        self._step_n = 0
        self.step_increment = 100 / n_steps
        self.window = None
        self.percent_label = None
        self.progress_bar = None
        self.complete_label = None

        self._sort_inputs()
        self._create_bar()

    def _sort_inputs(self) -> None:
        """Sorts inputs to the object."""
        supported_handles = ["warning", "error"]
        if self.handle_n_exceeded not in supported_handles:
            raise NotImplementedError(
                "Error: The method for handling instances of the total number "
                f"of steps being exceeded '{self.handle_n_exceeded}' is not "
                f"supported. Supported inputs are {supported_handles}."
            )

    def _create_bar(self) -> None:
        """Creates the tkinter root object and progress bar window with the
        requested title and fancy style."""
        self.root = tkinter.Tk()
        self.root.wm_attributes("-topmost", True)
        self.root.title(self.title)

        # Create custom style for the progress bar
        style = ttk.Style()
        style.configure(
            "TProgressbar",
            thickness=30,
            troughcolor="gray",
            background="royalblue",
            barcolor="mediumseagreen",
        )

        # Progress bar widget
        self.progress_bar = ttk.Progressbar(
            self.root, length=350, mode="determinate", style="TProgressbar"
        )
        self.progress_bar.pack(padx=10, pady=20)

        # Percent label widget
        self.percent_label = tkinter.StringVar()
        self.percent_label.set(self.progress)
        ttk.Label(self.root, textvariable=self.percent_label, font=("Arial", 14)).pack()

        # Completion label widget (initially hidden)
        self.complete_label = ttk.Label(
            self.root, text="Process Completed!", font=("Arial", 16, "bold"), foreground="green"
        )
        self.complete_label.pack(pady=10)
        self.complete_label.pack_forget()  # Hide it initially

        self.root.update()

    @property
    def progress(self) -> str:
        """Getter for returning the percentage completion of the progress bar."""
        return f"{self.title}\n{str(int(self.progress_bar['value']))}% complete"

    @property
    def step_n(self) -> int:
        """Getter for the number of steps completed in the process."""
        return self._step_n

    @step_n.setter
    def step_n(self, value) -> None:
        """Setter for the number of steps completed in the process."""
        if value > self.n_steps:
            if self.handle_n_exceeded == "warning":
                print(
                    "Warning: The maximum number of steps in the progress bar "
                    "has been exceeded.\n"
                )
            else:
                raise ValueError(
                    "Error: The maximum number of steps in the progress bar "
                    "has been exceeded."
                )

        self._step_n = value

    def update_progress(self, n_steps: int = 1) -> None:
        """Increments the step number and updates the progress bar."""
        self.step_n += n_steps
        self.progress_bar["value"] += self.step_increment * n_steps
        self.percent_label.set(self.progress)
        self.root.update()

        # If the progress bar reaches 100%, show the completion label
        if self.progress_bar["value"] == 100:
            self.complete_label.pack()

    def close(self) -> None:
        """Closes the progress bar window."""
        self.root.destroy()