"""Base Garage Environment API."""

import abc


class Environment(abc.ABC):
    """The main API for garage environments.

    The main public API methods are:

        step
        reset
        render
        visualize
        close

    And set the following attributes:

        action_space: The action space specification
        observation_space: The observation space specification
        spec: The environment specifications
        render_modes: The list of supported render modes

    Make sure your environment is pickle-able:
        Garage pickles the environment via the `cloudpickle` module
        to save snapshots of the experiment. However, some environments may
        contain attributes that are not pickle-able (e.g. a client-server
        connection). In such cases, override `__getstate__()` and
        `__setstate()__` to add your custom pickle logic.

    """

    @property
    @abc.abstractmethod
    def action_space(self):
        """Return the action space specification.

        Some environments do not provide their action space specifications as
        type akro.Space. In such cases, subclasses should silently convert
        the environment's action_space to akro.Space.

        Returns:
            akro.Space: The envionrment specification.

        """

    @property
    @abc.abstractmethod
    def observation_space(self):
        """Return the observation space specification.

        Some environments do not provide their observation space
        specifications as type akro.Space. In such cases, subclasses should
        silently convert the environment's observation_space to akro.Space.

        Returns:
            akro.Space: The envionrment specification.
        """

    @property
    @abc.abstractmethod
    def spec(self):
        """Return the environment specification.

        Returns:
            garage.envs.env_spec.EnvSpec: The envionrment specification.

        """

    @property
    @abc.abstractmethod
    def render_modes(self):
        """Return the list of supported render modes.

        Returns:
            list: A list of string representing the supported render modes.
                  See render() for a list of modes.

        """

    @abc.abstractmethod
    def reset(self):
        """Resets the environment and returns the first `TimeStep` of sequence.

        Returns:
            garage.TimeStep: The first TimeStep.
                Note that this `TimeStep` will have attributes `observation`,
                `action`, `reward` as None as they are undefined.

        """

    @abc.abstractmethod
    def step(self, action):
        """Steps the environment using to the action and returns a `TimeStep`.

        If the environment returned the last time step of a sequence (either
        because of terminal or time out) at the previous step, this call to
        `step` will start a new sequence and `action` will be ignored.

        This method will also start a new sequence if called after the
        environment has been constructed and `reset()` has not been called.
        Again, in this case `action` will be ignored.

        If possible, update the visualization display as well.

        Args:
            action (np.ndarray): A NumPy array, or a nested dict, list or tuple
                of arrays corresponding to `action_spec()`.

        Returns:
            garage.TimeStep: The TimeStep after the action.

        """

    @abc.abstractmethod
    def render(self, mode):
        """Renders the environment.

        The set of supported modes varies per environment. By convention,
        if mode is:

        * rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        * ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).

        Make sure that your class's 'render_modes' includes the list of
        supported modes.

        Args:
            mode (str): the mode to render with. The string must be present in
            `self.render_modes`.

        For example:

        class MyEnv(Env):
            def render_modes(self):
                return ['rgb_array', 'ansi']

            def render(self, mode):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode == 'ansi':
                    ... # return text output
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an
                    exception
        """

    @abc.abstractmethod
    def visualize(self):
        """Creates a visualization of the environment.

        This function should be called only once after reset() to set up the
        visualization display. Update of the visualization should be done in
        `step()`.

        A typical workflow:

        env.reset() # reset the environment
        env.visualize() # open a window for visualization

        act_space = env.action_space
        action = act_space.sample() # sample a random action
        env.step(action) # step the environment
        # the display should be updated as well

        env.close() # close the environment and the visualization

        """

    @abc.abstractmethod
    def close(self):
        """Closes the environment.

        This method should close all windows invoked by `visualize()`.

        Override this function in your subclass to perform any necessary
        cleanup.

        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """

    def __del__(self):
        """Environment destructor."""
        self.close()
